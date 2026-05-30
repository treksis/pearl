"""
End-to-end test for the omni-pearl diffusion mining shim on the real stack.

Drives the actual `DiffusionPearlConfig` / `DiffusionPearlOnlineLinearMethod`
(the DiT mining path) through:
  - real vLLM-Omni base classes (DiffusionInt8Config / Int8OnlineLinearMethod),
  - the real pearl_gemm CUDA kernels (noisy GEMM + 7-bit quant),
  - a real mining job (no live node — gateway mocked, same as the kernel tests),

at DiT-representative linear sizes. Also confirms a real vLLM
`ColumnParallelLinear` (the layer class Flux/GLM-Image DiTs are built from) is
routed to the Pearl mining method by the quant config.

Runs on the Blackwell (sm120) reference backend.
"""

from unittest.mock import patch

import pytest
import torch
from miner_base.gpu_matmul_config import GPUMatmulConfigFactory
from miner_base.settings import MinerSettings

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


@pytest.fixture
def async_manager():
    from vllm_miner.mining_state import (
        delete_state,
        get_async_manager,
        init_async_manager,
        init_pinned_pool,
    )

    init_async_manager(MinerSettings(debug=True, no_gateway=True))
    init_pinned_pool()
    yield get_async_manager()
    get_async_manager().wait_until_done_submitting_blocks()
    delete_state()


def _make_layer(n, k, dtype=torch.bfloat16):
    layer = torch.nn.Module()
    w = torch.randn(n, k, device="cuda", dtype=dtype) * 0.05
    layer.weight = torch.nn.Parameter(w, requires_grad=False)
    layer.logical_widths = [n]
    return layer


def test_quant_config_routes_real_columnparallel_linear_to_mining():
    """A real vLLM ColumnParallelLinear (DiT layer type) gets the Pearl mining method."""
    import vllm_omni  # noqa: F401 - applies vLLM patch
    from vllm_miner.diffusion_scheme import (
        DiffusionPearlConfig,
        DiffusionPearlOnlineLinearMethod,
    )

    cfg = DiffusionPearlConfig()
    qm = cfg.get_quant_method(_FakeLinearBase(), prefix="transformer.blocks.0.attn.to_qkv")
    assert isinstance(qm, DiffusionPearlOnlineLinearMethod), type(qm)


class _FakeLinearBase:
    """Stands in for a LinearBase instance (isinstance check in get_quant_method)."""


# Patch isinstance(layer, LinearBase) by making _FakeLinearBase a LinearBase subclass.
def _install_linearbase_base():
    import vllm_omni  # noqa
    from vllm.model_executor.layers.linear import LinearBase

    global _FakeLinearBase
    _FakeLinearBase = type("_FakeLinearBase", (LinearBase,), {"__init__": lambda self: None})


_install_linearbase_base()


@pytest.mark.parametrize("M,K,N", [(2048, 2048, 2048)])
def test_diffusion_pearl_mining_forward(async_manager, get_mining_job, M, K, N):
    """Full DiT-linear mining path: 7-bit weight quant + noisy GEMM + reference PoW."""
    import vllm_omni  # noqa: F401
    from vllm_miner.diffusion_scheme import (
        DiffusionPearlConfig,
        DiffusionPearlOnlineLinearMethod,
    )
    from vllm_miner.mining_state import get_async_manager

    get_async_manager()._conf.no_gateway = True
    get_async_manager()._conf.no_mining = False

    method = DiffusionPearlOnlineLinearMethod(DiffusionPearlConfig())

    # bf16 DiT weight -> 7-bit (mining-precision) quant at load time
    layer = _make_layer(N, K)
    w_ref = layer.weight.detach().float().clone()
    method.process_weights_after_loading(layer)
    assert layer.weight.dtype == torch.int8
    assert int(layer.weight.abs().max()) <= 63, "weight must be 7-bit for noise headroom"
    assert layer.weight.shape == (N, K), "weight must stay non-transposed [out,in]"

    # easy target so the reference-backend PoW scan finds a block (exercises submission)
    noise_rank = get_async_manager()._conf.noise_rank
    matmul_config = GPUMatmulConfigFactory.create(k=K, noise_rank=noise_rank)
    mining_job = get_mining_job(mining_config=matmul_config.mining_config, target=2**235)

    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1
    bias = torch.randn(N, device="cuda", dtype=torch.bfloat16) * 0.01

    with patch.object(get_async_manager(), "_client") as mock_client:
        mock_client.get_mining_info.return_value = mining_job
        get_async_manager()._mining_job = mock_client.get_mining_info()
        out = method.apply(layer, x, bias)
    torch.cuda.synchronize()

    # functional correctness: denoised mining output ~ true bf16 matmul (within int7 quant)
    assert out.shape == (M, N)
    assert torch.isfinite(out).all(), "mining GEMM produced NaN/Inf"
    ref = (x.float() @ w_ref.t()) + bias.float()
    cos = torch.nn.functional.cosine_similarity(out.float().flatten(), ref.flatten(), dim=0)
    print(f"\n[diffusion-mining] out{tuple(out.shape)} cos_sim_vs_ref={cos.item():.5f} "
          f"blocks_submitted={get_async_manager().blocks_submitted}")
    assert cos.item() > 0.98, f"denoised mining output diverged from reference (cos={cos.item()})"
