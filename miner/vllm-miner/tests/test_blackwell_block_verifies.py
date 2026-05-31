"""Regression: a block mined by the native fused kernel on consumer Blackwell
must pass the canonical Rust verify_plain_proof.

This guards the omni-pearl P0 fix: the consensus hash tile (cols_pattern) must
match the 2-row x 32-col fragment a bN=128 mma.sync thread opens under consumer
Blackwell's 99 KB SMEM cap. If cols_pattern is resized back to the upstream
64-entry (Hopper 128x256) pattern, the kernel opens only half the consensus tile
and this test fails — which is exactly the bug this fix resolved.

Requires a CUDA device and a native-Blackwell build (PEARL_GEMM_NATIVE_BLACKWELL,
i.e. an sm_12xf build). Skips otherwise.
"""
import base64
import time

import pytest
import torch

# A real incomplete block header (nbits -> 2**233 difficulty target).
HEADER_B64 = (
    "AAAAICcOqViZtp4qeqp5mnm0R3Yl7+D2GJ/5coUqW1OAhhKyiYZG4kAi7Bg+/A4"
    "40ZG+21mb+7on6Nyg4JUtABGkU8SLg39pAAABHg=="
)


def _native_blackwell_available() -> bool:
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability(0)[0] < 10:
        return False
    try:
        import pearl_gemm_cuda

        return bool(pearl_gemm_cuda.native_blackwell_build())
    except Exception:
        return False


@pytest.mark.skipif(
    not _native_blackwell_available(),
    reason="requires a CUDA native-Blackwell (sm_12xf) build",
)
def test_native_blackwell_block_passes_verify_plain_proof():
    from unittest.mock import patch

    from miner_base.gpu_matmul_config import GPUMatmulConfigFactory
    from miner_base.block_submission import create_proof
    from miner_base.settings import MinerSettings
    from pearl_gateway.comm.dataclasses import MiningJob
    from pearl_mining import IncompleteBlockHeader, verify_plain_proof
    from vllm_miner import gemm_operators
    from vllm_miner.mining_state import (
        delete_state,
        get_async_manager,
        init_async_manager,
        init_pinned_pool,
    )
    from vllm_miner.quantization_operators import quant_7bit

    hb = base64.b64decode(HEADER_B64)
    init_async_manager(MinerSettings(debug=True, no_gateway=True))
    init_pinned_pool()
    am = get_async_manager()
    am._conf.no_gateway = True
    am._conf.no_mining = False
    R, M, N, K = am._conf.noise_rank, 2048, 2048, 2048

    matmul_config = GPUMatmulConfigFactory.create(k=K, noise_rank=R)
    mining_job = MiningJob.from_dict(
        {"incomplete_header_bytes": HEADER_B64, "target": 2**232}
    )

    g = torch.Generator(device="cuda").manual_seed(1234)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, generator=g) * 0.1
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16, generator=g) * 0.05
    A, sa, _ = quant_7bit(x)
    B, sb, _ = quant_7bit(w)

    cap = {}
    try:
        with patch.object(am, "_client") as mc, patch.object(
            am, "handle_submit_block", side_effect=lambda obi, mj: cap.update(obi=obi)
        ):
            mc.get_mining_info.return_value = mining_job
            am._mining_job = mining_job
            gemm_operators.pearl_gemm_noisy(
                A.contiguous(),
                B.contiguous(),
                sa.squeeze(-1),
                sb.squeeze(-1),
                torch.bfloat16,
                submit_block=True,
            )
            torch.cuda.synchronize()
            for _ in range(200):
                if "obi" in cap:
                    break
                time.sleep(0.1)

        assert "obi" in cap, "native kernel found no block at the test target"
        obi = cap["obi"]
        ncols = len({int(i) for i in obi.B_column_indices})
        assert ncols == len(MinerSettings().cols_pattern), (
            f"kernel opened {ncols} cols but consensus cols_pattern has "
            f"{len(MinerSettings().cols_pattern)} — tile/pattern mismatch"
        )

        proof = create_proof(obi, hb)
        hdr = IncompleteBlockHeader.from_bytes(hb)
        ok, msg = verify_plain_proof(hdr, proof)
        assert ok, f"native-Blackwell block failed verify_plain_proof: {msg}"
    finally:
        delete_state()
