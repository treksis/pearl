"""
Pearl mining quantization for vLLM-Omni diffusion (DiT) transformers.

This mirrors vLLM-Omni's own ``DiffusionInt8Config`` online path, but routes the
heavy DiT linear GEMMs through Pearl's noisy GEMM so that diffusion inference
doubles as proof-of-useful-work mining.

The proof machinery is *not* re-implemented here: the commitment hashing, PoW
target check, block-found signalling and async submission all live in
``gemm_operators.pearl_gemm_noisy`` (the exact same code the LLM mining path
uses). Only the layer plumbing differs:

- Activation: int7 dynamic per-token (mining precision), like the LLM mining
  layers.
- Weight: online int8, quantized from the bf16/fp16 checkpoint at load time, and
  kept **non-transposed** ``[out, in]`` because the Pearl GEMM expects that
  layout (vLLM's vanilla int8 kernel transposes to ``[in, out]``; we must not).

Only the ``LinearBase`` layers of a DiT are hooked (QKV / attention-out / MLP);
the small raw ``nn.Linear`` tail (e.g. final patch->pixel projection) is left to
its dtype and never mined. Heavy compute is in the hooked layers.

Activated by serving an omni diffusion model with ``--quantization
pearl_diffusion`` (online quant of a bf16/fp16 checkpoint).
"""

from typing import Optional

import torch
from miner_utils import get_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.fp8 import _copy_missing_attrs
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.model_loader.weight_utils import initialize_single_dummy_weight
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm_omni.quantization.int8_config import (
    DiffusionInt8Config,
    Int8OnlineLinearMethod,
)

from .config import config
from .gemm_operators import pearl_gemm_noisy, pearl_gemm_vanilla
from .mining_state import get_async_manager
from .quantization_operators import quant_7bit

_LOGGER = get_logger("vllm.pearl_miner")

# Method name registered into vLLM's quantization registry; selected via
# ``--quantization pearl_diffusion``.
PEARL_DIFFUSION_METHOD = "pearl_diffusion"


class _DiffusionPearlApplyMixin:
    """Shared apply path: int7-quantize the activation and run the Pearl GEMM.

    Mirrors ``vllm_kernels.PearlKernel._apply_weights_mining`` but adds the
    activation reshape and bias handling that DiT linear layers need (the
    CompressedTensors LLM layers are bias-free, so the LLM path omits both).
    """

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ori_shape = x.shape
        in_dtype = x.dtype
        # Pearl GEMM only emits bf16/fp16; quantize/compute in a supported dtype
        # and cast the result back to the caller's dtype.
        gemm_dtype = in_dtype if in_dtype in (torch.bfloat16, torch.float16) else torch.bfloat16

        x2d = x.reshape(-1, ori_shape[-1])
        if x2d.dtype is not gemm_dtype:
            x2d = x2d.to(gemm_dtype)

        w_q = layer.weight  # [n, k] int8, non-transposed
        w_s = layer.weight_scale  # [n, 1] fp32 (per output channel)

        # int7 dynamic per-token quantization of the activation.
        x_q, x_s, _ = quant_7bit(x2d)
        m, k, n = x_q.shape[0], x_q.shape[1], w_q.shape[0]

        scale_a = x_s.squeeze(-1)
        scale_b = w_s.squeeze(-1)

        if config.should_use_noisy_gemm(m, n, k) and not config.settings.no_mining:
            out = pearl_gemm_noisy(
                x_q.contiguous(),
                w_q.contiguous(),
                scale_a=scale_a,
                scale_b=scale_b,
                out_dtype=gemm_dtype,
                layer=layer,
                submit_block=not get_async_manager()._conf.skip_block_submission,
            )
        else:
            out = pearl_gemm_vanilla(
                x_q.contiguous(),
                w_q.contiguous(),
                scale_a=scale_a,
                scale_b=scale_b,
                out_dtype=gemm_dtype,
            )

        out = out.reshape(*ori_shape[:-1], out.shape[-1])
        if bias is not None:
            out = out + bias.to(out.dtype)
        return out.to(in_dtype)


class DiffusionPearlOnlineLinearMethod(_DiffusionPearlApplyMixin, Int8OnlineLinearMethod):
    """Online quantization: load bf16/fp16, quantize to int8 at load time.

    Reuses the lazy/meta weight-loading machinery from ``Int8OnlineLinearMethod``
    (via ``LazyWeightMixin.create_weights``) and only changes the weight layout
    (kept non-transposed for the Pearl GEMM).
    """

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        # Materialize a meta-device weight just-in-time (same as the int8 online
        # path) before quantizing.
        if layer.weight.device == torch.device("meta"):
            weight = ModelWeightParameter(
                data=torch.empty_like(layer.weight, device=layer._load_device),
                input_dim=1,
                output_dim=0,
                weight_loader=layer.weight.weight_loader,
            )
            _copy_missing_attrs(layer.weight, weight)
            layer.register_parameter("weight", weight)
            initialize_single_dummy_weight(layer.weight)

        # 7-bit (NOT 8-bit) symmetric per-output-channel weight quantization.
        # The noisy GEMM adds integer noise to the weight in int8 space
        # (BpEB = B + EB); Pearl's mining layers use 7-bit (|v| <= 63) precisely
        # so that B + noise cannot overflow int8. Full int8 weights would wrap
        # and corrupt both the inference output and the proof-of-work. This
        # matches the LLM mining layers (7-bit weight + 7-bit activation).
        # quantize_kernel quantizes per row; for a [n, k] weight that is per
        # output channel, yielding qweight [n, k] int8 and weight_scale [n, 1].
        qweight, weight_scale, _ = quant_7bit(layer.weight)

        # Pearl GEMM expects the weight NON-transposed: [out, in] == [n, k].
        # (Int8OnlineLinearMethod stores qweight.t() for the vanilla kernel.)
        replace_parameter(layer, "weight", torch.nn.Parameter(qweight.data, requires_grad=False))
        replace_parameter(layer, "weight_scale", torch.nn.Parameter(weight_scale.data, requires_grad=False))

        layer._already_called_process_weights_after_loading = True


class DiffusionPearlConfig(DiffusionInt8Config):
    """Pearl mining quant config for diffusion transformers.

    Inherits the int8 config's parsing/loading contract; only changes the method
    name and which linear method is returned for ``LinearBase`` layers.
    """

    @classmethod
    def get_name(cls) -> str:  # type: ignore[override]
        return PEARL_DIFFUSION_METHOD

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            if not current_platform.is_cuda():
                raise NotImplementedError("DiffusionPearlConfig requires CUDA (Pearl GEMM is Hopper+).")
            if self.is_checkpoint_int8_serialized:
                # Offline int8-serialized checkpoints are not produced for Pearl
                # mining yet; the online path quantizes a bf16/fp16 checkpoint.
                raise NotImplementedError(
                    "DiffusionPearlConfig only supports online quantization of bf16/fp16 "
                    "checkpoints; serialized int8 checkpoints are not supported."
                )
            return DiffusionPearlOnlineLinearMethod(self)
        return None


def register_pearl_diffusion_config() -> None:
    """Register ``pearl_diffusion`` into vLLM's quantization registry (idempotent).

    Imports are deferred to call time so importing this module never pulls vLLM
    in on its own. Safe to call in every process/worker.
    """
    from vllm.model_executor.layers.quantization import (
        QUANTIZATION_METHODS,
        register_quantization_config,
    )

    if PEARL_DIFFUSION_METHOD in QUANTIZATION_METHODS:
        return
    register_quantization_config(PEARL_DIFFUSION_METHOD)(DiffusionPearlConfig)
