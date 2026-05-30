"""
Pearl quantization plugin for vLLM.

Extends CompressedTensorsConfig to add support for:
- Mining layer (7-bit quantization + noisy GEMM) -> PearlScheme(mining_enabled=True)
- Non-mining layer (8-bit quantization + vanilla GEMM) -> PearlScheme(mining_enabled=False)

Both use pearl GEMM kernels and handle smooth_quant_scale internally.
"""

import multiprocessing

from miner_base.settings import MinerSettings
from miner_utils import get_logger

from .mining_state import (
    get_async_manager,
    init_async_manager,
    init_pinned_pool,
)
from .vllm_config import PearlConfig

_LOGGER = get_logger("vllm.pearl_miner")


def _is_vllm_worker() -> bool:
    # v1 engine: worker processes show up as "EngineCore_DP{rank}"
    # In multi-gpu setups, "VllmWorker-{number}" is used (has a different name in logs)
    #
    # vLLM-Omni runs the model in its own process types, which do NOT start with
    # "EngineCore"/"VllmWorker":
    #   - AR / LLM stages:  "StageEngineCoreProc" (and
    #     "StageEngineCoreProc_stage{id}_replica{id}[_DP{n}]")
    #   - diffusion stages: "StageDiffusionProc"
    # These are where the model forward (and thus the mineable GEMMs) run, so the
    # async manager + pinned pool must be initialized here too. (Omni loads the
    # plugin via the `vllm_omni.general_plugins` group inside these processes.)
    name = multiprocessing.current_process().name or ""
    return name.startswith(
        ("EngineCore", "VllmWorker", "StageEngineCoreProc", "StageDiffusionProc")
    )


def register_pearl_miner_layer() -> None:
    """
    Register the PearlMiner layer.
    The gateway socket path is loaded from the configuration file.
    """
    from vllm.model_executor.layers.quantization import register_quantization_config

    # Initialize the global mining state, but only if we're running in a vLLM *worker*
    # We only want to start threads or pre-allocate the pinned pool in workers.
    if _is_vllm_worker():
        init_async_manager()
        init_pinned_pool(get_async_manager()._conf.pinned_pool_size)
        init_plugin = not get_async_manager()._conf.no_vllm_plugin
    else:
        init_plugin = not MinerSettings().no_vllm_plugin

    if init_plugin:
        register_quantization_config("pearl")(PearlConfig)

        # Also register the diffusion (DiT) mining config when running under
        # vLLM-Omni, so `--quantization pearl_diffusion` mines on diffusion
        # transformers. Optional: skipped cleanly if vLLM-Omni is unavailable.
        try:
            from .diffusion_scheme import register_pearl_diffusion_config

            register_pearl_diffusion_config()
        except ImportError:
            _LOGGER.debug("vLLM-Omni not available; skipping pearl_diffusion registration")
