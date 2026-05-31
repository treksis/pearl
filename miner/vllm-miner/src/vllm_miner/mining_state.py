from __future__ import annotations

from miner_base.async_loop_manager import AsyncLoopManager
from miner_base.settings import MinerSettings
from miner_utils import get_logger
from pearl_gateway.config import MinerRpcConfig
from pearl_gemm import HostSignalHeaderPinnedPool

from .config import config

_LOGGER = get_logger("vllm.pearl_miner")

# Global mining state instances, should be initialized per-process
_async_manager: AsyncLoopManager | None = None
_pinned_pool: HostSignalHeaderPinnedPool | None = None


def get_async_manager() -> AsyncLoopManager:
    if not _async_manager:
        raise AssertionError("Async Loop Manager has not been initialized yet")
    return _async_manager


def _apply_blackwell_tile_caps(miner_settings: MinerSettings) -> None:
    """Cap the mining tile to fit consumer Blackwell's SMEM.

    Consumer Blackwell (sm120/sm121) caps shared memory at ~99 KB/CTA, which
    cannot fit the default 128x256x128 mining tile (~146 KB). The execution tile
    is NOT part of the consensus mining config (rows/cols pattern + rank + k), so
    a smaller tile is consensus-compatible (verified: a 128x128x64 tile produces
    blocks that verify against the canonical config). Datacenter parts with
    larger SMEM keep the default.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return
        props = torch.cuda.get_device_properties(0)
        smem = getattr(props, "shared_memory_per_block_optin", 0)
        # If SMEM is unknown (0), cap conservatively on Blackwell-family parts to
        # avoid a 146KB-tile launch failure; known large-SMEM parts (datacenter)
        # keep the default tile.
        consumer_blackwell = props.major >= 10 and (not smem or smem < 128 * 1024)
        tile_too_big = miner_settings.tile_size_n > 128 or miner_settings.tile_size_k > 64
        if consumer_blackwell and tile_too_big:
            _LOGGER.info(
                f"Consumer Blackwell SMEM cap ({smem // 1024}KB): clamping mining tile "
                f"{miner_settings.tile_size_m}x{miner_settings.tile_size_n}x"
                f"{miner_settings.tile_size_k} -> 128x128x64 (execution-only, consensus-safe)"
            )
            miner_settings.tile_size_m = 128
            miner_settings.tile_size_n = 128
            miner_settings.tile_size_k = 64
    except Exception as e:
        _LOGGER.warning("Blackwell tile-cap check failed: %r", e)


def init_async_manager(miner_settings: MinerSettings | None = None) -> None:
    """Initialize the global mining state."""
    global _async_manager

    if _async_manager is None or _async_manager._pool is None:
        miner_settings = miner_settings if miner_settings is not None else MinerSettings()
        miner_settings.enable_async_cuda_event_processing = True
        _apply_blackwell_tile_caps(miner_settings)

        _async_manager = AsyncLoopManager(
            MinerRpcConfig(transport="uds", socket_path=config.gateway_socket_path),
            miner_settings,
        )
        _async_manager.start()
        config.settings = miner_settings
        _LOGGER.info(f"Mining state initalized, {miner_settings=}")


def get_pinned_pool() -> HostSignalHeaderPinnedPool:
    global _pinned_pool

    if _pinned_pool is None:
        raise AssertionError("Pinned pool has not been initialized yet")
    return _pinned_pool


def init_pinned_pool(pool_size: int = 128) -> None:
    global _pinned_pool

    if _pinned_pool is None:
        _pinned_pool = HostSignalHeaderPinnedPool(pool_size)
        _LOGGER.info(f"Pinned pool initialized, {pool_size=}")


def delete_state() -> None:
    global _async_manager
    global _pinned_pool

    if _async_manager is not None:
        _async_manager.wait_until_done_submitting_blocks()
        _async_manager.stop()
        del _async_manager
        _async_manager = None

    if _pinned_pool is not None:
        del _pinned_pool
        _pinned_pool = None
