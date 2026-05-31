from pydantic_settings import BaseSettings, SettingsConfigDict


class MinerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="miner_")

    noise_range: int = 128
    noise_rank: int = 128
    idxs_per_col: int = 2

    # GEMM tile sizes. omni-pearl consensus tile is 128x128 (consumer Blackwell's
    # 99KB SMEM cannot fit the upstream 128x256). A bN=128 MMA thread opens a
    # 2-row x 32-col accumulator fragment, so the hash pattern below is sized to
    # exactly that fragment (see cols_pattern).
    tile_size_m: int = 128
    tile_size_n: int = 128
    tile_size_k: int = 128

    # fmt: off
    # Hash tile pattern for the omni-pearl 128x128 consensus tile.
    #
    # FORK CHANGE vs upstream Pearl: upstream uses a 64-entry cols_pattern (span
    # 0..249) sized for Hopper's 128x256 tile. Consumer Blackwell (sm120/121) caps
    # SMEM at 99KB and can only run a 128-wide tile, where one mma.sync thread
    # holds exactly 2 rows x 32 cols. The 64-entry pattern made the kernel open
    # only HALF the consensus tile (2x32), so its blocks failed canonical
    # verify_plain_proof. Sizing cols_pattern to the 32 columns a bN=128 thread
    # actually opens makes native Blackwell blocks consensus-valid (verified
    # end-to-end against the Rust verify_plain_proof + plonky2 ZK proof).
    rows_pattern: list[int] = [0, 8]
    cols_pattern: list[int] = [
    0, 1, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49, 56, 57,
    64, 65, 72, 73, 80, 81, 88, 89, 96, 97, 104, 105, 112, 113, 120, 121,
    ]
    # fmt: on

    pinned_pool_size: int = 128

    debug: bool = False
    print_header_hash: bool = False
    no_gateway: bool = False
    no_mining: bool = False
    skip_block_submission: bool = False
    no_vllm_plugin: bool = False
    quantization_fast_math: bool = False

    enable_async_cuda_event_processing: bool = True
