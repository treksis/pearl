"""
Pearl-GEMM Blackwell (sm120) stress test.

Hammers the compiled mining kernels under sustained load and reports
throughput, stability (no crash / NaN / context loss), and peak memory.

  1. Vanilla int8 GEMM  (pearl_gemm.gemm)        — inference path
  2. Noisy GEMM         (pearl_gemm.noisy_gemm)  — the mining kernel
"""
import os
import time

import torch

from pearl_gemm import gemm
from pearl_gemm.testing import GEMMParam, GemmTensorGenerator

DEV = "cuda"
torch.manual_seed(1)


def _stats(name, iters, seconds, flops_per_iter, bad):
    ips = iters / seconds
    tops = (flops_per_iter * iters) / seconds / 1e12
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"[{name}] iters={iters} time={seconds:.1f}s  {ips:.1f} it/s  "
          f"~{tops:.1f} TOPS(int8)  peak_mem={peak:.2f}GB  anomalies={bad}")
    return ips, tops


def stress_vanilla(M=4096, K=4096, N=4096, iters=300):
    A = torch.randint(-63, 64, (M, K), device=DEV, dtype=torch.int8)
    B = torch.randint(-63, 64, (N, K), device=DEV, dtype=torch.int8)
    As = (torch.rand(M, device=DEV) * 0.02 + 1e-3)
    Bs = (torch.rand(N, device=DEV) * 0.02 + 1e-3)
    C = torch.empty((M, N), dtype=torch.bfloat16, device=DEV)
    # warmup
    for _ in range(5):
        gemm(A=A, B=B, A_scales=As, B_scales=Bs, C=C,
             tile_size_m=128, tile_size_n=128, tile_size_k=64)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    bad = 0
    t0 = time.time()
    for i in range(iters):
        gemm(A=A, B=B, A_scales=As, B_scales=Bs, C=C,
             tile_size_m=128, tile_size_n=128, tile_size_k=64)
        if i % 50 == 0:
            torch.cuda.synchronize()
            if not torch.isfinite(C).all():
                bad += 1
    torch.cuda.synchronize()
    return _stats(f"vanilla {M}x{N}x{K}", iters, time.time() - t0, 2 * M * N * K, bad)


def stress_noisy(M=2048, K=2048, N=2048, R=128, iters=200):
    gp = GEMMParam(m=M, n=N, k=K, R=R)
    tg = GemmTensorGenerator(gp)
    tg.generate()  # easy/None pow target

    def one():
        from pearl_gemm import noisy_gemm
        noisy_gemm(
            A=tg.A, B=tg.B, EAL=tg.EAL, EAL_fp16=tg.EAL_fp16,
            EAR_R_major=tg.EAR_R_major, EBL_R_major=tg.EBL_R_major,
            EAR_K_major=tg.EAR_K_major, EBL_K_major=tg.EBL_K_major,
            EBR=tg.EBR, EBR_fp16=tg.EBR_fp16,
            AxEBL_fp16=tg.AxEBL_fp16, EARxBpEB_fp16=tg.EARxBpEB_fp16,
            ApEA=tg.ApEA, BpEB=tg.BpEB,
            A_scales=tg.A_scales, B_scales=tg.B_scales, C=tg.C,
            host_signal_header_pinned=tg.host_signal_header_pinned,
            host_signal_sync=tg.host_signal_sync,
            AxEBL_int32=tg.AxEBL_int32, EARxBpEB_int32=tg.EARxBpEB_int32,
            tile_size_m=gp.tile_size_m, tile_size_n=gp.tile_size_n, tile_size_k=gp.tile_size_k,
            pipeline_stages=gp.pipeline_stages,
            cluster_size_m=gp.cluster_size_m, cluster_size_n=gp.cluster_size_n,
            swizzle=gp.swizzle, swizzle_n_maj=gp.swizzle_n_maj,
            tile_size_m_noising_A=gp.tile_size_m_noising_A,
            tile_size_n_noising_B=gp.tile_size_n_noising_B,
            tile_size_k_noising_A=gp.tile_size_k_noising_A,
            tile_size_k_noising_B=gp.tile_size_k_noising_B,
            k_blocks_per_split_noising_A=gp.k_blocks_per_split_noising_A,
            k_blocks_per_split_noising_B=gp.k_blocks_per_split_noising_B,
            run_noising_A=not gp.skip_noising_a, run_noising_B=not gp.skip_noising_b,
            skip_reduction=gp.skip_reduction, skip_denoising=False,
            pow_target=tg.pow_target, pow_key=tg.pow_key,
        )

    for _ in range(5):
        one()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    bad = 0
    t0 = time.time()
    for i in range(iters):
        one()
        if i % 25 == 0:
            torch.cuda.synchronize()
            if not torch.isfinite(tg.C).all():
                bad += 1
    torch.cuda.synchronize()
    return _stats(f"noisy {M}x{N}x{K} R{R}", iters, time.time() - t0, 2 * M * N * K, bad)


if __name__ == "__main__":
    print("GPU:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability())
    vi = int(os.environ.get("VAN_ITERS", "300"))
    ni = int(os.environ.get("NOISY_ITERS", "200"))
    mem0 = torch.cuda.memory_allocated()
    stress_vanilla(iters=vi)
    stress_noisy(iters=ni)
    # leak check: allocated memory should return near baseline after both runs
    torch.cuda.empty_cache()
    leaked = (torch.cuda.memory_allocated() - mem0) / 1024**2
    print(f"residual allocated after runs: {leaked:.1f}MB (≈0 => no leak)")
    print("STRESS_OK")
