"""
Empirically validate PR #118's FUSED kernel + in-kernel PoW on Blackwell sm120,
by forcing PEARL_GEMM_FORCE_KERNEL=1 (bypasses PR #130's reference dispatch).

Answers: does the native fused kernel run correctly, and — the open question —
does the IN-KERNEL PoW (check_pow_target + write_host_signal_header) actually
fire and signal on consumer Blackwell?
"""
import os

os.environ["PEARL_GEMM_FORCE_KERNEL"] = "1"  # read at call time by both C++ and Python gates

import torch

from pearl_gemm import gemm, noisy_gemm, get_host_signal_header, HostSignalStatus
from pearl_gemm.testing import GEMMParam, GemmTensorGenerator

dev = "cuda"
torch.manual_seed(1)
print("PEARL_GEMM_FORCE_KERNEL =", os.environ["PEARL_GEMM_FORCE_KERNEL"])
print("GPU:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability())

# ---------- 1) fused vanilla GEMM correctness ----------
M = K = N = 2048
A = torch.randint(-63, 64, (M, K), device=dev, dtype=torch.int8)
B = torch.randint(-63, 64, (N, K), device=dev, dtype=torch.int8)
As = torch.rand(M, device=dev) * 0.02 + 1e-3
Bs = torch.rand(N, device=dev) * 0.02 + 1e-3
C = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
gemm(A=A, B=B, A_scales=As, B_scales=Bs, C=C, tile_size_m=128, tile_size_n=128, tile_size_k=64)
torch.cuda.synchronize()
ref = (A.float() @ B.float().T) * As[:, None] * Bs[None, :]
cos_v = torch.nn.functional.cosine_similarity(C.float().flatten(), ref.flatten(), dim=0).item()
print(f"[1 fused vanilla GEMM]  cos_vs_ref={cos_v:.6f}  -> {'OK' if cos_v > 0.99 else 'MISMATCH'}")


def run_noisy(tg, gp):
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


# ---------- 2) fused noisy GEMM denoised correctness (HARD target: no false hit) ----------
gp = GEMMParam(m=M, n=N, k=K, R=128, tile_size_m=128, tile_size_n=128, tile_size_k=64, pipeline_stages=2)
tg = GemmTensorGenerator(gp)
tg.generate(pow_target=2**230)  # hard -> denoised C must still equal the true GEMM
run_noisy(tg, gp)
torch.cuda.synchronize()
ref2 = (tg.A.float() @ tg.B.float().T) * tg.A_scales.view(M, 1) * tg.B_scales.view(1, N)
cos_n = torch.nn.functional.cosine_similarity(tg.C.float().flatten(), ref2.flatten(), dim=0).item()
hdr_hard = get_host_signal_header(tg.host_signal_header_pinned)
print(f"[2 fused noisy GEMM ]  cos_vs_ref={cos_n:.6f}  -> {'OK' if cos_n > 0.99 else 'MISMATCH'}"
      f"   host_signal(hard)={hdr_hard.status}")

# ---------- 3) THE KEY TEST: in-kernel PoW fires on an EASY target ----------
gp3 = GEMMParam(m=M, n=N, k=K, R=128, tile_size_m=128, tile_size_n=128, tile_size_k=64, pipeline_stages=2)
tg3 = GemmTensorGenerator(gp3)
tg3.generate(pow_target=2**255)  # trivially easy -> in-kernel check_pow_target should trigger
run_noisy(tg3, gp3)
torch.cuda.synchronize()
hdr = get_host_signal_header(tg3.host_signal_header_pinned)
triggered = hdr.status == HostSignalStatus.kSignalTriggered
print(f"[3 IN-KERNEL PoW    ]  host_signal.status={hdr.status}  triggered={triggered}  "
      f"-> {'IN-KERNEL PoW WORKS on sm120' if triggered else 'in-kernel PoW did NOT signal'}")
print("DONE")
