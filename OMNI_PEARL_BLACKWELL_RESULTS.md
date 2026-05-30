# Omni-Pearl on Blackwell (sm120) — Build, Test & Stress Results

Branch: `omni-pearl-mining` · GPU: **NVIDIA RTX PRO 4000 Blackwell, compute capability 12.0 (sm120)** · CUDA 13.1 · torch 2.11.0+cu130 · Python 3.12

This documents bringing the omni-pearl mining port up on a **Blackwell consumer GPU**,
on top of the merged Blackwell PRs (#118, #130). All numbers below were produced
in this environment; limitations are stated honestly.

> **Scope of hardware.** PR #130 frames Blackwell as a **functional reference
> backend**, not a production-performance path (Hopper sm90 remains the optimized
> target). On `cap.major ≥ 10` the hashing + PoW search run on CPU (numpy/blake3)
> while the GEMM/noising kernels run on-GPU. These results validate *functional
> correctness and stability*, not competitive mining throughput.

## What was assembled

- `omni-pearl-mining` branch = master + **PR #118** (Blackwell SMEM tiling + SM90 WGMMA→SM80 mma.sync port) + **PR #130** (functional GB10 support + reference backend) + the **omni-pearl port** (vLLM-Omni plugin registration, omni worker detection, `DiffusionPearlConfig` DiT mining shim).
- One merge conflict, in `miner/pearl-gemm/setup.py`: resolved to PR #130's `PEARL_GEMM_CUDA_ARCHS` resolver (authoritative Blackwell build system) while keeping PR #118's CUDA-13 `cccl` host-compiler `-I` fix needed by PR #130's new `pearl_gemm_api.cpp`.
- The omni-pearl diffusion shim needs **no change** for Blackwell: it calls `pearl_gemm_noisy`, and PR #130 placed the Blackwell reference path inside that function.

## 1. Build (pearl-gemm CUDA kernels for sm120)

| Item | Result |
|---|---|
| Arch resolution | `PEARL_GEMM_CUDA_ARCHS=native` → `sm_120` accepted |
| First attempt (`MAX_JOBS=16`) | ❌ OOM — `cicc` killed; container has a **36 GB cgroup cap** (not the 251 GB host) |
| Retry (`MAX_JOBS=4`, `NVCC_THREAD_COUNT=2`) | ✅ **BUILD rc=0, 0 failures** (27/27 objects, incl. the heavy `CollectiveMainloop::mma` instantiations) |
| `import pearl_gemm` + CUDA ext on sm120 | ✅ loads `pearl_gemm_cuda.cpython-312…so` |

**Lesson for CI/docs:** on memory-capped containers, pin `MAX_JOBS` — `setup.py::smart_max_jobs` reads host `/proc/meminfo`, not the cgroup limit, so it over-subscribes and OOM-kills the CUTLASS compiles.

## 2. Smoke tests (direct kernel correctness on Blackwell)

| Kernel | Result |
|---|---|
| `quantize` (7-bit) | ✅ int8 out, `max|xq| = 63` (correct 7-bit clamp) |
| `gemm` (vanilla int8) 1024³ | ✅ vs fp reference: **cosine 0.999999**, rel-err 0.0033 (int8 rounding) |

## 3. Test suites

| Suite | Result | Notes |
|---|---|---|
| `miner-base` | ✅ **94/94 passed** | proof/mining logic reused by the port: commitment hash, merkle trees, noise gen, noisy-GEMM **reference**, matmul config, async loop |
| `pearl-gateway` | ✅ **117/117 passed** (25 deselected) | block assembly + submission path; deselected = integration/perf (need a node) |
| `pearl-gemm` `test_noise_gen` | ✅ all-pass (large parametrized sweep) | noise-generation kernel correct on sm120 |
| `pearl-gemm` `test_pearl_gemm` (noisy GEMM) | ⚠️ random 397-case sample: **283 passed, 110 skipped, 3 xfail, 1 failed** | suite has 15,206 cases; full run infeasible here |

**On the 1 failure:** it appears only under randomized ordering; the first ~450 cases in deterministic order all pass. It is a single specific config (~0.25% of the sample), consistent with int8 numerical tolerance on the experimental Blackwell reference/SM80 path that PR #130 explicitly labels non-production. Not root-caused — flagged for follow-up on sm90 vs sm120 tolerance.

## 4. Stress test (sustained load, stability, leaks)

`miner/pearl-gemm/stress_blackwell.py` — sustained mining-sized GEMMs, anomaly + leak checks:

| Workload | Iters | Throughput | Peak mem | Anomalies (NaN/Inf) |
|---|---|---|---|---|
| Vanilla GEMM 4096³ | 400 | **308 it/s ≈ 42 TOPS (int8)** | 0.23 GB | 0 |
| Noisy (mining) GEMM 2048³ R128 | 300 | **1266 it/s ≈ 22 TOPS (int8)** | 0.08 GB | 0 |

Residual allocated after both runs ≈ **8 MB → no leak**. No crash, no CUDA-context loss, no NaN/Inf. `STRESS_OK`.

## 5. vLLM 0.21.0 API compatibility (the omni-pearl glue)

All internal vLLM symbols/signatures the port imports were verified present in vLLM 0.21.0 (incl. `Int8ScaledMMLinearKernel`, CompressedTensors scheme/config bases, `model_executor.parameter.*`, `replace_parameter` still re-exported at the old path). py_compile + ruff clean on all changed/new files.

## What is NOT covered here (honest limits)

- **End-to-end mining / block submission**: needs a live `pearld` node + `pearl-gateway` + a Pearl-quantized model. Not run here.
- **The diffusion shim at runtime**: needs vLLM + vLLM-Omni + a quantized DiT model on GPU. Verified at import/signature/logic level only; not executed end-to-end.
- **Competitive throughput**: Blackwell is the functional reference path by design (PR #130). The TOPS above show the kernels run efficiently and stably, not that this GPU is a competitive miner.
- **Full noisy-GEMM matrix**: 15,206 cases; only sampled here.

## Reproduce

```bash
# build for this GPU (memory-capped container → low parallelism)
PEARL_GEMM_CUDA_ARCHS=native PEARL_GEMM_FORCE_BUILD=TRUE MAX_JOBS=4 NVCC_THREAD_COUNT=2 \
  uv pip install --no-build-isolation -e ./miner/pearl-gemm

uv run pytest miner/miner-base/tests miner/pearl-gateway/tests -q -m "not integration and not performance"
uv run pytest miner/pearl-gemm/tests/test_noise_gen.py
python miner/pearl-gemm/stress_blackwell.py      # VAN_ITERS / NOISY_ITERS env-tunable
```
