# Blackwell (sm120) Fused-Kernel Findings & Port-vs-Native Analysis

Empirical investigation of running Pearl's **fused** noisy-GEMM mining kernel
(PR #118 approach) natively on consumer Blackwell (NVIDIA RTX PRO 4000 Blackwell,
sm120 / cap 12.0), instead of PR #130's torch-`_int_mm` + CPU-PoW reference path.
All results produced on this GPU; methodology and limits stated honestly.

## TL;DR

- Forcing the fused kernel (bypassing PR #130's reference dispatch) showed the
  fused path **runs** on consumer Blackwell **only when built for the `sm_120f`
  family target** — not `sm_120` (TMA launch failure) and not even PR #118's
  intended `sm_120a`.
- On `sm_120f`: **vanilla GEMM correct (cos 0.999999), noising bit-exact vs the
  oracle (cos 1.0), in-kernel PoW signal fires** — but the **denoised output is
  wrong (cos 0.665)**. So PR #118's port is *incomplete*: a localized correctness
  bug in the main-GEMM/denoise epilogue.
- This is exactly why PR #130 fell back to CPU.

## How we got here (the dispatch shadow)

PR #118 (port the fused kernel: WGMMA→SM80 `mma.sync`, fit 99 KB SMEM) and
PR #130 (functional GB10: bypass kernels with torch `_int_mm` + CPU PoW) are
**independent** branches. PR #130 puts `if (use_reference_backend()) { ...; return; }`
(`dprops->major >= 10`) at the top of **every** C++ API entry, plus the Python
`_use_reference_cuda_backend(major>=10)` gates. When both are merged, **PR #130's
gate shadows PR #118's kernels** — on Blackwell the fused kernel is compiled but
never launched.

**Consequence (correction to earlier results):** our prior stress/E2E numbers on
sm120 measured PR #130's **torch-`_int_mm` reference path**, not the fused kernel.
The correctness (cos 0.99972) was real; the throughput was the reference path, not
the mining kernel, and the PoW was the gated CPU scan (not real mining).

To probe the fused kernel we added a diagnostic override `PEARL_GEMM_FORCE_KERNEL=1`
to all three gates.

## The arch discovery (sm_120 vs sm_120a vs sm_120f)

The kernel uses `SM90_TMA_LOAD`, which requires `CUTE_ARCH_TMA_SM90_ENABLED`.
From CUTLASS `cute/arch/config.hpp`:

| Build target | CUTLASS macro | `CUTE_ARCH_TMA_SM90_ENABLED`? | Result |
|---|---|---|---|
| `sm_120` (what the merge built) | `MMA_SM120_ENABLED` | ❌ (only `TMA_SM120`) | **launch failure** ("TMA Descriptor Prefetch without CUTE_ARCH_TMA_SM90_ENABLED") |
| `sm_120a` (PR #118's intent) | `MMA_SM120A_ENABLED` | ❌ | would also fail |
| **`sm_120f`** (family target) | `MMA_SM120F_ENABLED` | ✅ | **kernel runs** |

So the family arch `sm_120f` is the one that maps the SM90 TMA atoms onto consumer
Blackwell. `normalize_cuda_arch` was extended to accept the `f` suffix. (`128x256x128`
still can't run — 146 KB SMEM > 99 KB cap; `128x128x64` fits.)

## Layered correctness result on sm_120f

Fused kernel vs the reference oracle on identical inputs (`128x128x64`, R=128):

```
reference C  vs true GEMM : cos = 1.00000   (oracle correct)
fused C      vs true GEMM : cos = 0.66546   (fused denoised output WRONG)
ApEA fused   vs reference : cos = 1.00000   (noising A bit-exact)
BpEB fused   vs reference : cos = 1.00000   (noising B bit-exact)
in-kernel PoW (easy target): host_signal = kSignalTriggered  (signal path works)
```

| Layer | sm_120f status |
|---|---|
| Build / TMA | ✅ via `sm_120f` |
| Vanilla GEMM | ✅ cos 0.999999 |
| Noising (A+EA, B+EB) | ✅ bit-exact |
| In-kernel PoW signaling | ✅ fires |
| **Main GEMM + denoise epilogue** | ❌ **wrong (cos 0.665)** |

**The bug is localized** to the SM80 main-GEMM/denoise epilogue
(`collective_epilogue.hpp` / `collective_mainloop.hpp` `__CUDA_ARCH__ >= 1000`
branches; the accumulator-fragment remap `kernel_traits.hpp` claims is
"bit-identical" — empirically it is for noising, not for the denoise epilogue).

## Open, decisive question (not yet tested)

The **PoW transcript is the int32 noised product (ApEA@BpEB.T) before denoise**;
the denoise produces the fp `C` (the inference output). Since noising is bit-exact,
the transcript *may* be correct even though the denoised `C` is wrong — i.e. the
bug may be only in the denoise subtraction, not the consensus transcript.
**Decisive test (todo):** compare the in-kernel found indices/hash to the CPU
reference for the same easy target. If they match → in-kernel PoW is
consensus-valid (only inference is broken); if not → blocks are invalid.

---

# Port (fix PR #118) vs. native Blackwell rewrite — the real analysis

The instinct "just write Blackwell-specific code" is reasonable. Digging in, the
answer is nuanced and dominated by one constraint most people miss.

## 1. The MMA is already native
Consumer Blackwell (sm120) has **no WGMMA and no tcgen05** (tcgen05 is datacenter
sm100 only). Its int8 tensor-core path **is** `mma.sync` (the `SM80_16x8x32_S32S8S8S32_TN`
atom) — exactly what PR #118 uses. So a "native rewrite" would use the **same MMA
instruction**; there is no faster int8 MMA on sm120 to switch to. A rewrite buys
**zero** MMA performance.

## 2. The consensus constraint removes most of the freedom a rewrite would give
This is the crux. The PoW transcript must hash to **byte-identical** bits as the
canonical verifier (the node / the CPU reference) computes from the same opened
`A, B_t, indices`. The reduction order, partition structure (`rows_pattern`,
`cols_pattern`, `rank`), XOR-fold + rotate, and blake3 keying are **pinned by the
protocol**. A clean native kernel is **not free** to reorganize the
consensus-relevant accumulator traversal — if it produces a different transcript,
its blocks **don't verify**. PR #118's "ugly" WGMMA-layout emulation exists
*specifically* to preserve that bit-exact transcript. So the freedom a rewrite
seems to offer is largely the freedom you **cannot use**.

## 3. What a rewrite *would* legitimately improve
- **Tiling / SMEM / occupancy** designed for 99 KB from scratch (vs retrofitting
  Hopper's 228 KB tiles via the 2-phase denoise split).
- **Epilogue clarity / correctness** — avoiding the exact subtle layout bug we
  found, by deriving the accumulator→logical mapping natively for `mma.sync`.
- These are real but **incremental**, and concentrated in the *non-consensus* GEMM
  scaffolding.

## 4. The real hard part is not "a fast Blackwell GEMM"
A fast int8 GEMM on sm120 is routine. The hard part is **"a fast GEMM whose
accumulator reduction reproduces the canonical transcript byte-for-byte, and whose
host-signal path is correct."** That hard part is **identical** for the port and a
rewrite — both must match the reference oracle. So a rewrite does not dodge the
difficulty; it re-opens bit-exactness questions the port has *already closed*
(noising bit-exact, vanilla GEMM exact, PoW signaling working).

## 5. Performance ceiling reality
Even a perfect native sm120 kernel will be far below H100 in absolute hashrate
(fewer/smaller tensor cores, less bandwidth, no tcgen05). The economic case for
consumer Blackwell mining is **fleet of cheap cards**, not per-card peak.

## Recommendation

1. **Now (fastest to competitive): fix the port's localized denoise bug.** It's one
   epilogue path, with a byte-exact oracle (the reference) to validate against, and
   everything around it is proven correct. First settle the decisive question above
   (is the int32 transcript already bit-exact?) — if yes, mining may already be
   consensus-valid and only the inference output needs the denoise fix.
2. **For scale (right long-term architecture): per-arch native backends behind the
   adapter registry** (Hopper / Blackwell-consumer / Blackwell-DC / AMD), **each
   validated byte-exact against the one canonical CPU reference** via the conformance
   harness. "Write Blackwell-specific code" is correct at the *system* level — but
   it is an optimization/cleanliness play bounded by consensus exactness, **not** a
   shortcut around the bug, and **not** an MMA speedup.

**Net:** writing native Blackwell code is legitimate for the scalable system, but it
is *not* the fast path to working competitive mining and it does *not* escape the
one constraint that actually matters (byte-exact transcript). Fix the localized
epilogue bug first; build native backends second, gated by the conformance harness.

## Repro

```bash
PEARL_GEMM_CUDA_ARCHS=120f PEARL_GEMM_FORCE_BUILD=TRUE MAX_JOBS=4 NVCC_THREAD_COUNT=2 \
  uv pip install --no-build-isolation -e ./miner/pearl-gemm
PEARL_GEMM_FORCE_KERNEL=1 python <fused-vs-reference comparison>   # see findings above
```
