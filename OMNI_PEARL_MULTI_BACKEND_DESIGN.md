# Omni-Pearl — Competitive Mining & a Multi-Backend PoW-GEMM Architecture

Design notes (mental model, not an implementation). Covers: what "competitive
mining" requires, the GPU-architecture landscape, and how to structure the
proof-of-work GEMM so it scales across vendors/architectures (Hopper, Blackwell
datacenter, Blackwell consumer, AMD) the way a plugin/adapter system does —
without ever forking the consensus-critical math.

---

## 1. What "competitive mining" actually requires

Pearl mining yield ≈ **PoW attempts/sec** = (mineable matmuls/sec) × (candidates/matmul).
Two layers must run at tensor-core speed:

1. **GEMM + noising + denoising** — already runs on every backend we've tried
   (verified correct on sm120: denoised cos-sim 0.99972).
2. **PoW search** — hash the noised "transcript", compare to target, signal a hit.
   This is the layer that is *fused into the GEMM epilogue on Hopper* but
   currently **falls back to CPU on consumer Blackwell** (PR #130).

The CPU fallback is **gated off at real difficulty** (`expected_hits ≥ 0.25`),
so at mainnet difficulty it finds **nothing**. Competitive mining therefore means:
**the PoW search must run in-kernel (or in a fused GPU pass) on the target hardware.**

---

## 2. Why it isn't "one kernel for all" — the architecture landscape

| Hardware | Compute cap | Int8 MMA | Async copy | SMEM | In-kernel PoW today |
|---|---|---|---|---|---|
| H100 / H200 (Hopper) | sm90 | **WGMMA** (warpgroup) | TMA + clusters | 228 KB | ✅ works |
| B200 / GB200 (Blackwell DC) | sm100 | **tcgen05** (tensor mem) | TMA + clusters | 228 KB+ | needs MMA adapt |
| RTX 6000 Pro / 5090 (Blackwell consumer) | sm120 | **mma.sync** (per-warp) | partial | 99 KB | ⚠️ GEMM works, PoW = CPU |
| MI300 / RDNA (AMD) | CDNA/RDNA | **MFMA / WMMA** (HIP) | own DMA | varies | ❌ not implemented |

**Key fact:** RTX 6000 Pro Blackwell is **sm120 — the same arch we already
validated**, just 96 GB VRAM. It is *not* the datacenter part; B200 is (sm100).

The matmul instruction *and the layout of its accumulator fragments* differ per
arch. The PoW "transcript" is extracted by reading those raw accumulator
fragments — so **that one extraction step is the only thing that is genuinely
arch-specific in the PoW path.** Everything else is portable integer math.

---

## 3. The mental model: split invariant core from backend mechanics

Three layers, hard boundary between them:

```
┌─────────────────────────────────────────────────────────────┐
│  WORKLOAD ADAPTERS  (which matmuls to mine — already plugins) │
│  vLLM-AR · vLLM-Omni-diffusion · raw GEMM · …                 │
└───────────────────────────┬───────────────────────────────────┘
                            │  logical call: noisy_gemm(A,B,scales,target,key)
┌───────────────────────────▼───────────────────────────────────┐
│  PoW-GEMM BACKEND INTERFACE  (the contract — speaks matrices,  │
│  scales, target, key; NEVER fragment layouts)                  │
│   • capabilities(): dtypes, min M/N/K, SMEM, in_kernel_pow?    │
│   • noisy_gemm(...) -> (C, host_signal)                        │
│   • quantize / noise_gen / tensor_hash                         │
└───────────────────────────┬───────────────────────────────────┘
             ┌───────────────┼───────────────┬───────────────┐
┌────────────▼───┐ ┌─────────▼────┐ ┌─────────▼────┐ ┌────────▼──────┐
│ HopperBackend  │ │ Blackwell-DC │ │ Blackwell-   │ │ ROCmBackend   │
│ WGMMA+TMA      │ │ tcgen05      │ │ consumer     │ │ MFMA/WMMA     │
│                │ │              │ │ mma.sync     │ │ (HIP/CK)      │
└───────┬────────┘ └──────┬───────┘ └──────┬───────┘ └──────┬────────┘
        └─────────────────┴──── shared ────┴────────────────┘
              SHARED DEVICE PoW-CORE  (one source of truth)
        jackpot fold · blake3 · target compare · header bytes
                          ▲
                          │ must match, byte-for-byte
              ┌───────────┴────────────┐
              │  REFERENCE BACKEND (CPU)│  = the executable SPEC / oracle
              └─────────────────────────┘
```

### 3a. The invariant core (write once, never fork)
The consensus-critical computation — noise generation from the commitment,
the jackpot accumulation (XOR-fold + rotate), blake3, the exact byte order
hashed, and the target comparison — is **pure integer/bitwise math**. It is
**identical on every backend or blocks won't verify on the same chain.** Express
it once as a backend-agnostic device header (`#include`d into each backend's
epilogue) *and* as the CPU reference. The CPU reference is the **spec**: it's
already in the repo (PR #130) and doubles as the conformance oracle.

### 3b. The backend-specific sliver
What each backend must supply, and *only* this:
1. The **int8 MMA atom** (WGMMA / tcgen05 / mma.sync / MFMA).
2. The **accumulator-fragment → logical (row,col) mapping** so the transcript is
   read in the canonical order before hashing. ← *this is the actual sm120 gap.*
3. The **async load / SMEM tiling** (TMA+clusters, cp.async, or AMD DMA).
4. The **host-signal** mechanism (mapped pinned mem write on a hit).

Everything downstream of the fragment read is the shared core.

### 3c. The decisive insight
Don't hand-write the extraction (`__byte_perm` selectors) per arch. **Express it
in terms of the MMA atom's thread-value (TV) layout** — CuTe/CUTLASS already
carries this layout for each atom. Then "add a backend" = "provide its MMA atom +
copy atom"; the epilogue, transcript extraction, and PoW core are *reused
unchanged*. This is exactly how CUTLASS scales one kernel across architectures.
AMD's analog is Composable Kernel (CK), which has the same layout abstraction.

---

## 4. Why this is an adapter/plugin system (your ComfyUI / device-backend intuition)

There are **two orthogonal plugin axes**, and they should never be entangled:

- **Axis A — workload** (where the GEMMs come from): vLLM AR, vLLM-Omni diffusion,
  SGLang, raw bench. *We already built this axis* (the quant-config plugins).
- **Axis B — hardware backend** (how PoW runs on this silicon): Hopper, Blackwell-DC,
  Blackwell-consumer, ROCm, CPU-reference. *This doc is about Axis B.*

This mirrors how vLLM has `current_platform` (CUDA/ROCm/XPU/NPU) *and* a
quantization-method registry, independently. Pearl already has the seed of Axis B:
`_use_reference_cuda_backend(major>=10)`. The scalable move is to **generalize that
ad-hoc check into a real backend registry + dispatch**:

- **Detect → select**: at startup, probe the device (vendor, compute cap, SMEM,
  available MMA) and pick the best registered backend. Same shape as an auth
  provider registry or a torch device dispatcher.
- **Capability negotiation**: each backend advertises `capabilities()`. The caller
  asks "can you do in-kernel PoW for this (M,N,K,dtype)?" and routes accordingly.
- **Graceful degradation chain** (the ComfyUI behavior): fused-in-kernel PoW →
  separate GPU PoW pass → CPU reference. Correctness is *always* preserved because
  every tier validates against the same spec; only speed degrades. A miner on an
  unsupported card still produces *valid* blocks (just slowly), never invalid ones.

---

## 5. The consensus safety net: differential conformance testing

The thing that makes a multi-backend system *safe* to grow: a **conformance
harness** that runs a fixed corpus of (A, B, header, target) vectors through the
**CPU reference** and through **each backend**, asserting **byte-exact** equality
of: commitment hashes, noise, denoised C, transcript, jackpot hash, and
opened-block indices. A backend is "consensus-valid" iff it passes. This is the
gate for adding Blackwell-consumer, B200, or AMD without risking a fork. It's
cheap to run and is the single most important artifact.

---

## 6. Roadmap & effort (per backend, behind the same interface)

1. **Backend registry + capability + fallback dispatch** — refactor the existing
   `_use_reference_cuda_backend` into Axis B. Small, no kernel work. *Do first.*
2. **Conformance harness** vs the CPU reference. Small. *Do first — it gates the rest.*
3. **Blackwell-consumer (sm120) in-kernel PoW** — the focused fix: remap transcript
   extraction to the `mma.sync` accumulator layout; validate byte-exact. **No
   datacenter hardware needed** — develop on the sm120 we have / RTX 6000 Pro.
   *This is the highest-leverage step for "cheap consumer cards as miners".*
4. **Blackwell-DC (sm100)** — swap WGMMA→tcgen05 atom; TMA/clusters/SMEM already
   present. Medium; needs a B200.
5. **AMD (CDNA/RDNA)** — new HIP/CK backend: MFMA/WMMA atom + its fragment mapping +
   AMD DMA + signal. Largest, but **fully additive** — the core + harness are reused.

Hopper (sm90) already works and is the reference for "correct in-kernel PoW".

---

## 7. TL;DR

- **Competitive** = in-kernel PoW on the target GPU; the CPU path can't do real difficulty.
- Only **one sliver** is arch-specific: the MMA atom + its accumulator→logical
  layout used for transcript extraction. Everything else (the proof math) is a
  shared, write-once, consensus-critical core.
- Build it as **two orthogonal adapter axes** (workload × hardware) with a
  **capability-based backend registry**, a **shared device PoW-core**, **CuTe/CK
  layout-driven extraction**, and a **byte-exact conformance harness** as the
  consensus gate. New silicon (incl. AMD) is then purely additive.
- For hardware: **RTX 6000 Pro = sm120 (same as now, +VRAM)**; **B200 = sm100
  (datacenter, tcgen05)**; **H100 = the kernel's native, zero-port target.** The
  sm120 in-kernel-PoW fix needs **no** new hardware.
