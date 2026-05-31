# Omni-Pearl — Known Issues & Critical Self-Review

Adversarial review of what we built (Blackwell kernel fix + dispatch wiring +
diffusion shim + omni plugin). Honest accounting of what is verified, what is
assumed, and where the bugs/risks are. Prioritized P0 (blocks mainnet) → P2 (ops).

## Verified (retired risks)
- **Denoise fix generalizes:** all 8 SMEM-fitting compiled configs (R128/R64 ×
  {128x128x64, 64x128x64, 64x64x64} × stages 2/3) produce denoised C at
  **cos = 1.00000** vs the oracle. Not a one-config fluke.
- PoW hash is consensus-valid at the **hash level** (canonical recompute: found
  block's hash ≤ target).
- Native kernel is the default on `sm_120f` builds with no env flag (dispatch
  wired); Hopper byte-identical.

## P0 — OPEN BLOCKER: in-kernel PoW transcript is wrong on consumer Blackwell

1. **The fused kernel's in-kernel PoW finds blocks that FAIL canonical
   `verify_plain_proof` on consumer Blackwell (sm120).** Established end-to-end
   (the real plain-proof verifier, not a numpy approximation):
   - Reference-path block → `verify_plain_proof: True` ✅
   - Kernel-path block → `verify_plain_proof: False` ("hash does not meet target") ❌
   Same harness; only difference is native kernel vs reference. So **the in-kernel
   PoW transcript computation on the SM80 `mma.sync` path diverges from the
   canonical jackpot** — the kernel's "blocks" would be rejected by the chain.
   (This corrects an earlier over-claim: a numpy recompute matched the kernel, but
   the authoritative Rust verifier does not. The denoise fix made the GEMM output
   correct; the PoW *transcript* is still wrong.)

   **Ruled out:** `tensor_hash` (GPU == reference), commitment (shared), GEMM/noising
   values (bit-exact), header/target consistency, and the cols_pattern *span* — a
   tested hypothesis (truncating `cols_pattern` to fit the 128-wide tile) **did not
   fix it.** The divergence is in how the per-thread accumulator fragment maps to /
   reduces into the canonical per-partition jackpot on the `mma.sync` layout.

   **Compounding hardware constraint:** the canonical PoW pattern is *designed for a
   128×256 tile* (`settings.py` comment), which is ~146 KB SMEM and **does not fit
   consumer Blackwell's 99 KB cap**. We cap to 128×128, which (besides the transcript
   bug) truncates the partition. A real fix likely needs **both** an `mma.sync`
   transcript-layout fix **and** a SMEM-compatible mining config, co-designed and
   validated against the ZK oracle.

   **Status: NOT fixed.** This is the genuine hard core of competitive consumer-
   Blackwell mining and needs dedicated, iterative kernel debugging (dump per-thread
   fragment (row,col,value) and diff against the canonical jackpot, byte-exact),
   beyond a single session. Two fix attempts (denoise — succeeded; transcript pattern
   — failed) so far. Until this is fixed, **native Blackwell mining is not
   consensus-valid / not competitive.** The reference path is consensus-valid but
   CPU/easy-target-only (not competitive).

2. **Full diffusion serving end-to-end is UNTESTED.** The shim is validated on a
   synthetic layer + real `ColumnParallelLinear`, not a real DiT generating images
   through vLLM-Omni with mining state live in the actual diffusion worker process.
   The worker-name detection (`StageDiffusionProc`) is inferred from source, not
   confirmed against the running multi-stage runtime — if the model forward runs in
   a differently-named subprocess, the async manager never inits and **mining is
   silently off**.

## P1 — investigated; NOT real bugs (kept here for the record, deliberately NOT "fixed")

3. **"Two tile sources" — NOT a bug.** Verified by grep: `MatmulConfig.matmul_tile_*`
   is only ever *set* (in `create()`) and read in tests — **never** in any
   consensus/block/verify path. `adjust_target()` uses the `mining_config`
   (rows/cols pattern + rank), the block/PlainProof carries no tile, and the
   execution tile comes from `config.settings` (capped). `matmul_tile_*` is
   vestigial metadata; changing it would be churn. Left as-is.

4. **`quant_7bit` ordering — already handled.** If the async manager isn't
   initialized, `get_async_manager()` already raises a clear
   `AssertionError("Async Loop Manager has not been initialized yet")`. No silent
   failure; a guard would be redundant. Left as-is.

5. **`PEARL_GEMM_FORCE_KERNEL` footgun — intentional.** Explicit diagnostic escape
   hatch; guarding it defeats the purpose. Documented, left as-is.

**Fixed (genuine, minimal):** the tile cap silently no-op'd if
`shared_memory_per_block_optin` were unavailable (→ 0 → no cap → 146KB launch
crash). Now caps conservatively when SMEM is unknown on Blackwell-family parts.

## P2 — build / ops

6. **`uv.lock` not regenerated** for the new deps (`vllm==0.21.0`, `vllm-omni`);
   clean-environment dependency resolution + `vllm-omni` PyPI availability for the
   pinned version are unverified.
7. **`sm_120f` needs CUDA ≥ 12.8/13.** Older toolkits will reject the family target.
8. **Datacenter Blackwell (sm100 / B200) is not accelerated.** The native define is
   set only for `sm_12xf` (consumer); on B200 the dispatch falls back to the
   reference path (safe, but not GPU-speed). A separate `sm_100a/f` enablement +
   validation would be needed.

## Strategic gaps the positioning depends on
- **Canonical DiT must be chosen so its heavy-layer `(k, rank)` satisfy zk-pow
  constraints** and map to compiled tiles (else those layers don't mine).
- **7-bit image-quality** has only been checked as matmul cosine, not as actual
  generated-image fidelity — the "useful media" claim needs a real quality eval.
- **"Useful work" is incentive-based, not enforced** (a miner could run arbitrary
  matmuls, not real inference) — true for Pearl too, but relevant to the value claim.

## Bottom line
The kernel-level work is solid and now broadly validated. The unproven surface is
**above the kernel**: end-to-end ZK/node acceptance and full diffusion serving.
Those are the two things to close before claiming a launch-ready fork.
