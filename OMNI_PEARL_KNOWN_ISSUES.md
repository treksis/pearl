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

## P0 — must verify before any mainnet/fork launch

1. **End-to-end ZK proof + node acceptance is UNTESTED for a Blackwell/diffusion
   block.** We only recomputed the PoW jackpot hash in Python. The real path —
   `generate_proof()` (plonky2) → `verify_proof()` → node block acceptance — has
   not been run for a block mined on Blackwell or sourced from diffusion. If the
   `zk-pow` `sanity_checks` (rank ∈ 2⁵..2¹⁰, `common_dim`/tile divisibility) reject
   the diffusion layer's `(k, rank)`, blocks fail despite a valid PoW hash. **This
   is the single biggest unverified link.** Must: run `generate_proof`/`verify_proof`
   on a kernel-found block, end to end against a node.

2. **Full diffusion serving end-to-end is UNTESTED.** The shim is validated on a
   synthetic layer + real `ColumnParallelLinear`, not a real DiT generating images
   through vLLM-Omni with mining state live in the actual diffusion worker process.
   The worker-name detection (`StageDiffusionProc`) is inferred from source, not
   confirmed against the running multi-stage runtime — if the model forward runs in
   a differently-named subprocess, the async manager never inits and **mining is
   silently off**.

## P1 — latent footguns / inconsistencies

3. **Two tile sources (inconsistent).** `GPUMatmulConfigFactory.create()` builds
   `MatmulConfig` from a **fresh** `MinerSettings()` (uncapped 128x256x128), while
   the **execution** tile passed to `noisy_gemm` comes from `config.settings`
   (capped to 128x128x64 on Blackwell). Currently harmless because `matmul_tile_*`
   is metadata and the consensus `mining_config` (patterns + rank + k) is
   tile-independent — but if `matmul_tile_*` is ever serialized into the block or
   checked by a verifier, the mismatch is a bug. **Action:** confirm `matmul_tile_*`
   is never consensus-checked, or thread the capped tile through `create()`.

4. **`quant_7bit` at weight-load needs the async manager initialized.**
   `DiffusionPearlOnlineLinearMethod.process_weights_after_loading` →
   `quant_7bit` → `get_async_manager()._conf.quantization_fast_math`. If weight
   loading ever runs before `register_pearl_miner_layer` inits the async manager,
   it raises. Ordering is currently correct (plugin loads before model load) but
   it's an unguarded assumption.

5. **`PEARL_GEMM_FORCE_KERNEL=1` on a non-native (plain `sm_120`) build crashes**
   (TMA launch failure, no fallback). It's a diagnostic escape hatch, documented,
   but a footgun if used on the wrong build.

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
