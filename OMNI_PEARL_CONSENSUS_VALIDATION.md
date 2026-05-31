# Omni-Pearl — Consensus Validation (Blackwell P0 closed)

**Result:** a block mined by the native fused kernel on consumer Blackwell
(NVIDIA RTX PRO Blackwell, sm120) is **consensus-valid end to end**:

```
[1] verify_plain_proof: True   Mining solution verified successfully
[2] generate_proof (plonky2):  ~15s
[3] verify_proof (ZK):         True   Verified
ZK_E2E_PASS: Blackwell/diffusion-dim block is plain-valid AND ZK-verifiable
```

at mining dims `m=n=k=2048, rank=128`, native_blackwell_build=True.

## The fix (one consensus parameter)

`miner-base/.../settings.py`: `cols_pattern` resized from the upstream **64** entries
(span 0..249, for Hopper's 128×256 tile) to the **32** entries a 128-wide tile opens,
and the consensus tile set to **128×128**.

## Why this is the fix (how it was proven, not guessed)

The earlier hypothesis was an "mma.sync transcript-layout bug." That was **wrong**.
The transcript is correct. The actual chain of evidence:

1. **Isolate the data.** The fused kernel exposes its own noised tensors
   (`ApEA`/`BpEB`). Scoring the canonical jackpot (ported from zk-pow
   `compute_jackpot`) over those tensors at the kernel's *own reported indices*
   **passes** at the kernel's adjusted per-candidate target → the transcript math is
   correct.

2. **A/B test, identical inputs/noise/target.** Reference backend opens a full
   **2×64** tile → `verify_plain_proof = True`. Native kernel opens **2×32** (it
   physically cannot hold 64 cols at bN=128 under the 99 KB SMEM cap) →
   `verify_plain_proof = False`. The *only* difference is tile width.

3. **Difficulty bound vs opened tile.** The verifier's `extract_difficulty_bound`
   scales by `rows_pattern.size × cols_pattern.size × k` (the declared 2×64 tile),
   but the kernel only opened 32 columns → mismatch → rejection.

4. **Apply the fork parameter.** Size `cols_pattern` to the 32 columns the bN=128
   thread actually opens → kernel blocks verify (3/3 + full ZK chain above).

## Two test-environment errors that masked the real cause earlier

- An earlier `cols_pattern=32` attempt edited the **repo source** while the venv
  imported a **non-editable installed copy** still at 64 — so the change never ran.
  Lesson: the active `miner_base` is the installed copy, not the working tree.
- The E2E harness mixed `mining_job.target` (raw) with the header `nbits` target.
  The miner/verifier both scale the target by `h*w*k`; the per-candidate (adjusted)
  target is far easier than the raw header target, which initially looked like a
  transcript failure.

## Reproduce

- Regression test: `miner/vllm-miner/tests/test_blackwell_block_verifies.py`
  (skips without a CUDA native-Blackwell `sm_12xf` build).
- Full ZK chain: mine a block via `pearl_gemm_noisy(..., submit_block=True)` on a
  native-Blackwell build, then `create_proof → verify_plain_proof → generate_proof →
  verify_proof`.

## Scope / honest caveats

- Validated on **sm120** (RTX PRO Blackwell). Datacenter Blackwell (sm100/B200) still
  falls back to the reference path (see KNOWN_ISSUES P2).
- This closes the **mining-consensus** P0. Full **diffusion serving** end-to-end and
  live-node/economic calibration remain open (see `OMNI_PEARL_KNOWN_ISSUES.md`).
- The 32-col tile is half the per-tile work of upstream's 64-col tile; fork genesis
  difficulty must account for the `h*w*k` factor change.
