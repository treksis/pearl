# Omni-Pearl — Hard-Fork Positioning & Differentiation

Honest rationale for omni-pearl as a hard fork of Pearl: **Blackwell + diffusion**
(accessible consumer GPUs, generative-media useful work) vs **Pearl's Hopper + LLM**
(datacenter GPUs, text). Written to be defensible under scrutiny, not as hype —
a fork only survives if the differentiation is real, and the trade-offs are stated.

## The thesis (the Litecoin parallel, done honestly)

Litecoin didn't beat Bitcoin on cryptography — it reused PoW and changed the
*parameters and accessibility*: a hash that was (initially) consumer-GPU-mineable,
faster blocks, larger supply. "Silver to gold." It earned a durable niche by being
**more accessible**, not by claiming better security.

Omni-pearl makes the same move: **same proven Proof-of-Useful-Work security as
Pearl, retargeted to hardware almost anyone can buy and to a useful-work market
almost everyone understands.** We did not invent a new consensus; we removed the
hardware gate and changed the workload.

## What is genuinely better (and why it's defensible)

1. **Mine with the GPU you already own.** Pearl's mining kernel is tuned for
   **Hopper (H100/H200)** — $25k–40k datacenter GPUs, scarce and gatekept.
   Omni-pearl runs on **consumer Blackwell (RTX 5090 / RTX PRO)** — $2–10k cards
   in gaming rigs and workstations worldwide. This is an order-of-magnitude larger,
   cheaper, more decentralized miner base. *This is the core pitch — and it's real:*
   we have a native consumer-Blackwell kernel running at GPU speed, consensus-valid,
   which upstream Pearl does not ship.

2. **Useful work = generative media, not just text.** Pearl's useful work is LLM
   text inference; omni-pearl's is **diffusion image/video/audio generation** — a
   larger, more consumer-visible market (the Stable-Diffusion / Midjourney economy).
   The network doubles as a decentralized generative-AI engine producing marketable
   output, not just tokens.

3. **Diffusion is a better mining substrate per GPU.** A diffusion request is N
   denoise steps, each a full compute-bound DiT forward — so it emits far more
   mineable, tensor-core-saturating matmul volume per request than an LLM (whose
   decode phase is memory-bandwidth-bound and under-utilizes compute). More
   proof-of-work per unit of useful output, and the cheap GPU stays maxed.

4. **Multi-vendor by design.** The mining backend is an adapter layer (Hopper /
   Blackwell-consumer / Blackwell-DC / AMD) gated by a byte-exact conformance
   harness. The roadmap is "mine on the broadest possible hardware," vs a
   Hopper-centric incumbent.

## Why Hopper gates Pearl — and why diffusion ungates it (the real argument)

The hardware barrier is **not** the mining kernel; it's the *useful work's memory
profile*. This is the deepest reason the fork makes sense:

- **Mining itself is tensor-core (compute) bound, not VRAM-bound.** Large matmuls
  have high arithmetic intensity → bound by tensor-core TOPS. VRAM is capacity (fit
  the model), not throughput. So *for mining*, you want TOPS — which consumer cards
  have. You don't need Hopper to mine.
- **Pearl needs Hopper because of LLM serving, not mining.** Its useful work is
  large LLMs (70B-class). Those need **80 GB VRAM** to hold the model and are
  **memory-bandwidth-bound** during decode (every token streams the whole model's
  weights) → they require Hopper's HBM (~3.35 TB/s). A 24 GB consumer card can't
  even load a 70B model. Mining just rode along on the datacenter hardware the LLM
  workload forced.

| | Pearl (LLM) | Omni-pearl (diffusion) |
|---|---|---|
| Useful-work bottleneck | memory-bandwidth + VRAM (decode streams weights) | **compute** (DiT reuses weights across thousands of patches) |
| Model size | ~70 B → ~70 GB → needs 80 GB H100 | DiT ~2–12 B → fits a 24 GB consumer card |
| Natural hardware | datacenter (HBM, big VRAM) | **consumer GPU (strong compute, modest VRAM/bandwidth)** |

**Diffusion is compute-bound and smaller — exactly what a consumer Blackwell card
is good at; LLM is bandwidth-bound and huge — exactly what forces Hopper.** So
omni-pearl isn't merely "Pearl's kernel on cheaper GPUs"; it's the one mainstream
AI workload whose hardware profile *matches* cheap GPUs. The kernel port made it
*possible*; the workload choice makes it *natural*. (Honest caveat: a consumer card
still loses **per-card** to an H100 on raw compute — the edge is **per-dollar** and
**accessibility**, not head-to-head.)

## Same security, lower risk than a brand-new chain

The PoUW mechanism, the ZK verifier (plonky2), and the consensus transcript are
**inherited from Pearl, unchanged and model-agnostic** (the on-chain circuit proves
the opened tile, blind to LLM-vs-diffusion). So omni-pearl isn't an unproven new
cryptosystem — it's a battle-tested mechanism made accessible. That is a *lower*
technical-risk story than most new L1s, which speculators should value.

## Honest trade-offs (state these — credibility depends on it)

- **Lower per-GPU hashrate.** A consumer Blackwell card is weaker than an H100;
  decentralization comes from *volume of cheap cards*, not per-card power. (Same
  truth as GPUs-vs-ASICs for Litecoin.)
- **Quality at 7-bit.** Mining-precision (int7) diffusion output is lower fidelity
  than full precision — a real trade-off for the "useful media" claim.
- **It's a fork, not new cryptography.** The novelty is hardware target + workload +
  economic calibration, not a new consensus primitive. (Litecoin was the same.)
- **Bootstrapping was the hard part — and that's the moat.** Making Pearl's
  Hopper-specific fused kernel run correctly and at speed on consumer Blackwell was
  non-trivial (we found and fixed a denoise no-op, the `sm_120f` family-arch
  requirement, and wired the dispatch). First-mover advantage on accessible-hardware
  PoUW mining is defensible precisely because the engineering bar is high.

## The speculator pitch (truthful, one paragraph)

> Omni-pearl is to Pearl what Litecoin was to Bitcoin: the same proven
> proof-of-useful-work, freed from the datacenter. Pearl needs $30k H100s;
> omni-pearl mines on the RTX 5090 already in millions of PCs — an order of
> magnitude more accessible and decentralized. Its useful work is generative
> image/video/audio, a market people pay for daily, and diffusion keeps cheap GPUs
> fully utilized. The hard part — a correct, consensus-valid, GPU-speed Blackwell
> mining kernel — is solved and is the moat. Lower per-card hashrate and 7-bit
> fidelity are the honest costs; accessibility and a generative-AI useful-work
> economy are the payoff.

## What still must ship for the claim to be fully true
(see `OMNI_PEARL_BLACKWELL_KERNEL_FINDINGS.md` + `OMNI_PEARL_BUILD_PLAN.md`)
- Full `test_pearl_gemm` config sweep under the native Blackwell kernel.
- Sustained end-to-end mining against a live node with a canonical quantized DiT.
- Fork genesis/difficulty/reward calibration for the consumer-GPU fleet.
- Confirm the canonical model's layer `(k, rank)` satisfy the zk-pow constraints.
