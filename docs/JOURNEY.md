# The cool_chic journey, in one page

Hash-grid INR -> learned prior -> caption FiLM -> bigger corpus -> bigger λ -> NeRV ->
NeRV-specific prior -> long-clip amortization -> M-NeRV -> stop.

Numbers throughout are on natural-content held-outs (Vimeo-90K septuplets and a 210-frame
spliced "long_natural" clip). All against AV1 (`libaom-av1`) and HEVC (`libx265`) at the same
resolution.

---

## Stage 1 — Hash-grid INR baseline (Cool-Chic)

Tri-plane (xy, xt, yt) hash grid + tiny MLP. Per-clip overfit. QAT with per-tensor scale
quantizer, rANS-coded under per-tensor Laplace prior.

- `sample.mp4` (synthetic test pattern): **23.7 dB / 1.09 bpp**
- AV1 50 kbps on same: 50.4 dB / 0.48 bpp ← *27 dB behind*

## Stage 2 — Learned weight prior (W2)

14k-param MLP predicts `(μ, σ)` per weight from `(tensor_type, level, offset)`. Trained on
6 INRs initially. **Surprise: it loses to per-tensor empirical σ as a *coder*** — but gains
big as a *training regularizer* via rate-aware QAT.

- W1 baseline → W2 prior-aware QAT (λ=3e-3): **-35.5% bpp at iso-PSNR** on `sample.mp4`

## Stage 2 push — turn the lambda knob

Add lambda warmup (linear ramp 0 → λ over first 25% of QAT steps) and longer QAT (3000 steps).
Unlocks much higher λ values:

| λ      | bpp    | PSNR   | Δ vs W1 |
|--------|--------|--------|---------|
| 0      | 1.190  | 24.22  | 0.0%    |
| 1e-1   | 0.514  | 23.50  | -56.8%  |
| **3.0**| **0.360** | 22.36 | **-69.8%** |

## Stage 3 — Caption FiLM prior (W3)

`sentence-transformers/all-MiniLM-L6-v2` → 384-d caption embedding → FiLM (γ, β) per hidden
layer of the prior. On 6 INRs, no transfer to held-out. **Diagnosis: dataset diversity.**

Scaled to **170 clips** (24 ffmpeg synthetic + 48 + 100 Vimeo-90K, BLIP-captioned).
Caption prior now beats non-caption on held-out at high-λ regime by **+4-6% bpp / +0.2-0.3 dB**.

## Stage 4 — Hash-grid hits a wall

On natural Vimeo content, the hash-grid INR plateaus at **~28 dB regardless of capacity or
training**. Verified at 6000 fp32 + 4000 QAT steps, raw 25.4 bpp uncompressed weights — the
ceiling is architectural, not training.

## Stage 5 — Hybrid x265 + INR-residual

Encode with x265 at low CRF, train INR to fit residual, ship both. **Loses by 9+ dB** at matched
bpp on natural content. The INR's residual fit quality (~30 dB) becomes the ceiling on combined
output, while x265 spends the same bits on its own DCT residual path 5-10x more efficiently.

## Stage 6 — NeRV backbone

Replace tri-plane hash with per-frame embedding + upsampling CNN decoder.

| backbone | best PSNR on Vimeo held-out |
|----------|------------------------------|
| hash-grid INR | 27-28 dB (cap) |
| **NeRV medium** | **34.7 dB** |

Ceiling lifted by ~7 dB. Bpp gets terrible because the decoder amortizes poorly over 7 frames.

## Stage 7 — NeRV-specific prior + long clips

Two follow-ups, both necessary:

1. The hash-grid prior's weight statistics don't match NeRV (dense conv weights vs scrambled
   hash entries). Build `prior_nerv_cap.pt` on 170 NeRV INRs.
2. NeRV's ~50-100 KB decoder amortizes properly only on long clips. Generate
   `long_natural.mp4` (210 frames).

| 210-frame long_natural    | bpp    | PSNR  |
|---------------------------|--------|-------|
| NeRV tiny + prior, λ=1e-1 | **0.080** | 24.55 |
| NeRV small + prior, λ=1e-1| 0.237  | 27.03 |
| AV1 10kbps                | 0.092  | 29.36 |
| AV1 50kbps                | 0.421  | 37.31 |

Bpp gap is closed. We have valid RD points where AV1 also operates. **The remaining 4-8 dB
PSNR gap is a CNN-decode-from-frame-embedding ceiling on natural content.**

## Stage 8 — Motion-compensated NeRV (M-NeRV)

Final attempt: optical-flow head + warp(prev, flow) + residual decode. With scheduled sampling
and `K=8` periodic keyframes for exposure-bias mitigation.

**Negative result.** Across all test clips M-NeRV underperformed vanilla NeRV by 4-7 dB at our
training budget. Loss even went up between checkpoints — gradient flow issues through warp +
recurrent path. Architecture is sound (works fine with full teacher forcing) but the optimization
is a much harder problem than vanilla NeRV.

To make motion comp work properly: pretrained flow estimator (RAFT) + separate motion network
(DCVC-style). That's a bigger research effort than a tuning pass.

---

## Things that didn't move the needle

- **Hypernet for fast encoding** (predicting per-weight values from metadata). Trained MSE only
  3% better than "predict the mean" — per-position values aren't learnable across hash-grid INRs.
- **L1 sparsity in QAT**. Either zero (rate-aware QAT already gives 50-75% natural sparsity) or
  destructive (collapses PSNR with λ ≥ 1e-4). Not a useful knob.
- **Adding more synthetic clips beyond ~70**. Returns saturate quickly without semantic diversity.
- **Larger prior MLPs**. Going from 14k to 162k params didn't help generalization on small
  datasets; small prior plus big diverse dataset wins.

## Things worth revisiting if you pick this up

- DCVC-style motion compensation (separate flow network, pretrained). Likely 5-10 dB gain on
  natural content.
- HiNeRV (hierarchical multi-scale features) for the backbone. Lifts the spatial-detail ceiling
  before motion comp matters.
- Specialize for content where AV1's block-DCT struggles: line art, screen recordings, slide
  decks, game UI. INRs have natural advantages there at very low bpp.

## File map

| topic                | file                                  |
|----------------------|---------------------------------------|
| Hash-grid + tri-plane | `cool_chic/{hash_grid,inr,codec}.py` |
| QAT + STE quantizer  | `cool_chic/quantize.py`               |
| Hash-grid training   | `cool_chic/train_per_*.py`, `train_video_qat.py` |
| Weight serialization | `cool_chic/serialize.py`              |
| Learned prior        | `cool_chic/prior.py`                  |
| Caption encoder      | `cool_chic/caption.py`                |
| Bitstream            | `cool_chic/bitstream.py`, `bitstream_v2.py` |
| NeRV                 | `cool_chic/nerv.py`, `train_nerv.py`  |
| M-NeRV               | `cool_chic/mnerv.py`, `train_mnerv.py` |
| Hybrid x265+INR      | `cool_chic/residual_codec.py`         |
| Hypernet (negative)  | `cool_chic/hypernet.py`, `train_hypernet.py` |
| Benchmarks           | `cool_chic/benchmarks/rd_*.py`        |
