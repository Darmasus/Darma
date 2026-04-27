# cool_chic — an INR-based neural video codec exploration

A research codebase that started as "can a neural codec beat AV1" and ended somewhere more
interesting and honest. This repo documents the full journey: what worked, what didn't, and
why — with reproducible RD curves on real video against AV1 and x265.

> **Headline:** we built a working caption-conditioned INR video codec that compresses 200-frame
> 96×160 natural video to **0.08 bpp at 25 dB PSNR**, an operating point AV1/x265 don't address.
> But on bpp-vs-PSNR efficiency at AV1's regime, we're still **4–8 dB behind**, and the remaining
> gap is architectural (no motion compensation, no DCT) rather than a tuning problem.

## What's in here

```
cool_chic/
├── codec.py              ImageINR / VideoINR (hash-grid + tri-plane + MLP)
├── hash_grid.py          multi-resolution hash encoding (Instant-NGP)
├── inr.py                small ReLU MLP
├── nerv.py               per-frame embed + upsampling CNN decoder
├── mnerv.py              NeRV + optical-flow head + warp (negative result)
├── quantize.py           per-tensor STE-round quantizer
├── serialize.py          canonical weight tokenization for prior training
├── prior.py              learned per-weight (μ,σ) predictor with optional FiLM caption
├── caption.py            MiniLM (384-d sentence encoder) wrapper
├── bitstream.py          per-tensor Laplace rANS bitstream (`.cc`)
├── bitstream_v2.py       prior-conditioned + caption-conditioned variants
├── residual_codec.py     hybrid x265 + INR residual (negative result)
├── train_per_*.py        fp32 overfit loops
├── train_*_qat.py        rate-aware QAT (per-tensor + prior + L1 sparsity)
├── train_prior.py        weight-prior trainer (with caption FiLM)
├── train_hypernet.py     value-predicting hypernet trainer (negative result)
├── benchmarks/           RD harnesses, plot helpers, JSON dumps
└── scripts/              dataset builders (Vimeo-90K + BLIP captions)
```

## Architectural journey

### Week 1 — hash-grid INR baseline

Start from [Cool-Chic](https://arxiv.org/abs/2212.05650) -style design:

```
(x, y, t) ∈ [0,1]³ ─► tri-plane hash grid (3 × HashGrid2D over xy, xt, yt)
                  ─► sum into L·F-dim feature
                  ─► tiny MLP ─► RGB
```

Per-clip overfit. Quantization-aware training (STE round) with per-tensor learnable scale.
rANS-coded bitstream under per-tensor Laplace prior.

**Result on `sample.mp4`:** 23.7 dB at 1.09 bpp.
**Result vs AV1 50 kbps on the same clip:** AV1 = 50.4 dB at 0.48 bpp. **27 dB behind.**

### Week 2 — learned weight prior + rate-aware QAT

The interesting empirical finding: a small MLP prior (14k params) trained on a corpus of 22 INRs
predicts per-weight `(μ, σ)` for the rANS coder. We expected it to be a better encoder than
per-tensor σ. **It's not** — per-tensor empirical σ always fits a single trained INR best.

But the prior is enormously valuable as a **training regularizer**. Plug it into QAT as a
differentiable rate term, and it pulls weights toward a prior-predictable distribution. The
*existing* per-tensor Laplace coder then has an easier job. Net result on `sample.mp4`:

| run                    | bpp   | PSNR  | vs W1 |
|------------------------|-------|-------|-------|
| W1 baseline            | 1.090 | 23.93 | 0.0%  |
| W2 prior-aware (λ=3e-3)| 0.707 | 23.92 | **-35.5%** |

After "W2 push" (longer QAT, lambda warmup, λ up to 1.0):
| W2 push (λ=1e-1)       | 0.514 | 23.50 | **-56.8%** |
| W2 push (λ=3.0)        | 0.360 | 22.36 | **-69.8%** |

### Week 3 — caption-conditioned prior

Add a FiLM-modulated caption pathway to the prior using `sentence-transformers/all-MiniLM-L6-v2`
(384-d). On a tiny corpus the caption signal didn't move held-out at all. The reason was
**dataset diversity**, not architecture.

Scaled to **170 clips** (24 ffmpeg-synthetic + 48 Vimeo-90K + 100 more Vimeo, all auto-captioned
with `Salesforce/blip-image-captioning-base`):

| held-out sample.mp4 | W2 (no-cap prior)    | W3 (cap prior)        |
|---------------------|----------------------|------------------------|
| λ=1e-1              | 64,511 B / 23.29 dB  | **61,735 B / 23.58 dB**|
| λ=1e0               | 52,627 B / 22.85 dB  | **50,207 B / 23.04 dB**|

W3 dominates W2 at the high-λ end: smaller bytes *and* higher PSNR. Caption signal pays off.

### Week 4 — hypernet for fast encoding (negative)

Tried predicting per-weight VALUES from `(tensor_type, level, offset, caption_emb)`. Trained MSE
on the 22-INR dataset converged to 1.03× over "predict the dataset mean" — meaning **per-position
values are not learnable across hash-grid INRs**. The hash function deterministically maps positions
to buckets, but each clip's bucket VALUES are content-dependent and uncorrelated across clips.
Hypernet-initialized INRs converged *slower* than PyTorch's default Uniform(-1e-4, 1e-4) init.

**Lesson kept:** per-tensor σ statistics are learnable across clips (that's what `WeightPrior`
captures); per-position weight values are not.

### NeRV backbone — break the PSNR ceiling

On natural Vimeo content, hash-grid INR plateaus at **~28 dB PSNR regardless of capacity or
training budget** (verified at 6000 fp32 + 4000 QAT steps, raw 25.4 bpp uncompressed weights).
The hash function inductive bias just doesn't fit dense natural texture.

Switched to a NeRV-style backbone:

```
frame_idx (T,) ─► frame_embed (T, D)
              ─► Linear stem -> tiny (C, h₀, w₀)
              ─► K upsample blocks: Upsample×2 + Conv3×3 + GELU + Conv3×3 + GELU
              ─► out conv ─► (T, 3, H, W)
```

| held-out Vimeo (7f, 96×160) | best PSNR | bpp at best |
|-----------------------------|-----------|-------------|
| hash-grid INR               | 27.5 dB   | (any)       |
| **NeRV medium**             | **34.7 dB** | 31.6 bpp |

PSNR ceiling went from ~28 → ~35 dB. Bpp got terrible because a 60–425 KB decoder amortizes
poorly over a 7-frame clip.

### NeRV-specific prior + long-clip amortization

Two follow-ups:

1. **Train a fresh prior on NeRV weight distributions.** The hash-grid prior is useless for
   NeRV because the weight statistics are completely different (dense conv weights vs scrambled
   hash entries). Built `prior_nerv_cap.pt` on 170 NeRV INRs.

2. **Long clips.** A 60 KB decoder spread over 7 frames is 4.4 bpp; spread over 210 frames
   it's 0.08 bpp. Generated `long_natural.mp4` (210 frames spliced from 30 random Vimeo
   septuplets) and `long_continuous.mp4` (140 frames from one parent video).

| 210-frame long_natural    | bpp    | PSNR  |
|---------------------------|--------|-------|
| NeRV tiny + prior, λ=1e-1 | **0.080** | 24.55 |
| NeRV small + prior, λ=1e-1| 0.237  | 27.03 |
| AV1 10kbps                | 0.092  | 29.36 |
| x265 10kbps               | 0.102  | 28.80 |
| AV1 50kbps                | 0.421  | 37.31 |

NeRV+prior produces valid RD points at 0.08–0.3 bpp where AV1/x265 also operate. The bpp gap
is closed. **The remaining 4–8 dB PSNR gap is the ceiling of CNN-decode-from-frame-embedding
on natural content.**

### M-NeRV — motion compensation (negative)

Final attempt: add an optical-flow head, decode each frame as `warp(prev_frame, flow_t) + residual_t`,
with periodic keyframes (`K=8`) to bound autoregressive error compounding. Standard scheduled
sampling for exposure bias.

Across three test clips, M-NeRV **lost** to vanilla NeRV by 4–7 dB at our training budget. Loss
even went up between checkpoints, indicating gradient flow problems through the warp + recurrent
path. The architecture is sound (works fine with full teacher forcing), but training optical
flow from random init by MSE on warped output is a brutally hard optimization problem.

**Conclusion:** to do motion compensation properly, you need a pretrained flow estimator
(RAFT-style) and a separate motion network — i.e., the DCVC architecture. That's a bigger
research effort than a tuning pass.

## Final RD curves

| benchmark                    | path                                                  |
|------------------------------|-------------------------------------------------------|
| Cool-Chic INR vs AV1/x265    | `cool_chic/benchmarks/out/rd_plot_heldout_00075.png` |
| NeRV (small/medium)          | `cool_chic/benchmarks/out/rd_nerv_combined.png`      |
| **NeRV + prior on long clip**| `cool_chic/benchmarks/out/rd_nerv_with_prior.png`    |

JSON dumps for every benchmark are alongside.

## Reproducing

```bash
# Install (Python 3.9, PyTorch + CUDA)
pip install -r requirements.txt
pip install transformers          # for MiniLM + BLIP captioner

# 1) Build a tiny hash-grid INR codec on sample.mp4 (Week 1)
python cool_chic/scripts/encode_video.py sample.mp4 sample.cc

# 2) The big benchmark — NeRV with caption-conditioned prior vs AV1/x265 on a Vimeo held-out clip
python cool_chic/benchmarks/rd_nerv.py \
    cool_chic/data/long_clips/long_natural.mp4 \
    --fp32-steps 4000 --qat-steps 2500 \
    --sizes "tiny:16:32;small:32:64" \
    --lambdas "0.0,1e-3,1e-2,1e-1" \
    --frames-per-step 16 \
    --prior cool_chic/data/prior_nerv_cap.pt \
    --caption "natural video clip with people and motion"
```

To build the data and priors from scratch:

```bash
# Generate ffmpeg-synthetic clips with captions
python cool_chic/scripts/make_diverse_clips.py

# Sample real Vimeo-90K septuplets and BLIP-caption them
# (assumes /d/datasets/vimeo_septuplet/ exists)
python cool_chic/scripts/make_vimeo_clips.py --n 50
python cool_chic/scripts/make_vimeo_clips_more.py --n 100

# Train one INR per clip in the manifest
python cool_chic/scripts/make_inr_dataset_v2.py \
    --manifest cool_chic/data/combined_manifest_v2.json \
    --out cool_chic/data/inr_dataset_v4.pt

# Train priors (hash-grid version)
python cool_chic/train_prior.py --data cool_chic/data/inr_dataset_v4.pt \
    --out cool_chic/data/prior_v4_nocap.pt --batch-tokens 131072
python cool_chic/train_prior.py --data cool_chic/data/inr_dataset_v4.pt \
    --out cool_chic/data/prior_v4_cap.pt --caption --batch-tokens 131072

# NeRV variant
python cool_chic/scripts/make_nerv_dataset.py
python cool_chic/train_prior.py --data cool_chic/data/nerv_dataset.pt \
    --out cool_chic/data/prior_nerv_cap.pt --caption --batch-tokens 131072
```

## What we learned, ranked

1. **Per-tensor empirical σ is hard to beat as a *coder*.** A learned prior never beat per-tensor
   σ on the *same* trained INR — but the prior shines as a *training regularizer* that reshapes
   weights to be more compressible.

2. **Caption conditioning needs scale.** On 6 INRs, FiLM moved nothing. On 170 with diverse
   BLIP captions, it cleanly beat the non-caption prior on held-out content. Architecture was
   right from day 1; the bottleneck was data.

3. **Hash-grid is not the right primitive for natural video.** It hits an architectural ~28 dB
   PSNR ceiling that no amount of training/capacity/lambda can break. NeRV's CNN decoder lifts
   it to ~35 dB.

4. **Per-position weight values are not learnable across clips with hash grids.** The hash
   scrambles positions; "what value lives at bucket 472" is content-dependent and uncorrelated
   across trained INRs. Hypernet for INR init is a dead end with this primitive.

5. **Hybrid x265+INR-residual loses to pure x265.** x265 is itself a hyper-optimized residual
   coder (DCT + entropy). Our INR residual ends up 5–10× less efficient per byte than x265's
   own residual path.

6. **Motion compensation needs pretrained flow.** From-scratch optical flow learned via MSE on
   warped output is too bumpy a loss surface. Want motion comp? Use DCVC.

## What we didn't beat AV1 at — and why

AV1 won on every natural-video benchmark we ran. The remaining gap is structural:
- AV1 has explicit motion vectors + warping + DCT residuals, all engineered over decades.
- Our codec has a CNN that decodes a per-frame embedding to a full frame from scratch.

Closing that gap honestly needs either (a) a DCVC-style design with separate motion estimation,
or (b) accepting a different operating regime — for instance, content where AV1's block-based
DCT struggles (animation, line-art, screen recordings). On synthetic test patterns (`sample.mp4`)
NeRV-style INRs are surprisingly competitive at very low bpp.

## Acknowledgements

- [Cool-Chic](https://arxiv.org/abs/2212.05650) for the per-clip-overfit + tri-plane design
- [Instant-NGP](https://nvlabs.github.io/instant-ngp/) for the multi-resolution hash grid
- [NeRV](https://arxiv.org/abs/2110.13903) for the per-frame embedding + upsampling CNN backbone
- [DCVC](https://arxiv.org/abs/2109.15047) family for the motion-compensated baseline we couldn't match
- [`constriction`](https://github.com/bamler-lab/constriction) for the rANS coder
- [Vimeo-90K](http://toflow.csail.mit.edu/) for the per-clip septuplets
- BLIP and MiniLM via HuggingFace for the caption pathway

## License

Research code, no warranty. Vimeo-90K usage is non-commercial only per the dataset's license.
