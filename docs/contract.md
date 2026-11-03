# Concept-Steered Mamba-CLIP: v1 Rebuild Contract and Milestones

Purpose: two independent builds (ours and Chethan's) must be comparable. This document fixes the shared evaluation, the build order with pass/fail gates, and the logs we keep. Anything marked **OPEN** needs agreement from both sides before work starts.

Status: AGREED v1.0 by both sides (2026-11-03)

---

## 1. Ground rules

1. **Shared evaluation, independent models.** Data splits, preprocessing protocols, metrics and the reporting template are shared. Model code is written independently.
2. **No peeking at the other model's code until v1 is complete** (both sides). Sharing data/eval code and split manifests is fine. This keeps the final comparison honest.
3. **v1 is spec-faithful** to `Concept_Steered_Mamba_CLIP_AIGI_Detection_Architecture.docx`. Where the spec leaves a choice open, each side decides independently and logs it (Section 6).
4. **v2 changes one thing at a time** on top of v1, each logged as its own experiment.
5. **The held-out generator is touched once per run**, for the final test. No model selection on it.

## 2. Shared evaluation contract

| Item | Value | Status |
|---|---|---|
| Dataset | Tiny-GenImage (HuggingFace `TheKernel01/Tiny-GenImage`), 7 non-empty generators + real (SD14 is empty) | Fixed |
| Folds (leave-one-generator-out) | Midjourney, BigGAN, ADM | Fixed |
| Images | 300 per seen generator, real balanced 1:1 with fake; 90/10 train/val; test = 500 fake (held-out) + 500 real (dataset val split) | Fixed |
| Split manifests | Canonical split files (image IDs) exported once and shared; both builds load the manifests, not their own split code | **OPEN**: request `splits/*.json` from Chethan's repo, or generate once and share |
| Protocols | `raw` and `controlled` (below) | Fixed |
| Primary metric | AUROC on the held-out test set with 95% bootstrap CI (1000 resamples) | Fixed |
| Secondary metrics | Average precision; accuracy at threshold 0 (reported, not headlined) | Fixed |
| Seeds | 0, 1, 2 (v1 minimum: seed 0 for all folds, 3 seeds for the main experiment) | **OPEN** |
| Epochs | Fixed at 5 for every arm; **primary result = final epoch**; best-validation epoch reported as secondary | **OPEN** |
| Backbone | CLIP ViT-L/14@336 (OpenAI), HF `hidden_states[16]` and `[24]` | Fixed |

### Protocol definitions

- **raw:** bicubic resize of the short side to 336, centre crop 336, CLIP normalisation. Training adds random horizontal flip.
- **controlled:** applied identically to real and fake, train and test: centre-crop to a square, resize to 256x256, re-encode as JPEG quality 90, then CLIP preprocessing (resize to 336). Order matters: resize *before* JPEG so every image has exactly one compression.
- **Known residuals (listed, not claimed removed):** upsampling blur differs by generator (BigGAN 128px, up; Midjourney 1024px, down).

### Sanity checkpoint (gate for M2)

Our frozen-CLIP probes on the controlled protocol must reproduce Chethan's reported numbers within the confidence intervals:

| Held out | UnivFD-style probe | Multi-layer probe |
|---|---|---|
| Midjourney | 0.826 [0.799-0.851] | 0.875 [0.855-0.896] |
| BigGAN | 0.960 [0.951-0.969] | 0.976 [0.968-0.983] |
| ADM | 0.889 [0.866-0.908] | 0.977 [0.970-0.984] |

If we cannot reproduce these, the harness is wrong or the splits differ; fix before building any model.

## 3. Environment (Colab)

- Colab free tier, Tesla T4 (15 GB). No local GPU. Local machine is an editor and Git client only.
- Code on GitHub (private repo, token in Colab Secret, never in a notebook cell). Checkpoints and run outputs on Google Drive. Dataset downloaded to Colab local disk each session.
- Every training script must **resume from a Drive checkpoint** after disconnect.
- Claude writes code and CPU unit tests with small stand-in models; the real runs happen on Colab. Real CLIP and real data are first exercised on Colab, so every module needs a Colab smoke notebook.
- Budget reference (from the other build): ~7.5 img/s training at batch 8 with fp16 autocast; peak 6.6 GB at batch 8; batch 16 peaked at 12.4 GB. A pure-PyTorch 576-step scan is the suspected bottleneck.
- Colab free tier has session and daily GPU limits: keep every experiment under ~1 hour and resumable.

## 4. Milestones and gates

Each milestone ends with a gate. Do not start the next milestone until the gate passes.

| # | Milestone | Gate (pass/fail) | Status |
|---|---|---|---|
| M0 | Repo, this contract, decision log | Contract agreed by both sides | Done (tag m0-done) |
| M1 | Data + eval harness (loader, manifests, protocols, metrics, bootstrap CI) | Held-out generator absent from train/val; splits balanced and deterministic; controlled protocol verified (single JPEG compression) | Not started |
| M2 | Frozen-CLIP probes (UnivFD-style and multi-layer) | Reproduce reference numbers in Section 2 within CI | Not started |
| M3 | Backbone + LoRA (r=8) | Taps are 2 x (B,576,1024) with CLS dropped; LoRA tensors get gradients, base CLIP gets none | Not started |
| M4 | Fusion + concept bank + affinity | X is (B,576,384); alpha is (B,576,16) and rows sum to 1; gradients reach both projections | Not started |
| M5 | Plain Mamba block | Scan causal (perturbing token t changes nothing before t); matches an independent reference scan on random input; memory OK | Not started |
| M6 | Steering gates | With gates at init, steered output equals plain output exactly; gates receive gradient | Not started |
| M7 | Concept readout + 24x24 map | Map sum plus biases equals the logit (to fp16 tolerance) | Not started |
| M8 | Training loop | Overfits 4 images; resumes from checkpoint; fp16 stable | Not started |
| M9 | Main experiment: steered vs plain, same fold, matched settings | Both arms complete, results JSON written | Not started |
| M10 | Replicate (other folds, 3 seeds) + ablations (random concepts, no concepts, no LoRA, single block) | Results table complete | Not started |
| M11 | Compare with the other build; write-up | Comparison table filled; report drafted | Not started |

Tracking rule: update the Status column and add a git tag (`m3-done`, etc.) when a gate passes.

## 5. Spec items that must be decided and logged (from architecture doc, Section 8)

Both builds must record their choice for each:

1. CLIP checkpoint and preprocessing
2. LoRA placement and which weights stay frozen
3. Block-16/24 fusion equation
4. Visual and text projection definitions
5. Concept prompts and templates
6. Affinity function, temperature, normalisation
7. How alpha conditions each Mamba block
8. Mamba scan order and normalisation
9. Concept-wise pooling and scoring heads
10. 24x24 explanation-map construction
11. **Loss function and any auxiliary concept loss** (not stated in the other build's documentation)
12. Train/val/test generator splits (fixed by Section 2)

## 6. Decision log template

See `decision_log.md` (pre-seeded with D001-D012).

## 7. Experiment log template

See `experiment_log.md`. One row per run. Never delete failed runs.

## 8. Final comparison table (fill in at M11)

| Component | Chethan's choice | Our choice | Outcome (measured) | Verdict |
|---|---|---|---|---|
| Fusion | | | | |
| Concept bank / grounding | | | | |
| Affinity (norm, temperature) | | | | |
| Steering mechanism | | | | |
| Scan order / implementation | | | | |
| Readout and map | | | | |
| Loss | | | | |
| Training recipe | | | | |

## 9. v2 candidates (one change at a time, after v1)

Hypotheses from the review of the other build; each becomes its own logged experiment:

- Ground concepts in CLIP's joint space (shared or frozen projection) instead of two independent learned projections
- Auxiliary loss tying concept scores to real/fake polarity
- Sharper or rescaled alpha; non-zero or larger gate initialisation; higher gate learning rate
- Early stopping / augmentation (random JPEG, blur, resize jitter) to stop memorisation of seen generators
- Validation that reflects generalisation (hold out one seen generator for validation)
- 2-D-aware scan order; faster scan implementation
- More generators / images; a second dataset

## 10. Open items for both sides

See `m0_open_items.md` for proposed resolutions.

- [x] Agree split manifests (source and format)
- [x] Agree seeds and fixed epoch count
- [x] Confirm the "no peeking at model code until v1 is done" rule with Chethan
- [x] Create repo and add this document
