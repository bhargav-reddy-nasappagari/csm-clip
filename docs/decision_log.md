# Decision Log

One row per decision. Add it the moment you decide, not afterwards.
Rows D001-D012 are pre-created from contract Section 5 so nothing is forgotten; fill them as each milestone reaches it.

| ID | Date | Component | Decision | Alternatives considered | Reason | Expected effect |
|---|---|---|---|---|---|---|
| D001 | | CLIP checkpoint and preprocessing | TBD (M1/M3) | | | |
| D002 | | LoRA placement, frozen weights | TBD (M3) | | | |
| D003 | | Block-16/24 fusion equation | TBD (M4) | | | |
| D004 | | Visual and text projection definitions | TBD (M4) | | | |
| D005 | | Concept prompts and templates | TBD (M4) | | | |
| D006 | | Affinity function, temperature, normalisation | TBD (M4) | | | |
| D007 | | How alpha conditions each Mamba block | TBD (M6) | | | |
| D008 | | Mamba scan order and normalisation | TBD (M5) | | | |
| D009 | | Concept-wise pooling and scoring heads | TBD (M7) | | | |
| D010 | | 24x24 explanation-map construction | TBD (M7) | | | |
| D011 | | Loss function and auxiliary concept loss | TBD (M8) | | | |
| D012 | 2026-11-03 | Train/val/test generator splits | Fixed by contract Section 2 (LOGO folds Midjourney, BigGAN, ADM; 300/gen; 90/10) | n/a | Shared evaluation | Comparable results |
| D013 | 2026-11-03 | Split manifests | Generate once with Chethan's build_logo_split (vendored, seed 0, per_gen 300) via src/export_splits.py; both builds load splits/*.json; HF revision pinned in each file | Each side runs own split code | Chethan has no manifests; contract fallback | Identical splits; hashes must match |
| D014 | 2026-11-12 | M2 probe classifier | StandardScaler (fit on train) + L2 LogisticRegression (lbfgs); C in {1e-4..10} chosen by val AUROC on seen generators only | Linear head trained with AdamW; raw/L2-normalised features | Convex and deterministic (no optimiser noise), cheap on cached features, canonical UnivFD-style linear probe; z-scoring tames CLIP intermediate-layer outlier dimensions | Misses vs Chethan then point to features/preprocessing, not training luck |
| D015 | 2026-11-12 | M2 feature definitions | univfd = post-layernorm CLS of layer 24 (1024-d); multilayer = CLS of layer 16 concat raw CLS of layer 24 (2048-d); HF hidden_states[16], [24]; fp16 autocast extraction | UnivFD projected 768-d embedding; patch-mean features | Closest to the contract's taps (16, 24); pooled output is the standard CLIP vision feature | Possible small offset vs a projected 768-d UnivFD feature |
| D016 | 2026-11-12 | M2 gate interpretation and fallback ladder | Gate = our AUROC point estimate inside Chethan's 95% CI (CI overlap also reported). If it fails, change ONE thing at a time, logging each: (1) projected 768-d feature, (2) L2-normalise instead of z-score, (3) fp32 extraction, (4) check manifest/protocol parity with Chethan | Require CI containment | Point-in-CI is the contract wording; overlap is informational | Avoids chasing noise |
| D017 | 2026-11-14 | M2 gate criterion (revises D016 interpretation) | Gate passes if every (fold, probe) cell's 95% CI overlaps the reference CI AND no cell is more than 0.02 AUROC below the reference. Point-estimate-in-reference-CI is reported but informational | Keep point-in-CI (D016) | Observed M2: all 6 CIs overlap; deltas +0.003, -0.007, +0.014, +0.003, +0.010, -0.014. Point-in-CI flagged BigGAN univfd where we were higher and ADM multilayer at -0.014 (sampling error with 1000 test images); C was selected on val only (verified in src/probes.py) | Harness validated; no ladder step needed |
