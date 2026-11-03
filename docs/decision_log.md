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
