# M0 Open Items (gate: contract agreed by both sides)

Each item has a proposed resolution so you can just send it to Chethan for a yes/no.

## 1. Split manifests (source and format)
- Proposal: Chethan exports his existing `splits/*.json` (one file per fold: `midjourney.json`, `biggan.json`, `adm.json`). You can share these as individual files (no zip needed).
- Format: `{ "fold": "...", "seed": 0, "train": [ids], "val": [ids], "test_fake": [ids], "test_real": [ids] }`, IDs = dataset-relative file paths in `TheKernel01/Tiny-GenImage`.
- Fallback if his repo has none: we generate once with a fixed seed in M1 and he adopts ours.

## 2. Seeds and epochs
- Proposal: seeds 0, 1, 2; main experiment uses all three on the first fold, other folds seed 0 (matches contract minimum).
- Proposal: 5 epochs for every arm; headline = final epoch; best-val epoch secondary.

## 3. "No peeking at model code until v1 is done"
- Proposal: confirm as written. Sharing data/eval code and manifests stays allowed.
- Note: the contract's sanity checkpoint (M2) relies on his reported probe numbers, which is numbers-only and does not violate the rule.

## 4. Repo
- [ ] Create private GitHub repo from this folder (`csm-clip`), add docs/contract.md
- [ ] Add Colab token as a Colab Secret (never in a notebook cell)

## M0 gate (PASSED 2026-11-03: Chethan agreed)
Contract agreed by both sides -> mark M0 "Done" in docs/contract.md and tag `m0-done`.
