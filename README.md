# csm-clip: Concept-Steered Mamba-CLIP (our independent v1 build)

Independent rebuild of the architecture in `Concept_Steered_Mamba_CLIP_AIGI_Detection_Architecture.docx`,
built to be compared against Chethan's build under a shared evaluation contract.

- Contract and gates: `docs/contract.md`
- Open items needing agreement: `docs/m0_open_items.md`
- Decisions (log the moment you decide): `docs/decision_log.md`
- Runs (never delete failed runs): `docs/experiment_log.md`

## Layout

```
docs/        contract, decision log, experiment log, M0 checklist
splits/      canonical split manifests (image IDs), shared by both builds
src/         model + data + eval code (added from M1 onward)
tests/       CPU unit tests with small stand-in models
notebooks/   one Colab smoke notebook per module
runs/        results JSON per run (checkpoints live on Google Drive, not here)
```

## Rules (from the contract)

1. Shared evaluation, independent models. Do not look at the other build's model code until v1 is done.
2. v1 is spec-faithful; every open spec choice is logged in the decision log.
3. v2 changes one thing at a time.
4. The held-out generator is touched once per run, for the final test only.
5. Gate passes -> update Status in the contract and tag (`m0-done`, `m1-done`, ...).

## Environment

Colab free tier (T4). Code on GitHub (private), token in Colab Secret only. Checkpoints/outputs on Drive.
Training scripts must resume from a Drive checkpoint. Keep each experiment under ~1 hour.
