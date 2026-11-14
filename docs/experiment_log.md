# Experiment Log

One row per run. Never delete failed runs.

| Run ID | Date | Git commit | Fold | Protocol | Arm | Seed | Epochs | Config changes vs v1 | Test AUROC [95% CI] | AP | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R001 | 2026-11-14 | c6a2cc6 | Midjourney | controlled | frozen-probe-univfd | n/a (deterministic) | n/a | v1 probe, C=0.01 (chosen on val) | 0.829 [0.802, 0.853] | n/a | ref 0.826; M2 gate run |
| R002 | 2026-11-14 | c6a2cc6 | Midjourney | controlled | frozen-probe-multilayer | n/a (deterministic) | n/a | v1 probe, C=0.01 (chosen on val) | 0.868 [0.846, 0.891] | n/a | ref 0.875; M2 gate run |
| R003 | 2026-11-14 | c6a2cc6 | BigGAN | controlled | frozen-probe-univfd | n/a (deterministic) | n/a | v1 probe, C=0.01 (chosen on val) | 0.974 [0.965, 0.981] | n/a | ref 0.960; M2 gate run |
| R004 | 2026-11-14 | c6a2cc6 | BigGAN | controlled | frozen-probe-multilayer | n/a (deterministic) | n/a | v1 probe, C=0.10 (chosen on val) | 0.979 [0.971, 0.987] | n/a | ref 0.976; M2 gate run |
| R005 | 2026-11-14 | c6a2cc6 | ADM | controlled | frozen-probe-univfd | n/a (deterministic) | n/a | v1 probe, C=0.01 (chosen on val) | 0.899 [0.880, 0.916] | n/a | ref 0.889; M2 gate run |
| R006 | 2026-11-14 | c6a2cc6 | ADM | controlled | frozen-probe-multilayer | n/a (deterministic) | n/a | v1 probe, C=0.10 (chosen on val) | 0.963 [0.952, 0.972] | n/a | ref 0.977; M2 gate run |
