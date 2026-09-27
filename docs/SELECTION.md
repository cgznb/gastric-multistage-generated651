# Historical Configuration Selection

Source experiment: Generated651 complete retraining, 2026-09-26; final results
written 2026-09-27. Fifteen configurations combine batch sizes 16/8/4 with five
architecture/optimization variants. Every configuration uses the same 20 seed
identifiers. Stage 1 is retrained per seed and batch size; stage 2 jointly fits
the world, surgery transition and endpoint heads.

The prespecified rule selected `bs4_baseline` by maximum 20-seed mean validation
mean-endpoint AP: **0.44464777**. `configuration_choice.json` records
`test_used: false`. This is the recommendation criterion; it was not replaced
using test results or a single favorable seed.

Historical test summaries for the selected configuration:

| Endpoint | AUROC mean | Seed SD | AP mean | Seed SD |
| --- | ---: | ---: | ---: | ---: |
| pCR | 0.634541 | 0.072103 | 0.319879 | 0.082620 |
| Recorded recurrence/metastasis | 0.619734 | 0.069955 | 0.384830 | 0.100185 |

These are existing results, not a new run performed for this release.
All configurations' aggregate results remain in `reference/seed_summary.csv`.
This is a comparatively promising validation-selected configuration within
the existing study, not proof of superiority over every baseline. Test sets
overlap across seeds and the cohort has repeatedly participated in development.

Default training in this release runs only this selected configuration.
`--arm all` retains the original 15-configuration matrix. A new selected-only
run cannot reproduce the original configuration-selection comparison by itself.
