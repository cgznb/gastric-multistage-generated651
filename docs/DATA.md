# Required Local Inputs

The cache loader needs two externally supplied files:

- `$GENERATED651_SOURCE_ROOT/artifacts/pool/pool.pt`
- `$GENERATED651_EVENT_ROOT/artifacts/pool/events.pt`

The source pool is the complete651 `generated700_data.Pool` dictionary. It
contains identifiers used only for local joins and patient splits, baseline
clinical/treatment dictionaries, intervals, CT0 and CT1 feature tensors,
endpoint labels, validity masks and artifact provenance. CT feature grids are
`[651,27,768]`. The event cache contains the same 651 identifiers, an int64
`[651,3]` event tensor and the matching source-pool artifact identifier.

`load_verified_pool` joins by identity and verifies cohort, feature masks,
finite values and the surgery column. Do not edit the provenance fields to
bypass a mismatch. Fit preprocessing, CT statistics and logistic anchors only
on each seed's training patients. The original split sizes are 521/65/65.

Only `pool.pt` and `events.pt` are needed by the selected cache-based training
path. The original cache creation and CT preprocessing modules remain in
`src/stageworld/`; rebuilding them also requires authorized raw data, the
corresponding original configuration and external encoder weights.

New inference bundles contain fitted transforms and model weights, so clients
need only current clinical/treatment inputs, an interval scenario, CT0 features
and surgery status. CT1 and outcomes are never prediction inputs.
