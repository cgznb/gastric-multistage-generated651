# Release Validation

Date: 2026-09-27. Python 3.12.13, PyTorch 2.12.1+cu130.

| Check | Result |
| --- | --- |
| Original regularization tests plus release configuration tests | 9 passed, 1 CUDA test skipped in CPU-only invocation |
| Selected-arm protocol | Batch size 4, one arm, 20 seeds, 400-epoch cap confirmed |
| Original full experiment matrix | 15 arms retained; unknown arm rejected |
| Configurable local cache roots | Actual authorized cache loaded: 651 records, CT0 `[651,27,768]`, events `[651,3]` |
| Historical selection replay | Highest validation mean AP equals recorded `bs4_baseline`; `test_used` is false |
| Historical selected-arm aggregate coverage | All reported selected-arm metrics cover 20 seeds |

The cache check read local inputs only and did not write patient data into the
repository. CPU tests cover model parity, endpoint gradient boundaries, loss
logging, checkpoint recovery, selection/test gating and report contracts.
The CUDA-specific recovery case was not executed in this release invocation.
No new complete651 training or full clinical inference run was performed.
Historical statistics remain historical, not newly reproduced results.
