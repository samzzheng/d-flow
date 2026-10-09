# Timing-scope correction — 2026-10-09

The earlier64px scaling experiment reported fast DLR0.371s versus uncheckpointed D-Flow1.057s per fresh gradient. The October8 TF32 timing probe used different scope and execution conditions; its3.868s/1.727s numbers should not be interpreted as a matched historical regression test.

Confirmed differences:

- The earlier DLR helper uses the endpoint returned by factor evolution. The TF32 helper first runs a separate100-step canonical forward and then evolves the factors, so it includes an additional forward evaluation. The active sweep can reuse the accepted line-search endpoint for gradient calculation.
- GPU1 was shared with the active sweep in the TF32 tests. Earlier profiling found fast DLR sensitive to GPU work while eager D-Flow had substantial dispatch overhead; different contention penalties are plausible, but the share of this reversal caused by contention has not been isolated.
- The first TF32 helper forced manual GroupNorm statistics at64px, whereas the current sweep uses native statistics at64/96px. The helper now follows the sweep selection. Original measurements remain valid for their recorded code, but they were not an exact reproduction of that sweep backend.
- Checkpoints, thread counts and derivative batching also differ from the earlier scaling protocol. The older helper constructs its graph provider without explicit per-step chunk limits; the sweep uses6/4 JVP/VJP chunk limits. Their individual effects have not been measured here.

## Corrected small benchmark

Same64px n2 epoch200 CT10 source as October8;100 Euler steps, rank8 fast DLR, fixedV, native GroupNorm statistics matching the sweep. Four on/off/off/on blocks, one warmup and two timed calls per method/block, with four measured samples per mode. Exact D-Flow is uncheckpointed. Existing GPU1 workers continued, GPU0 remained unused. No optimizer or line search was run.

| Work timed | TF32 median (s) | FP32 median (s) | FP32 change |
| --- | ---: | ---: | ---: |
| D-Flow | 1.5965 | 1.8364 | +15.0% |
| DLR D-Flow | 3.7006 | 4.6795 | +26.5% |
| DLR cached endpoint | 2.7827 | 3.5518 | +27.6% |
| P-Flow | 0.9429 | 1.0503 | +11.4% |
| objective | 0.9497 | 1.0899 | +14.8% |

The full DLR row includes the canonical forward; the cached-endpoint row excludes it while still evolving fresh factors over all100 flow steps. Their TF32 median difference is0.918s, close to the standalone forward/objective0.950s. Thus roughly one second was counted differently. Cached-endpoint DLR still takes2.783s versus exact D-Flow1.597s in this shared workload: the extra forward does not explain the entire gap.

Background conditions drifted during the run: TF32-on cached DLR averaged2.974s in the first block and2.587s in the last; exact D-Flow1.708s and1.448s. This is not an uncontended speed ranking. A matching idle-GPU benchmark is still required to establish whether an implementation/configuration regression remains. No GPU workers were stopped to obtain idle measurements.

The precision penalty for DLR remained approximately26–28% in the corrected configurations, broadly consistent with the earlier29% result, but these are shared-device estimates. Neither test establishes a universal percentage or a resolution-independent ranking.

Validation: all40 timing outputs finite, alpha exactlyzero, compilation passes, all30 monitored production source hashes unchanged, queue admission resumed. Raw data: `outputs/tf32_speed_20261009/r64_components.json` and `.log`. The helper accepts `--components` for this decomposition. No production algorithm was changed.
