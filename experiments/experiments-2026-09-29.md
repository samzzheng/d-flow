# Experiments — 2026-09-29

Times are local America/New_York; durations in HH:MM.

| Time | Experiment | Motivation | What Tried / Evidence | Correctness | Timing | Merge decision |
| --- | --- | --- | --- | --- | --- | --- |
| 00:03 | Isolated P-Flow CPU pilot | An identity sensitivity proxy removes all differentiation through the generative flow; positive alignment may preserve useful source descent. | Implement supplied proxy-gradient + spherical projection algorithm. Compare with exact D-Flow and fast rank-8 DLR using 100 Euler steps, matched projected GD, validation-selected fixed rates, and held-out noisy blur problems. Running; results pending. | Three independent CPU tests pass: proxy differs correctly from true source gradient, projection preserves per-sample radius, no flow parameter gradients. | Running | Keep isolated in p_flow; no production changes. |
