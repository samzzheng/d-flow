# Small curvature information at the CT plateau

At the same saved CT200 sample1 source (attempt310), compare the original three-pair L-BFGS history, that history plus the aggregate secant from attempts250–310, and the unchanged direction with cap-started geometric Armijo. Native FP64 endpoint/objective values agree exactly with the saved state; finite differences confirm the directional derivatives.

| Control | Direction norm before cap | Accepted displacement | Objective decrease | Trials |
|---|---:|---:|---:|---:|
| Original history | 8.65603e-08 | 8.65603e-08 | 2.98521e-11 | 1 |
| Append aggregate secant | 37.6392 | 0.125 | 5.31641e-06 | 5 |
| Original history; cap start | 8.65603e-08 | 0.125 | 1.98903e-05 | 5 |

The aggregate secant has s dot y = 4.016e-16 and curvature cosine 0.1445. Its scale is below the optimizer's fixed 1e-10 admission threshold. Appending it produces a genuinely descending direction and restores a useful step size. This is an aggregate secant across 60 steps, not a measurement of every rejected per-step curvature pair.

Cap-starting the unchanged ray gives a larger one-step decrease than appending this pair. Thus the observed tiny proposal can already be addressed by the simpler queued starting-length control. These fixed-point results do not establish reconstruction accuracy or long-term superiority of either approach. No curvature-admission rule has been changed, and no additional reconstruction variant is launched.

The approximate gradient still has 91.24% relative error against the exact gradient here. Recovering useful step lengths does not solve the missing-direction problem.

[Original](ct1_aggregate_secant_baseline.json) · [Aggregate secant](ct1_aggregate_secant.json) · [Cap-started](ct1_cap_started_at310.json)
