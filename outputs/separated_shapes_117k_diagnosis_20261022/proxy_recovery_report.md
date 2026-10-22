# P-Flow fallback pilot — 2026-10-22

The safeguarded P-Flow fallback did **not** rescue any of the three tested DLR endpoints. All proxy searches were rejected. Resetting L-BFGS alone also gave no meaningful recovery. These are three selected endpoints from **two samples**, not three independent reconstruction trials.

## Paired continuations

All cases use the separated-shapes 117k model at 250², noiseless CT10 or MRI4×, five Euler steps, FP32 with TF32 disabled, and unconstrained sources. Each case preserves its parent rank settings: fixed rank32 for CT and adaptive32–128 for MRI. The three arms start at the exact same saved source within each case; the model, acquisition, measurements and remaining original1000-attempt budget are preserved.

1. **Reset DLR:** clear L-BFGS history and cached gradient, recompute DLR at the parent source.
2. **One proxy then DLR:** try one safeguarded P-Flow update; after acceptance, reset L-BFGS and recompute DLR at the new source. If rejected, preserve the parent source.
3. **Proxy-only:** repeat the same safeguarded P-Flow update, without L-BFGS, until the original two-consecutive-rejection criterion or remaining budget is reached. This is a plain proxy control, not the earlier P-Flow+L-BFGS implementation.

The original previous-V subspace is retained only as an initialization sketch seed. No previous gradient or curvature pair crosses the proxy/DLR switch. Native derivative execution is used throughout the continuations; the reset arm also controls for restarting that execution path.

## Image relative L2 error (%)

| Endpoint | Before continuation | Reset DLR | One proxy then DLR | Proxy-only |
|---|---:|---:|---:|---:|
| MRI sample2 | 99.7683 | 99.7683 | 99.7683 | 99.7683 |
| CT sample0, original Gaussian initialization | 105.3665 | 105.3619 | 105.3665 | 105.3665 |
| CT sample0, measurement-informed initialization | 37.1184 | 37.1184 | 37.1184 | 37.1184 |

The two CT rows reuse the same sample with different optimization endpoints. The small reset-only CT improvement is 0.0046 percentage points, not a successful recovery. Final observed measurement residuals are99.6587%,85.0086%,15.8497%, respectively; reset-only CT differs by less than0.00001 percentage points.

The parent attempts were240,39,157. Reset DLR used another2,3,2 attempts; the one-proxy arm used one attempt each; proxy-only used two rejected attempts each. They stopped by the retained rejection safeguard, not by exhausting the remaining budget. One fallback search cost about2.5–2.7seconds on shared GPU1; complete reset/proxy-only continuations took about5–9seconds. These are pilot wall times, not a throughput benchmark.

## Acceptance and what failed

Each proxy search starts at displacement \(0.01\sqrt d=2.5\), then halves at most30times. It evaluates the actual objective \(\Phi(z)=\tfrac12\|AG(z)-y\|^2\); no proxy slope enters acceptance. With three baseline evaluations \(\Phi_i\), a candidate must pass twice:

\[
\Phi(z_{\mathrm{trial}})<\min_i\Phi_i-\delta,
\qquad
\delta=\max\left(4(\max_i\Phi_i-\min_i\Phi_i),\;8\epsilon_{\mathrm{FP32}}\max(1,\max_i|\Phi_i|)\right).
\]

This is a conservative declared roundoff guard, not a certified bound on all numerical errors. The full-cap proxy increased the objective in every case:

| Endpoint | Parent objective | Full-cap proxy objective | Acceptance margin |
|---|---:|---:|---:|
| MRI sample2 | 171.6301310 | 171.7608518 | 0.00016368 |
| CT sample0, Gaussian | 1.93463924 | 2.03587016 | 0.000001845 |
| CT sample0, conditioned | 0.06725409 | 0.97066956 | 0.000000954 |

No smaller tested step produced a decrease exceeding the guard. Tiny apparent improvements at some MRI trial lengths were below it. The CT proxy gradients were evaluated separately in the two proxy arms: trial objective differences reached3.92e-5 for Gaussian CT and2.55e-5 for conditioned CT, while the parent objectives, trial counts, step lengths within1e-7, and rejection decisions agreed. MRI trial records matched exactly. No claim of bitwise CT gradient reproducibility is made.

**Interpretation:** changing to the proxy direction after these DLR failures did not open an accepted descent route within the tested step range. This does not establish a local minimum, exclude other step lengths, or rule out earlier intervention. It also leaves the main intended mechanism—an accepted proxy step followed by sustained DLR progress—unobserved. Noisy cases and other resolutions remain untested. Keep this experimental; there is no evidence here to enable it by default.

## Verification and artifacts

Three CPU acceptance tests pass: actual-objective acceptance despite a misleading proxy slope, rejection of uphill/sub-roundoff proposals, and backtracking/projection/cap preservation. Saved-array verification checks parent hashes, within-case pairing, unchanged mathematical settings, attempt budgets, monotone histories, complex residual metrics and unchanged rejected sources. No accepted hybrid occurred, so its resume path is implemented but not exercised by these reconstruction pilots. Raw tensors are retained locally and excluded from Git.

- [Machine-readable verification](proxy_recovery_verification.json).
- [MRI continuation panel](proxy_recovery_mri2/comparison.png).
- [Gaussian CT continuation panel](proxy_recovery_ct0/comparison.png).
- [Conditioned CT continuation panel](proxy_recovery_conditioned_ct0/comparison.png).

Panels show the proxy-only and one-proxy-then-DLR endpoints, with D-Flow explicitly unavailable because no exact-gradient continuation was run. Reset-only results appear in the table. Per-case folders contain the parent source/hash, configuration, source-code hashes, full trial logs, histories, evaluation counts and elapsed times. The MRI pilot records the runner hash before CT support was added; the MRI branch's equations and fallback helper were unchanged by that extension.
