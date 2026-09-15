"""Generate the August 5, 2026 D-Flow/DLR weekly HTML report."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "outputs/reports"
REPORT_PATH = REPORT_DIR / "dflow_dlr_weekly_report_2026-08-05.html"
CT_DIR = ROOT / "outputs/dflow_sparse_ct_shepp_logan"
DLR_TIMINGS = ROOT / "dlra/outputs/toy_ode_cpu_benchmark/toy_ode_cpu_solver_timings.csv"
DLR_DFLOW_RESULTS = ROOT / "outputs/dlr_dflow_baseline_comparison_15_steps/results.csv"


def read_csv(path: Path) -> list[dict]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    return rows


def select_cases(rows: list[dict], resolution: int) -> list[tuple[str, dict]]:
    selected = sorted(
        (row for row in rows if int(row["resolution"]) == resolution),
        key=lambda row: row["image_mse"],
    )
    return [("Best", selected[0]), ("Median", selected[len(selected) // 2]), ("Worst", selected[-1])]


def fmt(value: float) -> str:
    return f"{value:.3e}"


def build_report() -> str:
    sample_rows = read_csv(CT_DIR / "sparse_ct_samples.csv")
    iteration_rows = read_csv(CT_DIR / "sparse_ct_iterations.csv")
    with (CT_DIR / "settings.json").open() as handle:
        ct_settings = json.load(handle)
    timing_rows = read_csv(DLR_TIMINGS)
    dlr_dflow_rows = read_csv(DLR_DFLOW_RESULTS) if DLR_DFLOW_RESULTS.exists() else []

    ct_accuracy_rows = []
    ct_time_rows = []
    gallery = []
    for resolution in (64, 128):
        samples = [row for row in sample_rows if int(row["resolution"]) == resolution]
        iterations = [row for row in iteration_rows if int(row["resolution"]) == resolution]
        image_mse = [row["image_mse"] for row in samples]
        measurement_mse = [row["measurement_mse"] for row in samples]
        iteration_time = [row["elapsed_s"] for row in iterations]
        ct_accuracy_rows.append(
            f"<tr><td>{resolution}×{resolution}</td><td>{len(samples)}</td>"
            f"<td>{fmt(statistics.mean(image_mse))}</td>"
            f"<td>{fmt(statistics.median(image_mse))}</td>"
            f"<td>{fmt(min(image_mse))}</td><td>{fmt(max(image_mse))}</td>"
            f"<td>{fmt(statistics.mean(measurement_mse))}</td></tr>"
        )
        ct_time_rows.append(
            f"<tr><td>{resolution}×{resolution}</td><td>{len(samples)}</td>"
            f"<td>{statistics.mean(iteration_time):.3f}</td>"
            f"<td>{statistics.median(iteration_time):.3f}</td>"
            f"<td>{min(iteration_time):.3f}</td><td>{max(iteration_time):.3f}</td></tr>"
        )
        cards = []
        for label, row in select_cases(sample_rows, resolution):
            image_path = f"../dflow_sparse_ct_shepp_logan/figures/res_{resolution}/sample_{int(row['sample_idx']):03d}.png"
            cards.append(
                f'<figure><img src="{image_path}" alt="{label} {resolution} sparse CT reconstruction">'
                f"<figcaption><strong>{label}: sample {int(row['sample_idx'])}</strong><br>"
                f"Image MSE {fmt(row['image_mse'])}; measurement MSE {fmt(row['measurement_mse'])}; "
                f"sample time {row['elapsed_s']:.1f} s.</figcaption></figure>"
            )
        gallery.append(
            f"<h4>{resolution}×{resolution}: best, median, and worst by image MSE</h4>"
            f'<div class="gallery">{"".join(cards)}</div>'
        )

    dlr_by_size = {}
    for row in timing_rows:
        dlr_by_size.setdefault(int(row["n"]), {})[row["method"]] = row
    speed_rows = []
    for size in sorted(dlr_by_size):
        euler = dlr_by_size[size]["Forward Euler"]
        dense = dlr_by_size[size]["Dense RK4"]
        dlra = dlr_by_size[size]["DLRA RK4"]
        speedup = dense["median_time_s"] / dlra["median_time_s"]
        speed_rows.append(
            f"<tr><td>{size}×{size}</td><td>4</td><td>800</td>"
            f"<td>{euler['median_time_s']:.6f}</td><td>{dense['median_time_s']:.6f}</td>"
            f"<td>{dlra['median_time_s']:.6f}</td><td>{speedup:.2f}×</td>"
            f"<td>{fmt(euler['endpoint_rel_fro_error'])}</td>"
            f"<td>{fmt(dense['endpoint_rel_fro_error'])}</td>"
            f"<td>{fmt(dlra['endpoint_rel_fro_error'])}</td></tr>"
        )

    family_order = {"circle": 0, "shepp_logan": 1}
    baseline_order = {"exponential": 0, "hutchinson_trace": 1}
    dlr_dflow_rows.sort(
        key=lambda row: (
            family_order[row["family"]], int(row["resolution"]),
            baseline_order[row["baseline"]],
        )
    )
    action_rows = []
    inverse_rows = []
    for row in dlr_dflow_rows:
        family = "Single circle" if row["family"] == "circle" else "Shepp–Logan"
        baseline = "Exponential" if row["baseline"] == "exponential" else "Hutchinson trace"
        action_rows.append(
            f"<tr><td>{family}</td><td>{int(row['resolution'])}×{int(row['resolution'])}</td>"
            f"<td>{baseline}</td><td>{row['alpha1_diagnostic']:.3e}</td>"
            f"<td>{row['gradient_rel_error']:.3f}</td><td>{row['gradient_cosine']:.3f}</td>"
            f"<td>{row['exact_action_time_s']:.2f}</td><td>{row['dlr_action_time_s']:.2f}</td>"
            f"<td>{row['dlr_over_exact_time']:.1f}×</td></tr>"
        )
        inverse_rows.append(
            f"<tr><td>{family}</td><td>{int(row['resolution'])}×{int(row['resolution'])}</td>"
            f"<td>{baseline}</td><td>{int(row['best_step'])}</td>"
            f"<td>{fmt(row['best_measurement_mse'])}</td><td>{fmt(row['best_image_mse'])}</td>"
            f"<td>{row['mean_outer_time_s']:.2f}</td><td>{row['inverse_total_time_s']:.1f}</td></tr>"
        )

    completed_dlr_cases = len(dlr_dflow_rows)
    trace_better_error = 0
    trace_better_cosine = 0
    trace_better_image = 0
    paired_cases = 0
    grouped = {}
    for row in dlr_dflow_rows:
        grouped.setdefault((row["family"], int(row["resolution"])), {})[row["baseline"]] = row
    for pair in grouped.values():
        if set(pair) != set(baseline_order):
            continue
        paired_cases += 1
        trace = pair["hutchinson_trace"]
        exponential = pair["exponential"]
        trace_better_error += trace["gradient_rel_error"] < exponential["gradient_rel_error"]
        trace_better_cosine += trace["gradient_cosine"] > exponential["gradient_cosine"]
        trace_better_image += trace["best_image_mse"] < exponential["best_image_mse"]
    dlr_time_ratios = [row["dlr_over_exact_time"] for row in dlr_dflow_rows]
    trace_errors = [
        row["gradient_rel_error"] for row in dlr_dflow_rows
        if row["baseline"] == "hutchinson_trace"
    ]
    minimum_time_ratio = min(dlr_time_ratios) if dlr_time_ratios else float("nan")
    maximum_time_ratio = max(dlr_time_ratios) if dlr_time_ratios else float("nan")
    median_time_ratio = statistics.median(dlr_time_ratios) if dlr_time_ratios else float("nan")
    minimum_trace_error = min(trace_errors) if trace_errors else float("nan")
    maximum_trace_error = max(trace_errors) if trace_errors else float("nan")

    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>August 5 D-Flow and DLR Weekly Report</title>
<style>
:root{{--ink:#17212b;--muted:#5a6774;--line:#d7dde3;--soft:#f4f7f9;--accent:#245b78;--good:#4d7358}}
body{{margin:0;font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink);background:white}}
main{{max-width:1180px;margin:auto;padding:36px 28px 64px}} h1{{font-size:31px;margin:0 0 6px}}
h2{{font-size:23px;margin:42px 0 14px;border-top:1px solid var(--line);padding-top:20px}}
h3{{font-size:18px;margin:25px 0 10px}} h4{{font-size:15px;margin:22px 0 9px}} p,li{{line-height:1.62}}
.subtitle,.small{{color:var(--muted);font-size:13px}} .goal{{border:1px solid #a9c5d4;border-left:5px solid var(--accent);border-radius:7px;background:#f3f8fa;padding:13px 16px}}
.outcome{{border-left:4px solid var(--good);background:#f1f7f2;padding:12px 15px;line-height:1.6}}
.conclusion{{border-left:4px solid var(--accent);background:#eef5f8;padding:12px 15px;line-height:1.6}}
table{{width:100%;border-collapse:collapse;margin:14px 0 24px;font-size:13px}} th,td{{border-bottom:1px solid var(--line);padding:9px 8px;text-align:right}}
th:first-child,td:first-child{{text-align:left}} th{{background:var(--soft);color:#273442}} .gallery{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:13px}}
.two{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}} figure{{border:1px solid var(--line);border-radius:7px;padding:9px;margin:0;background:white}}
figure img{{width:100%;height:auto;display:block}} figcaption{{color:var(--muted);font-size:12px;line-height:1.5;margin-top:8px}}
code{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}} @media(max-width:800px){{main{{padding:22px 14px}}.gallery,.two{{grid-template-columns:1fr}}table{{display:block;overflow-x:auto}}}}
.equation{{margin:14px 0;padding:12px;text-align:center;background:var(--soft);font-family:Georgia,"Times New Roman",serif;font-size:18px}}
pre{{overflow-x:auto;background:#17212b;color:#f5f7f9;border-radius:7px;padding:15px;line-height:1.5;font-size:12px}}
</style></head><body><main>
<h1>D-Flow and Dynamic Low-Rank Weekly Report</h1>
<p class="subtitle">Week of August 5, 2026</p>

<h2>Section 1. D-Flow accuracy and runtime on sparse-view CT</h2>
<h3>Goal</h3>
<div class="goal">Verify that D-Flow reconstructs Shepp–Logan images accurately from sparse CT measurements, while quantifying its high per-iteration computational cost at 64×64 and 128×128.</div>

<h3>Figures/Tables</h3>
<h4>Model training and dataset generation</h4>
<table><tr><th>Resolution</th><th>Parameters</th><th>Epochs</th><th>Optimizer</th><th>Learning rate</th><th>Batch size</th></tr>
<tr><td>64×64</td><td>2,705,517</td><td>50</td><td>AdamW</td><td>2×10<sup>−4</sup></td><td>512</td></tr>
<tr><td>128×128</td><td>2,705,517</td><td>50</td><td>AdamW</td><td>2×10<sup>−4</sup></td><td>512</td></tr></table>
<p>The datasets were generated reproducibly from <code>skimage.data.shepp_logan_phantom</code>. Random rotation, axis scaling, translation, gamma, and intensity transforms produced 80,000 training, 10,000 validation, and 10,000 test images per resolution.</p>

<h4>Sparse-CT D-Flow experiment</h4>
<p>Five held-out samples were reconstructed at each resolution from {ct_settings['angles']} evenly spaced parallel-beam views. D-Flow used {ct_settings['flow_steps']} Euler steps, LBFGS learning rate {ct_settings['lbfgs_lr']}, <code>max_iter={ct_settings['lbfgs_max_iter']}</code>, history {ct_settings['history_size']}, no line search, no TV regularization, and at most {ct_settings['outer_steps']} outer steps. The stop target was measurement MSE {ct_settings['stop_mse']:.0e}.</p>

<table><tr><th>Resolution</th><th>Samples</th><th>Mean image MSE</th><th>Median image MSE</th><th>Best image MSE</th><th>Worst image MSE</th><th>Mean measurement MSE</th></tr>
{''.join(ct_accuracy_rows)}</table>

{''.join(gallery)}

<h4>Observed time per outer iteration</h4>
<table><tr><th>Resolution</th><th>Samples</th><th>Mean (s)</th><th>Median (s)</th><th>Lowest (s)</th><th>Highest (s)</th></tr>
{''.join(ct_time_rows)}</table>
<p class="small">Timing used synchronized CUDA wall time on GPU 1. The GPU was already carrying an unrelated workload at launch, so these values measure the observed shared-GPU runtime rather than isolated-device performance.</p>

<h3>Outcome</h3>
<div class="outcome"><ul>
<li>Mean image MSE: 2.910×10<sup>−5</sup> at 64×64 and 5.508×10<sup>−5</sup> at 128×128.</li>
<li>Mean measurement MSE remained below 6×10<sup>−8</sup> at both resolutions.</li>
<li>All ten samples used 100 steps; none reached the 10<sup>−12</sup> stop target.</li>
<li>Median time per outer iteration was 3.695 s at 64×64 and 5.297 s at 128×128.</li>
</ul></div>

<h3>Conclusion</h3>
<div class="conclusion">The experiment supports the claim that D-Flow is accurate but computationally slow for sparse-view CT. Reconstruction quality is consistently high, but repeated flow evaluations inside each LBFGS step make the inverse solve expensive, and the current absolute stopping threshold remains ineffective.</div>

<h2>Section 2. Understanding the July toy-ODE accuracy figure</h2>
<h3>Goal</h3>
<div class="goal">Confirm that the dynamical low-rank (DLR) solver accurately evolves a known low-rank matrix ODE, and explain why the lower-right plot is near 10<sup>−2</sup> even though the integration error is near 10<sup>−11</sup>.</div>

<h3>Figures/Tables</h3>
<h4>The complete toy problem</h4>
<p>We solve the linear matrix ODE</p>
<div class="equation"><i>A</i>′(<i>t</i>) = <i>L A</i>(<i>t</i>) + <i>A</i>(<i>t</i>)<i>R</i>, &nbsp;&nbsp; <i>A</i>(0) = <i>A</i><sub>0</sub>, &nbsp;&nbsp; 0 ≤ <i>t</i> ≤ 1.</div>
<p>Its exact dense solution is known:</p>
<div class="equation"><i>A</i>(<i>t</i>) = exp(<i>tL</i>) <i>A</i><sub>0</sub> exp(<i>tR</i>).</div>
<p>The matrices <em>L</em> and <em>R</em> are full 40×40 generator matrices. Their skew-symmetric components rotate the singular vectors, while smaller symmetric components change the singular values. Because both matrix exponentials are invertible, they do not change rank: rank(<em>A</em>(<em>t</em>)) = rank(<em>A</em><sub>0</sub>) for all time.</p>
<p>Rather than evolving all 1,600 entries of <em>A</em>, DLR evolves the rank-4 factorization <em>Y</em>(<em>t</em>) = <em>U</em>(<em>t</em>)<em>S</em>(<em>t</em>)<em>V</em>(<em>t</em>)<sup>T</sup>. The derivative <em>LY + YR</em> is projected onto the tangent space of rank-4 matrices. Both experiments use <strong>n = 40</strong>, <strong>rank r = 4</strong>, and <strong>800 RK4 time steps</strong>.</p>

<h4>Why there are two experiments</h4>
<table><tr><th>Experiment</th><th>Initial singular values</th><th>Question being tested</th></tr>
<tr><td>Top row: exactly rank 4</td><td>1.0, 0.8, 0.6, 0.4; every remaining value is zero</td><td>Can DLR reproduce an ODE solution that lies exactly on the rank-4 manifold?</td></tr>
<tr><td>Bottom row: approximately rank 4</td><td>1.0, 0.8, 0.6, 0.4, followed by a nonzero tail starting at 10<sup>−2</sup></td><td>How well can a rank-4 DLR solution represent a full-rank ODE solution?</td></tr></table>

<p>Three matrices are compared throughout the figure:</p>
<table><tr><th>Symbol</th><th>Definition</th><th>Role</th></tr>
<tr><td><em>A</em>(<em>t</em>)</td><td>The exact, dense analytic solution</td><td>Ground truth</td></tr>
<tr><td><em>X</em>(<em>t</em>)</td><td>The best rank-4 approximation of <em>A</em>(<em>t</em>), computed independently by a truncated SVD at each time</td><td>Best accuracy any rank-4 representation can attain</td></tr>
<tr><td><em>Y</em>(<em>t</em>)</td><td>The rank-4 solution evolved dynamically by DLR</td><td>Solver output</td></tr></table>

<figure><img src="../../dlra/outputs/dlra_toy.png" alt="DLR toy ODE exact-rank and approximate-rank diagnostics"><figcaption>The figure that prompted the accuracy question. The top row tests integration accuracy for an exactly rank-4 solution. The bottom row tests rank-4 approximation of a deliberately full-rank solution.</figcaption></figure>

<h4>Diagnostics used</h4>
<table><tr><th>Diagnostic</th><th>What was compared</th><th>Interpretation</th></tr>
<tr><td>Exact-rank singular values</td><td>The four singular values of exact solution <em>A</em> and DLR solution <em>Y</em></td><td>DLR tracks the evolving singular values of the exactly rank-4 solution.</td></tr>
<tr><td>Exact-rank integration error</td><td>‖<em>Y−A</em>‖<sub>F</sub></td><td>This is the integration-accuracy test; the error is approximately 10<sup>−11</sup>.</td></tr>
<tr><td>Approximate-rank singular values</td><td>The four largest singular values of full-rank <em>A</em> and rank-4 <em>Y</em></td><td>DLR tracks the dominant rank-4 structure.</td></tr>
<tr><td>Approximate-rank errors</td><td>Rank-truncation, DLR-to-best-rank-4, and total errors</td><td>These measure the cost of representing a full-rank solution at rank 4, not only integration accuracy.</td></tr></table>

<h4>Meaning of the three lower-right curves</h4>
<table><tr><th>Curve</th><th>Meaning</th><th>Observed behavior</th></tr>
<tr><td>‖<em>X−A</em>‖<sub>F</sub></td><td>Unavoidable rank-truncation error</td><td>Starts near 1.7×10<sup>−2</sup> because <em>A</em> is full rank but <em>X</em> is restricted to rank 4.</td></tr>
<tr><td>‖<em>Y−X</em>‖<sub>F</sub></td><td>How far DLR is from the best possible rank-4 approximation</td><td>Starts at zero and grows to about 6×10<sup>−3</sup>.</td></tr>
<tr><td>‖<em>Y−A</em>‖<sub>F</sub></td><td>Total error of DLR against the full-rank truth</td><td>Remains close to the unavoidable ‖<em>X−A</em>‖<sub>F</sub> curve.</td></tr></table>

<h4>Numerical summary</h4>
<table><tr><th>Experiment</th><th>True solution rank</th><th>max ‖X−A‖/‖A‖<br><span class="small">rank-4 limit</span></th><th>max ‖Y−X‖/‖A‖<br><span class="small">DLR vs best rank 4</span></th><th>max ‖Y−A‖/‖A‖<br><span class="small">total error</span></th></tr>
<tr><td>Exactly rank 4</td><td>4</td><td>6.184e−15</td><td>2.432e−11</td><td>2.432e−11</td></tr>
<tr><td>Approximately rank 4</td><td>40</td><td>1.134e−2</td><td>3.088e−3</td><td>1.134e−2</td></tr></table>

<h3>Outcome</h3>
<div class="outcome"><ul>
<li>Exact rank-4 integration error: 2.432×10<sup>−11</sup>.</li>
<li>The bottom row compares a rank-4 DLR approximation against a rank-40 truth.</li>
<li>Its approximately 10<sup>−2</sup> error comes from rank truncation, not inaccurate integration.</li>
</ul></div>

<h3>Conclusion</h3>
<div class="conclusion">The figure supports two separate conclusions. First, DLR integrates an exactly rank-4 solution accurately to approximately 10<sup>−11</sup>. Second, when the truth is full rank, the rank-4 DLR solution stays close to the best possible rank-4 approximation. The lower-right plot measures rank-approximation quality and should not be used to judge the integrator's numerical precision.</div>

<h2>Section 3. CPU runtime: Forward Euler, dense RK4, and DLR</h2>
<h3>Goal</h3>
<div class="goal">Compare the time and accuracy of DLR against standard dense Forward Euler and RK4 solvers on the toy matrix ODE.</div>

<h3>Figures/Tables</h3>
<p>All methods integrated the same exactly rank-4 matrix ODE over t∈[0,1] using 800 steps. The benchmark ran on CPU with one BLAS thread, one warm-up run, and three timed repetitions; the table reports median wall time. Endpoint error was computed against the analytic solution exp(L)A(0)exp(R).</p>
<table><tr><th>Matrix size</th><th>Rank</th><th>Steps</th><th>Forward Euler (s)</th><th>Dense RK4 (s)</th><th>DLR-RK4 (s)</th><th>RK4 / DLR</th><th>Euler rel. error</th><th>RK4 rel. error</th><th>DLR rel. error</th></tr>
{''.join(speed_rows)}</table>

<h3>Outcome</h3>
<div class="outcome"><ul>
<li>Forward Euler was fastest, but its relative error was approximately 10<sup>−2</sup>.</li>
<li>Dense RK4 and DLR-RK4 achieved approximately 10<sup>−11</sup> relative error.</li>
<li>DLR was slower at 40×40 and 64×64, then 4.03×, 15.96×, and 22.58× faster than RK4 at 128×128, 256×256, and 512×512.</li>
</ul></div>

<h3>Conclusion</h3>
<div class="conclusion">Forward Euler has the lowest raw runtime but does not provide comparable accuracy at the tested step count. DLR retains approximately 10<sup>−11</sup> accuracy and becomes substantially faster than equally accurate dense RK4 as matrix size grows, with a clear low-rank scaling advantage from 128×128 onward.</div>

<h2>Section 4. DLR + D-Flow sensitivity approximation</h2>
<h3>Goal</h3>
<div class="goal">Implement DLR + D-Flow beginning with the scalar function <em>a</em><sub>t</sub>, derive the factor updates for <em>U</em><sub>t</sub>, <em>S</em><sub>t</sub>, and <em>V</em><sub>t</sub>, and test the resulting approximate gradient for accuracy and speed on single-circle deblurring and Shepp–Logan sparse CT.</div>

<h3>Figures/Tables</h3>
<h4>Problem and notation</h4>
<p>The generative flow satisfies <em>dx</em><sub>t</sub>/<em>dt</em> = <em>v</em><sub>θ</sub>(<em>t</em>,<em>x</em><sub>t</sub>) with initial state <em>x</em><sub>0</sub>=<em>z</em>. Here, <em>t</em> is flow time, <em>z</em> is the source optimized by D-Flow, <em>x</em><sub>t</sub> is the image state, and <em>v</em><sub>θ</sub> is the trained velocity network. The flow sensitivity is</p>
<div class="equation"><i>J</i><sub>t</sub> = ∂<i>x</i><sub>t</sub>/∂<i>z</i>, &nbsp;&nbsp; d<i>J</i><sub>t</sub>/d<i>t</i> = <i>A</i><sub>t</sub><i>J</i><sub>t</sub>, &nbsp;&nbsp; <i>J</i><sub>0</sub>=<i>I</i>, &nbsp;&nbsp; <i>A</i><sub>t</sub>=∂<i>v</i><sub>θ</sub>/∂<i>x</i>.</div>
<p><em>J</em><sub>t</sub> is the Jacobian of the flow with respect to its source, <em>A</em><sub>t</sub> is the Jacobian of the velocity network with respect to its image input, and <em>I</em> is the identity matrix. Neither Jacobian is explicitly formed. Products <em>A</em><sub>t</sub><em>Q</em> and <em>A</em><sub>t</sub><sup>T</sup><em>Q</em> with a thin matrix <em>Q</em> are evaluated by automatic-differentiation JVPs and VJPs.</p>

<h4>What rank tells us—and what it does not</h4>
<p>For a smooth invertible flow, <em>J</em><sub>t</sub> remains full rank because <em>J</em><sub>0</sub>=<em>I</em>. Therefore ordinary rank cannot define <em>a</em><sub>t</sub>. The useful quantity is the effective rank of the residual after removing a full-rank scalar baseline:</p>
<div class="equation"><i>J</i><sub>t</sub> ≈ <i>a</i><sub>t</sub><i>I</i> + <i>Y</i><sub>t</sub>, &nbsp;&nbsp; <i>Y</i><sub>t</sub>=<i>U</i><sub>t</sub><i>S</i><sub>t</sub><i>V</i><sub>t</sub><sup>T</sup>.</div>
<p><em>a</em><sub>t</sub> is one scalar describing the isotropic part of the sensitivity. <em>Y</em><sub>t</sub> is a rank-<em>r</em> correction; <em>U</em><sub>t</sub> and <em>V</em><sub>t</sub> contain <em>r</em> orthonormal directions, while <em>S</em><sub>t</sub> is their <em>r</em>×<em>r</em> core. The selected rank <em>r</em> should be based on decay in the singular values of this residual, not the exact rank of <em>J</em><sub>t</sub>.</p>

<h4>Two designs for <em>a</em><sub>t</sub></h4>
<table><tr><th>Baseline</th><th>Definition</th><th>Interpretation</th><th>Endpoint</th></tr>
<tr><td>Exponential</td><td><em>a</em><sub>t</sub>=1−exp(−(1−<em>t</em>)/ε<sub>a</sub>), ε<sub>a</sub>=0.05</td><td>Simple prescribed decay; it is not fitted to the current flow.</td><td><em>a</em><sub>1</sub>=0, forcing the endpoint approximation to be purely rank <em>r</em>.</td></tr>
<tr><td>Hutchinson trace</td><td>d(log <em>a</em><sub>t</sub>)/d<em>t</em>≈tr(<em>A</em><sub>t</sub>)/<em>d</em>, <em>a</em><sub>0</sub>=1</td><td>Matches average log-volume change. The trace is estimated from eight random products <em>p</em><sup>T</sup><em>A</em><sub>t</sub><em>p</em>.</td><td>Data-dependent and generally nonzero.</td></tr></table>
<p>Here, ε<sub>a</sub> controls the exponential decay width, <em>d</em> is the number of pixels, tr denotes matrix trace, and <em>p</em> is a random probe vector. The Hutchinson method evaluates <em>A</em><sub>t</sub><em>p</em> by a JVP and never stores <em>A</em><sub>t</sub>.</p>

<h4>Deriving the DLR component equations</h4>
<p>Substitution into the sensitivity equation gives the residual rate</p>
<div class="equation"><i>F</i><sub>t</sub> = <i>A</i><sub>t</sub>(<i>a</i><sub>t</sub><i>I</i>+<i>U</i><sub>t</sub><i>S</i><sub>t</sub><i>V</i><sub>t</sub><sup>T</sup>) − (d<i>a</i><sub>t</sub>/d<i>t</i>)<i>I</i>.</div>
<p><em>F</em><sub>t</sub> is the desired time derivative of the low-rank residual. Projecting it onto the tangent space of rank-<em>r</em> matrices gives</p>
<div class="equation">d<i>U</i><sub>t</sub>/d<i>t</i> = (<i>I</i>−<i>U</i><sub>t</sub><i>U</i><sub>t</sub><sup>T</sup>)<i>F</i><sub>t</sub><i>V</i><sub>t</sub><i>S</i><sub>t</sub><sup>−1</sup>,</div>
<div class="equation">d<i>S</i><sub>t</sub>/d<i>t</i> = <i>U</i><sub>t</sub><sup>T</sup><i>F</i><sub>t</sub><i>V</i><sub>t</sub>,</div>
<div class="equation">d<i>V</i><sub>t</sub>/d<i>t</i> = (<i>I</i>−<i>V</i><sub>t</sub><i>V</i><sub>t</sub><sup>T</sup>)<i>F</i><sub>t</sub><sup>T</sup><i>U</i><sub>t</sub><i>S</i><sub>t</sub><sup>−T</sup>.</div>
<p>The superscript −1 denotes a matrix inverse and −<em>T</em> denotes the transpose of that inverse. Because the residual begins near zero, <em>S</em><sub>t</sub> can be nearly singular. The implementation therefore evaluates equivalent K/S/L projector-splitting updates with thin QR factorizations rather than directly using these inverses.</p>

<h4>Pseudocode</h4>
<pre>Input: source z, observation y, trained velocity network, rank r
Initialize x = z
Initialize U and V as orthonormal d-by-r bases; set S = 10^-4 I_r
Set a_0 = 1

For each flow-time step:
    evaluate A_t Q and A_t^T Q using JVPs and VJPs
    compute a_t and da_t/dt using either baseline
    K-step: update K = U S, then QR-factor K to update U
    S-step: update the small r-by-r core S
    L-step: update L = V S^T, then QR-factor L to update V and S
    advance x with one Euler flow step

At t = 1:
    compute terminal loss gradient g_x
    approximate source gradient = a_1 g_x + V S^T U^T g_x
    update z with Adam

Reset the residual near zero and repeat from t = 0 for the new z.</pre>

<h4>CPU experiment structure</h4>
<table><tr><th>Setting</th><th>Value</th></tr>
<tr><td>Cases</td><td>One held-out sample at each single-circle resolution 64, 96, 128, 160 and Shepp–Logan resolution 64, 128</td></tr>
<tr><td>Forward operators</td><td>Gaussian blur, σ=5, for circles; 15-view Radon transform for Shepp–Logan</td></tr>
<tr><td>DLR settings</td><td>Rank 8; 15 flow steps; residual core 10<sup>−4</sup><em>I</em><sub>r</sub>; eight trace probes</td></tr>
<tr><td>Inverse optimizer</td><td>Adam, learning rate 0.02, 10 outer steps</td></tr>
<tr><td>Hardware</td><td>CPU, sequential cases, 16 computation threads</td></tr>
<tr><td>Reference</td><td>Exact <em>J</em><sub>1</sub><sup>T</sup><em>g</em><sub>x</sub> from autograd through the same 15-step Euler flow</td></tr></table>

<h4>Gradient-action accuracy and speed ({completed_dlr_cases}/12 runs complete)</h4>
<table><tr><th>Dataset</th><th>Resolution</th><th><em>a</em><sub>t</sub></th><th><em>a</em><sub>1</sub></th><th>Relative action error</th><th>Cosine</th><th>Exact action (s)</th><th>DLR action (s)</th><th>DLR / exact</th></tr>
{''.join(action_rows)}</table>
<p class="small">Relative action error is ‖exact gradient−DLR gradient‖/‖exact gradient‖. Cosine measures directional agreement: 1 is identical direction, 0 is orthogonal, and −1 is opposite.</p>

<h4>Ten-step inverse-problem results</h4>
<table><tr><th>Dataset</th><th>Resolution</th><th><em>a</em><sub>t</sub></th><th>Best step</th><th>Measurement MSE</th><th>Image MSE</th><th>Mean outer step (s)</th><th>Total (s)</th></tr>
{''.join(inverse_rows)}</table>

<h3>Outcome</h3>
<div class="outcome"><ul>
<li>The Hutchinson baseline had lower gradient-action error in {trace_better_error}/{paired_cases} paired cases and better gradient cosine in {trace_better_cosine}/{paired_cases}.</li>
<li>It produced lower ten-step image MSE in {trace_better_image}/{paired_cases} paired cases and lower measurement MSE in all six.</li>
<li>The DLR action was {minimum_time_ratio:.1f}×–{maximum_time_ratio:.1f}× slower than one exact adjoint action on CPU; the median slowdown was {median_time_ratio:.1f}×.</li>
<li>Even with the better trace baseline, relative action error ranged from {minimum_trace_error:.3f} to {maximum_trace_error:.3f}; rank 8 is not yet consistently accurate.</li>
</ul></div>

<h3>Conclusion</h3>
<div class="conclusion">The Hutchinson trace construction is the better <em>a</em><sub>t</sub> design in this test: it improves action error for every dataset-resolution pair and usually improves gradient direction and reconstruction. However, the present rank-8 implementation is neither consistently accurate nor faster than a single exact adjoint gradient. The next experiment should test higher rank and finer projector-splitting time grids, then evaluate whether reusing one DLR operator for multiple terminal gradients can justify its construction cost.</div>

<p class="small">Sources: <code>outputs/dflow_sparse_ct_shepp_logan/</code>, <code>outputs/fm_unet/shepp-logan-*/</code>, <code>dlra/dlra_toy.ipynb</code>, and <code>dlra/outputs/toy_ode_cpu_benchmark/</code>.</p>
</main></body></html>"""
    section_start = report.find("<h2>Section 4. DLR + D-Flow sensitivity approximation</h2>")
    sources_start = report.find('<p class="small">Sources:', section_start)
    if section_start >= 0 and sources_start >= 0:
        report = report[:section_start] + report[sources_start:]
    return report


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report())
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
