#!/usr/bin/env python3
"""Build the concise August 19, 2026 D-Flow / DLR meeting report."""

from __future__ import annotations

import csv
import html
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

from scipy.stats import ttest_rel


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "outputs/reports/dflow_dlr_meeting_report_2026-08-19.html"
ASSET_DIR = REPORT_PATH.parent / "2026-08-19_assets"
NOISE_RUNS = {
    1: ROOT / "outputs/dflow_sparse_ct_circles_cpu_equal_nfe_lr01_no_early_stop_rerun",
    2: ROOT / "outputs/dflow_sparse_ct_circles_cpu_equal_nfe_lr01_noise_2pct",
    3: ROOT / "outputs/dflow_sparse_ct_circles_cpu_equal_nfe_lr01_noise_3pct",
}


def read_rows(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def paired(rows):
    groups = defaultdict(dict)
    for row in rows:
        groups[(int(row["resolution"]), int(row["sample_index"]))][row["method"]] = row
    return [value for _, value in sorted(groups.items())
            if set(value) == {"forward_euler", "rk4"}]


def noise_summary(noise: int):
    pairs = paired(read_rows(NOISE_RUNS[noise] / "samples.csv"))

    def values(method, key):
        return [float(pair[method][key]) for pair in pairs]

    euler = values("forward_euler", "image_relative_l2")
    rk4 = values("rk4", "image_relative_l2")
    return {
        "pairs": len(pairs),
        "euler_mean": statistics.mean(euler),
        "rk4_mean": statistics.mean(rk4),
        "euler_median": statistics.median(euler),
        "rk4_median": statistics.median(rk4),
        "rk4_wins": sum(r < e for e, r in zip(euler, rk4)),
        "paired_t_p": float(ttest_rel(euler, rk4).pvalue),
    }


def copy_report_asset(artifact: str, destination: str):
    source = ROOT / artifact
    target = ASSET_DIR / destination
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return "2026-08-19_assets/" + destination


def representative_gallery():
    rows = read_rows(NOISE_RUNS[1] / "samples.csv")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(int(row["resolution"]), row["method"])].append(row)

    sections = []
    for resolution in (64, 96, 128, 160):
        methods = []
        for method, label in (("forward_euler", "Forward Euler"), ("rk4", "RK4")):
            ordered = sorted(
                grouped[(resolution, method)],
                key=lambda row: float(row["image_relative_l2"]))
            chosen = (("Best", ordered[0]), ("Median", ordered[2]), ("Worst", ordered[-1]))
            cards = []
            for position, row in chosen:
                filename = f'noise_1pct/{resolution}/{method}_{position.lower()}.png'
                source = html.escape(copy_report_asset(row["diagnostic_figure"], filename))
                cards.append(
                    f'<figure><img src="{source}" alt="{position} {label} at {resolution}x{resolution}">'
                    f'<figcaption><strong>{position}: sample {row["sample_index"]}</strong><br>'
                    f'Image relative L2 {float(row["image_relative_l2"]):.4f}</figcaption></figure>')
            methods.append(f'<h4>{label}</h4><div class="gallery">{"".join(cards)}</div>')
        sections.append(
            f'<section class="resolution"><h3>{resolution}×{resolution}: best / median / worst</h3>'
            f'{"".join(methods)}</section>')
    return "".join(sections)


def main():
    summaries = {noise: noise_summary(noise) for noise in (1, 2, 3)}
    noise_rows = []
    for noise, summary in summaries.items():
        noise_rows.append(
            f'<tr><td>{noise}%</td><td>{summary["pairs"]}</td>'
            f'<td>{summary["euler_mean"]:.4f}</td><td>{summary["rk4_mean"]:.4f}</td>'
            f'<td>{summary["euler_median"]:.4f}</td><td>{summary["rk4_median"]:.4f}</td>'
            f'<td>{summary["rk4_wins"]}/20</td><td>{summary["paired_t_p"]:.3f}</td></tr>')

    sparse_ct_figure = copy_report_asset(
        "outputs/dlr_dflow_sparse_ct_circle64_rank8_informed/hutchinson_dlr_sparse_ct_result.png",
        "dlr_sparse_ct_result.png",
    )

    document = rf'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>August 19 D-Flow / DLR Meeting Report</title>
<script>window.MathJax={{tex:{{inlineMath:[["\\(","\\)"]],displayMath:[["\\[","\\]"]]}}}};</script>
<script defer src="vendor/node_modules/mathjax/es5/tex-svg.js"></script>
<style>
:root{{--ink:#17212b;--muted:#5a6774;--line:#d7dde3;--soft:#f4f7f9;--accent:#245b78;--good:#4d7358;--warn:#8b542f}}
body{{margin:0;font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink)}}
main{{max-width:1160px;margin:auto;padding:30px 26px 60px}} h1{{font-size:30px;margin:0 0 5px}} h2{{font-size:22px;margin:36px 0 14px;border-top:1px solid var(--line);padding-top:18px}} h3{{font-size:17px;margin:22px 0 9px}} h4{{font-size:14px;margin:17px 0 8px}} p,li{{line-height:1.55}} .subtitle,.small{{color:var(--muted);font-size:13px}}
.goal,.interpretation{{border-left:5px solid var(--accent);background:#eef5f8;padding:12px 15px;border-radius:5px;line-height:1.55}} .interpretation{{border-color:var(--good);background:#f1f7f2}} .warning{{border-color:var(--warn);background:#fbf3ed}}
table{{width:100%;border-collapse:collapse;margin:12px 0 20px;font-size:12.5px}} th,td{{border-bottom:1px solid var(--line);padding:8px;text-align:right}} th:first-child,td:first-child{{text-align:left}} th{{background:var(--soft)}}
.gallery{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}} figure{{border:1px solid var(--line);border-radius:6px;padding:8px;margin:10px 0;background:white}} figure img{{width:100%;height:auto;display:block}} figcaption{{font-size:12px;color:var(--muted);line-height:1.45;margin-top:7px}}
.plot-crop{{overflow:hidden;aspect-ratio:2594/2028}} .plot-crop img{{transform:translateY(-4.475%)}}
.resolution{{border-top:1px solid var(--line);margin-top:24px;padding-top:2px}} .equation{{background:var(--soft);padding:8px 12px;text-align:center;overflow:auto}} pre{{background:#17212b;color:#f5f7f9;padding:13px;border-radius:6px;overflow:auto;font-size:12px;line-height:1.45}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:12px}} @media(max-width:800px){{main{{padding:20px 13px}}.gallery,.two{{grid-template-columns:1fr}}table{{display:block;overflow:auto}}}}
</style></head><body><main>
<h1>D-Flow / DLR + D-Flow Meeting Report</h1><p class="subtitle">August 19, 2026</p>

<h2>1. Sparse CT with noisy circle measurements: Euler versus RK4</h2>
<h3>Goal</h3><div class="goal">Test whether measurement noise makes Forward Euler reconstructions visibly worse than RK4, providing evidence to prefer RK4.</div>
<h3>Experiment</h3><p>Four resolutions (64, 96, 128, 160), five matched samples each, 15 CT angles, CPU execution, LBFGS learning rate 0.1, and 50 outer iterations. Compute was matched: Euler used 20 steps and RK4 used 5 steps, giving 20 velocity evaluations per flow solve for both.</p>
<h3>Results</h3>
<table><tr><th>Noise</th><th>Pairs</th><th>Euler mean image rel. L2</th><th>RK4 mean</th><th>Euler median</th><th>RK4 median</th><th>RK4 wins</th><th>Paired t-test p</th></tr>{''.join(noise_rows)}</table>
<div class="interpretation warning"><strong>Interpretation.</strong> RK4 did not have a statistically significant image-error advantage at 1%, 2%, or 3% noise. It won only 10/20, 10/20, and 6/20 matched comparisons. The hypothesis that noisy measurements would make Euler clearly worse is not supported.</div>
<h3>1% noise: best, median, and worst reconstructions</h3><p class="small">Samples are selected independently for each resolution and solver using image relative L2. Each resolution shows six diagnostic figures.</p>
{representative_gallery()}

<h2>2. DLR + D-Flow</h2>
<h3>Goal</h3><div class="goal">Approximate the flow sensitivity without storing its full \(d\times d\) Jacobian, then use that approximation to solve the inverse problem.</div>

<h3>Low-rank idea</h3>
<p>The flow sensitivity \(J_t=\partial x_t/\partial z\) satisfies \(\dot J_t=A_tJ_t\), where \(A_t\) is the velocity Jacobian. DLR stores only a scalar identity component and a rank-\(r\) correction:</p>
<div class="equation">\[J_t\approx a_tI+U_tS_tV_t^\top.\]</div>
<p>The columns of \(V_t\) are the important source directions, the columns of \(U_t\) are the corresponding output directions, and \(S_t\) records how strongly they are coupled. This replaces a large \(d\times d\) matrix with two \(d\times r\) matrices and one small \(r\times r\) matrix.</p>

<h3>Where \(U_t\), \(S_t\), and \(V_t\) come from</h3>
<p>Let \(Y_t=J_t-a_tI=U_tS_tV_t^\top\). Its desired derivative is \(F_t=A_t(a_tI+Y_t)-\dot a_tI\). Differentiating the factorization gives</p>
<div class="equation">\[\dot Y=\dot U S V^\top+U\dot S V^\top+US\dot V^\top.\]</div>
<p>We keep the columns of \(U\) and \(V\) orthonormal by imposing \(U^\top\dot U=0\) and \(V^\top\dot V=0\). Matching the three components of \(\dot Y\) to the closest rank-\(r\) tangent approximation of \(F\) gives</p>
<div class="equation">\[\dot S=U^\top FV,\qquad \dot U=(I-UU^\top)FV S^{{-1}},\qquad \dot V=(I-VV^\top)F^\top U S^{{-\top}}.\]</div>
<p>In plain language, each step rotates \(U\) and \(V\) toward the directions created by the flow and updates \(S\) with their strengths. The implementation uses a stable projector-splitting update and truncates the factors back to the requested rank.</p>

<h3>Pseudocode</h3>
<pre>initialize the source z and rank-r factors U, S, V
for each flow time t:
    advance the image state x
    evaluate A_t times only the needed skinny matrices
    update U, S, V with projector splitting
    truncate the factors back to rank r

compute the terminal measurement gradient g_x
approximate gradient_z = a(1) g_x + V S^T U^T g_x
update the source z</pre>

<h3>Visual result</h3>
<figure><div class="plot-crop"><img src="{sparse_ct_figure}" alt="Sparse CT ground truth, sinogram measurement, proposed solution, forward projection, residuals, and convergence"></div><figcaption>Sparse CT result. The panels show the ground truth and sinogram measurement, the proposed solution and its Radon transform, the corresponding residuals, and the convergence history.</figcaption></figure>
<div class="interpretation warning"><strong>Result.</strong> The measurement error decreases, but the reconstructed circle remains in the wrong location. Matching the sparse projections did not recover the correct image in this run.</div>

<p class="small">Sources: the three noisy sparse-CT comparison runs and the rank-8 sparse-CT DLR run.</p>
</main></body></html>'''

    REPORT_PATH.write_text(document)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
