"""Build the 2026-09-09 DLR-D-Flow report (RK4 noise test at 200/250; algorithm + speed)."""
from __future__ import annotations
import json, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "outputs/reports"
ASSETS = REPORTS / "2026-09-09_assets"
OUT = REPORTS / "dlr_dflow_report_2026-09-09.html"

noise = json.load(open(ASSETS / "noise_rk4.json"))
picks = json.load(open(ASSETS / "noise_picks.json"))
bench = json.load(open(ASSETS / "final_bench.json"))

RES_N = [200, 250]
RES_B = [64, 96, 128, 160, 200, 250]

def cell(res, meth):
    g = [noise[f"{res}_s{si}_{meth}"] for si in range(5)]
    img = sorted(100 * r["image_rel"] for r in g)
    return dict(best=img[0], med=img[2], worst=img[4], mean=st.mean(img),
                meas_med=st.median([100 * r["meas_rel"] for r in g]),
                s_it=st.mean([r["s_per_iter"] for r in g]))

# ── CSS (matches the 2026-09-01 report) ──────────────────────────────────
CSS = """:root{--ink:#17212b;--muted:#5a6774;--line:#d7dde3;--soft:#f4f7f9;--accent:#245b78;--accent-2:#7a4b2d}
body{margin:0;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:var(--ink);background:white}
main{max-width:1180px;margin:0 auto;padding:36px 28px 64px}
h1{font-size:32px;margin:0 0 10px;line-height:1.15}
h2{font-size:22px;margin:38px 0 14px;padding-top:8px;border-top:1px solid var(--line)}
h3{font-size:17px;margin:24px 0 10px}h4{font-size:15px;margin:18px 0 10px}
p{line-height:1.68;margin:12px 0}ol,ul{line-height:1.68}li{margin:5px 0}
code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
pre{background:var(--soft);border:1px solid var(--line);border-radius:8px;padding:14px 16px;overflow-x:auto;font-size:12.5px;line-height:1.55}
.subtitle{color:var(--muted);font-size:15px;margin-bottom:24px}
.status-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:20px 0 26px}
.status{border:1px solid var(--line);border-radius:8px;padding:14px;background:var(--soft)}
.status strong{display:block;font-size:13px;color:var(--muted);margin-bottom:5px}
.status span{font-size:18px;font-weight:650}
.note{border-left:4px solid var(--accent);padding:12px 16px;background:#eef5f8;margin:22px 0;line-height:1.62}
.warn{border-left-color:var(--accent-2);background:#fbf3ed}
.note.interpretation{border-left-color:#4d7358;background:#f1f7f2}
table{width:100%;border-collapse:collapse;margin:18px 0 28px;font-size:13px;line-height:1.45}
th,td{border-bottom:1px solid var(--line);padding:11px 9px;text-align:right;vertical-align:top}
th:first-child,td:first-child{text-align:left}
th{background:var(--soft);font-weight:650;color:#273442}
.compact-table{width:auto;max-width:100%;display:table}.settings-table{max-width:660px}
.compact-table th,.compact-table td{white-space:nowrap}
.grp-head{text-align:center}
.nz-grid{display:grid;grid-template-columns:104px repeat(3,minmax(0,1fr));gap:10px;align-items:start;margin:14px 0 26px}
.nz-meth{font-weight:650;font-size:12.5px;color:#273442;padding-top:14px}
.nz-grid figure{margin:0}
.nz-grid figcaption{font-size:11px;line-height:1.4;margin-top:5px}
.algo{border:1px solid var(--line);border-top:3px solid var(--accent);border-radius:6px;padding:6px 20px 16px;margin:20px 0 28px;background:#fcfdfe;font-size:13.5px}
.algo .algo-title{font-weight:700;font-size:14px;margin:12px 0 4px;color:#273442}
.algo .algo-io{background:var(--soft);border-radius:5px;padding:10px 14px;margin:12px 0;line-height:1.7}
.algo ol{margin:10px 0;padding-left:26px;line-height:1.85}
.algo ol ol{margin:6px 0;padding-left:24px;list-style-type:lower-alpha}
.algo li{margin:6px 0}
.algo .cmt{color:var(--muted);font-style:italic}
.algo .step{background:#eef5f8;border-radius:4px;padding:1px 6px;font-weight:650;font-size:12px;color:#1d4a61;margin-right:6px}
.samp-head{font-size:13px;font-weight:650;color:#273442;background:var(--soft);border:1px solid var(--line);border-radius:6px;padding:8px;text-align:center}
th.ge,td.ge{background:#e8f1f6}
th.gr,td.gr{background:#f9efe7}
th.ge,th.gr{color:#273442}
.settings-table td:nth-child(2),.settings-table th:nth-child(2){text-align:left}
figure img{width:100%;height:auto;display:block;border-radius:4px}
.card{border:1px solid var(--line);border-radius:8px;padding:12px;background:white;margin:0 0 22px}
figcaption{color:var(--muted);font-size:13px;line-height:1.55;margin-top:10px}
.small{color:var(--muted);font-size:13px}
.math-block{overflow-x:auto;margin:18px 0;text-align:center;line-height:1.8}
.win{font-weight:700}
@media(max-width:850px){main{padding:24px 16px 44px}.status-grid{grid-template-columns:1fr}table{font-size:12px;display:block;overflow-x:auto}.compact-table{display:table}.nz-grid{grid-template-columns:1fr}}"""

HEAD = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>DLR-D-Flow: RK4 Noise Test and Per-Iteration Cost</title>'
        '<script>window.MathJax={tex:{inlineMath:[["\\\\(","\\\\)"]],displayMath:[["\\\\[","\\\\]"]],'
        'processEscapes:true},svg:{fontCache:"global"},options:{skipHtmlTags:["script","noscript","style","textarea","pre","code"]}};</script>'
        '<script defer src="vendor/node_modules/mathjax/es5/tex-svg.js"></script>'
        f'<style>{CSS}</style></head><body><main>')

h = [HEAD]
A = h.append

A('<h1>DLR-D-Flow: RK4 Noise Test and Per-Iteration Cost</h1>')
A('<p class="subtitle">Report of 9 September 2026. Part 1 extends the RK4-versus-Euler noise '
  'study to 200&times;200 and 250&times;250 at 1% noise. Part 2 states the DLR-D-Flow algorithm '
  'in pseudocode and reports the per-iteration cost against exact-gradient D-Flow across six '
  'resolutions under one fixed configuration.</p>')
A('<section class="status-grid">'
  '<div class="status"><strong>Noise test</strong><span>1% at 200 and 250</span></div>'
  '<div class="status"><strong>Integrators</strong><span>Euler / RK4, 100 steps</span></div>'
  '<div class="status"><strong>DLR benchmark</strong><span>64 &rarr; 250, rank 8</span></div>'
  '<div class="status"><strong>Cost ratio</strong><span>19&ndash;41&times; per iteration</span></div>'
  '</section>')

# ══ 1. RK4 noise test ════════════════════════════════════════════════════
A('<h2>1. Noise Test at 200&times;200 and 250&times;250: RK4 versus Forward Euler</h2>')
A('<p><strong>Goal.</strong> Last week the two integrators were compared at 128 and 160 with '
  'matched step counts and no clear advantage for RK4. This week the study is extended to the '
  'two largest grids, 200&times;200 and 250&times;250, at 1% measurement noise. RK4 uses four '
  'velocity evaluations per step against Euler\'s one, so at matched step count it costs about '
  'four times as much; the question remains whether the reduced time-discretization error buys '
  'a corresponding gain in reconstruction quality.</p>')

A('<h3>1.1 Experimental Setup</h3>')
A('<table class="compact-table settings-table"><tr><th>Setting</th><th>Value</th></tr>'
  '<tr><td>Forward operator</td><td>Parallel-beam Radon, 15 angles over \\([0,\\pi)\\)</td></tr>'
  '<tr><td>Resolutions</td><td>200&times;200, 250&times;250</td></tr>'
  '<tr><td>Noise level \\(\\delta\\)</td><td>1% relative \\(L_2\\), added to the sinogram</td></tr>'
  '<tr><td>Integrators</td><td>Forward Euler and RK4, both 100 time steps</td></tr>'
  '<tr><td>Velocity evaluations per solve</td><td>100 (Euler), 400 (RK4)</td></tr>'
  '<tr><td>Optimizer</td><td>LBFGS on the source point \\(z\\), lr 1.0, no line search</td></tr>'
  '<tr><td>LBFGS max_iter / history</td><td>1 / 100</td></tr>'
  '<tr><td>Source prior</td><td>\\(\\lambda\\|z\\|_2^2\\) with \\(\\lambda=10^{-3}\\)</td></tr>'
  '<tr><td>Outer iterations</td><td>150 (fixed budget; the 2% target was reached once)</td></tr>'
  '<tr><td>Selection rule</td><td>Best measurement-objective iterate</td></tr>'
  '<tr><td>Samples per cell</td><td>5 (matched across integrators: same image, same noise, '
  'same \\(z_0\\))</td></tr>'
  '<tr><td>Error metric</td><td>Image relative \\(L_2\\), '
  '\\(\\|\\hat{x}-x_\\star\\|_2/\\|x_\\star\\|_2\\)</td></tr>'
  '<tr><td>Total reconstructions</td><td>20</td></tr></table>')

A('<h3>1.2 Numerical Results</h3>')
A('<p><strong>Table 1. Image relative \\(L_2\\) error as a percentage, with cost.</strong> '
  'Best, median and worst are over the five matched samples in each cell; spread is the sample '
  'standard deviation. The lower median of each pair is shown in bold.</p>')
A('<table><tr><th rowspan="2">Resolution</th>'
  '<th class="ge grp-head" colspan="4">Forward Euler</th>'
  '<th class="gr grp-head" colspan="4">RK4</th></tr>'
  '<tr>'
  '<th class="ge">Best (%)</th><th class="ge">Median (%)</th><th class="ge">Worst (%)</th>'
  '<th class="ge">s / iter</th>'
  '<th class="gr">Best (%)</th><th class="gr">Median (%)</th><th class="gr">Worst (%)</th>'
  '<th class="gr">s / iter</th></tr>')
for res in RES_N:
    e, k = cell(res, "euler"), cell(res, "rk4")
    e_win = e["med"] <= k["med"]
    em = f'<span class="win">{e["med"]:.2f}</span>' if e_win else f'{e["med"]:.2f}'
    km = f'{k["med"]:.2f}' if e_win else f'<span class="win">{k["med"]:.2f}</span>'
    A(f'<tr><td>{res}&times;{res}</td>'
      f'<td class="ge">{e["best"]:.2f}</td><td class="ge">{em}</td>'
      f'<td class="ge">{e["worst"]:.2f}</td><td class="ge">{e["s_it"]:.2f}</td>'
      f'<td class="gr">{k["best"]:.2f}</td><td class="gr">{km}</td>'
      f'<td class="gr">{k["worst"]:.2f}</td><td class="gr">{k["s_it"]:.2f}</td></tr>')
A('</table>')

A('<p><strong>Table 2. Per-sample image relative \\(L_2\\) error (%).</strong> Each row is one '
  'matched pair; the winner of the pair is shown in bold.</p>')
A('<table class="compact-table"><tr><th>Resolution</th><th>Sample</th>'
  '<th class="ge">Forward Euler</th><th class="gr">RK4</th><th>Winner</th></tr>')
def bold(v, is_win):
    return f'<span class="win">{v:.2f}</span>' if is_win else f'{v:.2f}'

for res in RES_N:
    for si in range(5):
        e = 100 * noise[f"{res}_s{si}_euler"]["image_rel"]
        k = 100 * noise[f"{res}_s{si}_rk4"]["image_rel"]
        ew = e < k
        winner = "Euler" if ew else "RK4"
        A(f'<tr><td>{res}&times;{res}</td><td>{si}</td>'
          f'<td class="ge">{bold(e, ew)}</td>'
          f'<td class="gr">{bold(k, not ew)}</td>'
          f'<td>{winner}</td></tr>')
A('</table>')

e2, k2 = cell(200, "euler"), cell(200, "rk4")
e25, k25 = cell(250, "euler"), cell(250, "rk4")
w200 = sum(1 for si in range(5)
           if noise[f"200_s{si}_rk4"]["image_rel"] < noise[f"200_s{si}_euler"]["image_rel"])
w250 = sum(1 for si in range(5)
           if noise[f"250_s{si}_rk4"]["image_rel"] < noise[f"250_s{si}_euler"]["image_rel"])
A(f'<div class="note"><strong>Outcome.</strong> The two resolutions disagree. At '
  f'200&times;200 RK4 wins {w200} of the 5 matched pairs and roughly halves the mean image '
  f'error ({k2["mean"]:.2f}% against {e2["mean"]:.2f}%), with the median falling from '
  f'{e2["med"]:.2f}% to {k2["med"]:.2f}%. At 250&times;250 the means are indistinguishable '
  f'({k25["mean"]:.2f}% against {e25["mean"]:.2f}%) and RK4 wins {w250} of 5. What does hold '
  f'at both sizes is the worst case: Euler produced one badly wrong reconstruction at each '
  f'resolution ({e2["worst"]:.1f}% at 200, {e25["worst"]:.1f}% at 250) while RK4\'s worst was '
  f'{k2["worst"]:.1f}% and {k25["worst"]:.1f}%. All of this costs a measured '
  f'4.0&ndash;4.1&times; per iteration, matching the velocity-evaluation count exactly. '
  f'Neither arm was run to convergence: both stopped at the 150-iteration budget with '
  f'residuals of 2&ndash;8% against a 1% noise floor.</div>')

A('<h3>1.3 Reconstructions</h3>')
A('<p>Best, median and worst of the five samples in each cell, ranked by image relative '
  '\\(L_2\\), with forward Euler on the top row and RK4 below it. Each panel is the standard '
  'diagnostic layout: ground truth, reconstruction and absolute image error on the top row; '
  'observed noisy \\(Ax\\), reconstructed \\(Ax\\) and absolute measurement residual on the '
  'bottom. Error panels use a fixed \\([0,1]\\) scale so panels are comparable across cells.</p>')
for res in RES_N:
    A(f'<h4>1% noise, {res}&times;{res}</h4>')
    A('<div class="nz-grid"><div></div>'
      '<div class="samp-head">Best</div><div class="samp-head">Median</div>'
      '<div class="samp-head">Worst</div>')
    for meth, label in (("euler", "Forward Euler"), ("rk4", "RK4")):
        A(f'<div class="nz-meth">{label}</div>')
        for want in ("best", "median", "worst"):
            q = next(x for x in picks if x["res"] == res and x["method"] == meth
                     and x["label"] == want)
            A(f'<figure class="card"><img src="2026-09-09_assets/{q["file"]}" '
              f'alt="{label} {want} at 1% noise, {res}x{res}">'
              f'<figcaption><strong>Sample {q["sample"]}</strong><br>'
              f'image rel. \\(L_2\\) = {100*q["image_rel"]:.2f}%<br>'
              f'measurement rel. \\(L_2\\) = {100*q["meas_rel"]:.2f}%</figcaption></figure>')
    A('</div>')

# ══ 2. DLR-D-Flow ════════════════════════════════════════════════════════
A('<h2>2. From D-Flow to DLR-D-Flow</h2>')
A('<p>The two methods solve the same optimization problem with the same outer loop. They differ '
  'in exactly one step. This section states D-Flow first, then that one step, then the '
  'machinery behind it, and finally the numerical comparison.</p>')

# ── 2.1 D-Flow ───────────────────────────────────────────────────────────
A('<h3>2.1 D-Flow</h3>')
A('<p>A pretrained flow-matching model transports a source point into an image along</p>')
A('<div class="math-block">\\[\\dot x_t=v_\\theta(t,x_t),\\qquad x_0=z,\\qquad t\\in[0,1],\\]</div>')
A('<p>discretized here as \\(N=100\\) forward-Euler steps of size \\(h=1/N\\). Given noisy, '
  'incomplete data \\(y_\\delta=\\mathcal{A}(x_\\star)+\\eta\\), D-Flow does not optimize the '
  'image. It optimizes the <em>source point</em>, so that every candidate is by construction '
  'something the flow can generate &mdash; that is what enforces the prior:</p>')
A('<div class="math-block">\\[\\min_z\\ \\Phi(z),\\qquad \\Phi(z)=\\tfrac12\\big\\|'
  '\\mathcal{A}(x_1(z))-y_\\delta\\big\\|_2^2+\\lambda\\|z\\|_2^2 .\\]</div>')
A('<p>The chain rule splits the gradient of \\(\\Phi\\) into a cheap piece and an expensive '
  'one:</p>')
A('<div class="math-block">\\[\\nabla_z\\Phi=\\underbrace{J_1^{\\top}g_x}_{\\text{expensive}}'
  '+\;2\\lambda z,\\qquad '
  'g_x=\\underbrace{D\\mathcal{A}(x_1)^{*}\\big(\\mathcal{A}(x_1)-y_\\delta\\big)}'
  '_{\\text{cheap: no flow involved}} .\\]</div>')
A('<ul>'
  '<li>\\(g_x\\in\\mathbb{R}^{d}\\) is the <em>terminal gradient</em>: the derivative of '
  '\\(\\Phi\\) with respect to the generated image, holding the flow fixed.</li>'
  '<li>\\(J_t=\\partial x_t/\\partial z\\in\\mathbb{R}^{d\\times d}\\) is the <em>flow '
  'sensitivity</em>: how a perturbation of the source point changes the state at time \\(t\\). '
  'Applying \\(J_1^{\\top}\\) to \\(g_x\\) pulls the image-space gradient back through the '
  'entire flow into source space.</li>'
  '</ul>')
A('<p>D-Flow performs that pullback <strong>exactly</strong>, by reverse-mode differentiation '
  'through the ODE solver: the forward pass records every intermediate activation of all '
  '\\(N\\) steps, and the backward pass sweeps through them. This is the classical adjoint '
  'computation, and it never forms \\(J_1\\).</p>')
A('<div class="algo">')
A('<div class="algo-title">Algorithm 1 &mdash; D-Flow (exact gradient)</div>')
A('<div class="algo-io"><strong>Input:</strong> data \\(y_\\delta\\); operator '
  '\\(\\mathcal{A}\\); velocity field \\(v_\\theta\\); initial source \\(z_0\\); flow steps '
  '\\(N\\); outer steps \\(K\\)<br>'
  '<strong>Output:</strong> the iterate with the smallest measurement misfit</div>')
A('<ol>')
A('<li><span class="step">outer</span> <strong>for</strong> \\(k=1,\\dots,K\\):<ol>')
A('<li><em>Generate.</em> Run the flow \\(x_1=\\operatorname{flow}(z)\\), recording the '
  'activation tape for all \\(N\\) steps. <span class="cmt">\\(N\\) network evaluations</span></li>')
A('<li><em>Terminal gradient.</em> '
  '\\(g_x=D\\mathcal{A}(x_1)^{*}\\big(\\mathcal{A}(x_1)-y_\\delta\\big)\\).</li>')
A('<li><em>Exact pullback.</em> \\(\\nabla_z\\Phi=J_1^{\\top}g_x+2\\lambda z\\) by reverse-mode '
  'differentiation over the tape. <span class="cmt">about \\(2N\\) further network '
  'evaluations</span></li>')
A('<li><em>Step.</em> \\(z\\leftarrow\\operatorname{LBFGS}(z,\\nabla_z\\Phi)\\); keep '
  '\\(x_1\\) if its misfit is the smallest seen.</li>')
A('</ol></li></ol>')
A('</div>')
A('<div class="note warn"><strong>What this costs.</strong> About \\(3N\\approx300\\) network '
  'evaluations per outer iteration, and an activation tape for the whole flow &mdash; 25.7 GB '
  'at 128&times;128, and beyond the card above that, which is why the runs at 160 and larger '
  'use activation checkpointing (recompute instead of store, at roughly double the '
  'compute).</div>')

# ── 2.2 the one change ───────────────────────────────────────────────────
A('<h3>2.2 DLR-D-Flow: approximating the sensitivity</h3>')
A('<p>DLR-D-Flow keeps Algorithm 1 intact and replaces step (c). Instead of differentiating '
  'back through the solver, it carries a cheap surrogate for the sensitivity <em>forward</em> '
  'alongside the state, so that at \\(t=1\\) an approximate pullback is available immediately. '
  'The surrogate is a scalar multiple of the identity plus a rank-\\(r\\) correction,</p>')
A('<div class="math-block">\\[J_t\\ \\approx\\ \\alpha_tI+Y_t,\\qquad '
  'Y_t=U_tS_tV_t^{\\top},\\qquad U_t,V_t\\in\\mathbb{R}^{d\\times r},\\ '
  'S_t\\in\\mathbb{R}^{r\\times r},\\ r\\ll d,\\]</div>')
A('<p>with \\(\\alpha_t\\in\\mathbb{R}\\) a single tracked scalar. Transposing and applying it '
  'to \\(g_x\\) gives the approximate gradient that replaces the exact one:</p>')
A('<div class="math-block">\\[\\widehat{\\nabla_z\\Phi}\\ =\\ \\alpha_1g_x'
  '\;+\;V_1S_1^{\\top}U_1^{\\top}g_x\;+\;2\\lambda z .\\]</div>')
A('<p>Read right to left, the second term costs three skinny matrix&ndash;vector products: '
  '\\(U_1^{\\top}g_x\\) compresses the image-space gradient to \\(r\\) numbers, '
  '\\(S_1^{\\top}\\) mixes them, and \\(V_1\\) expands back to source space. The first term '
  'passes \\(g_x\\) straight through, scaled. Nothing of size \\(d\\times d\\) is ever formed, '
  'and no tape is kept.</p>')
A('<div class="algo">')
A('<div class="algo-title">Algorithm 2 &mdash; DLR-D-Flow (approximate gradient)</div>')
A('<div class="algo-io"><strong>Input:</strong> as Algorithm 1, plus rank \\(r\\)<br>'
  '<strong>Output:</strong> the iterate with the smallest measurement misfit</div>')
A('<ol>')
A('<li><span class="step">outer</span> <strong>for</strong> \\(k=1,\\dots,K\\):<ol>')
A('<li><em>Generate and evolve together.</em> Run the flow '
  '\\(x_1=\\operatorname{flow}(z)\\) with <strong>no tape</strong>, and on the same time grid '
  'evolve the factors \\((U_t,S_t,V_t)\\) and the scalar \\(\\alpha_t\\) from '
  '\\(\\alpha_0=1,\\ Y_0=0\\). <span class="cmt">detailed in &sect;2.4&ndash;2.5</span></li>')
A('<li><em>Terminal gradient.</em> '
  '\\(g_x=D\\mathcal{A}(x_1)^{*}\\big(\\mathcal{A}(x_1)-y_\\delta\\big)\\). '
  '<span class="cmt">identical to Algorithm 1</span></li>')
A('<li><em>Approximate pullback.</em> '
  '\\(\\widehat{\\nabla_z\\Phi}=\\alpha_1g_x+V_1S_1^{\\top}U_1^{\\top}g_x+2\\lambda z\\). '
  '<span class="cmt">three skinny mat-vecs; this is the only step that differs</span></li>')
A('<li><em>Safeguarded step.</em> '
  '\\(z\\leftarrow\\operatorname{LBFGS}(z,\\widehat{\\nabla_z\\Phi})\\); if \\(\\Phi\\) is not '
  'finite or exceeds four times the best value so far, roll back to the best iterate, halve '
  'the step size and discard the curvature history.</li>')
A('</ol></li></ol>')
A('</div>')

# ── 2.3 the difference ───────────────────────────────────────────────────
A('<h3>2.3 The difference in one place</h3>')
A('<table class="compact-table"><tr><th></th><th>D-Flow</th><th>DLR-D-Flow</th></tr>'
  '<tr><td>Objective \\(\\Phi\\)</td><td>identical</td><td>identical</td></tr>'
  '<tr><td>Forward flow</td><td>100 Euler steps</td><td>100 Euler steps</td></tr>'
  '<tr><td>Terminal gradient \\(g_x\\)</td><td>exact</td><td>exact (same formula)</td></tr>'
  '<tr><td>Pullback through the flow</td><td>exact \\(J_1^{\\top}g_x\\)</td>'
  '<td>\\(\\alpha_1g_x+V_1S_1^{\\top}U_1^{\\top}g_x\\)</td></tr>'
  '<tr><td>How it is obtained</td><td>reverse sweep over a stored tape</td>'
  '<td>forward evolution of \\(r\\)-column factors</td></tr>'
  '<tr><td>Network evaluations / iteration</td><td>\\(\\approx3N\\)</td>'
  '<td>\\(\\approx N\\cdot O(r)\\)</td></tr>'
  '<tr><td>Memory</td><td>tape for all \\(N\\) steps, \\(O(Nd)\\)</td>'
  '<td>factors only, \\(O(dr+r^2)\\)</td></tr>'
  '<tr><td>Direction quality</td><td>the gradient, exactly</td>'
  '<td>its projection onto \\(r\\) directions</td></tr>'
  '<tr><td>Safeguard</td><td>not required</td>'
  '<td>required: the surrogate direction is not the gradient of any function</td></tr>'
  '</table>')
A('<div class="note interpretation"><strong>The trade.</strong> D-Flow spends memory to get the '
  'gradient exactly and cheaply: the tape is what lets one reverse sweep deliver the full-rank '
  'action \\(J_1^{\\top}g_x\\) for the price of two forward passes. DLR-D-Flow refuses the '
  'tape, so nothing can be reused and the sensitivity information has to be rebuilt from '
  'Jacobian products at every one of the \\(N\\) flow steps &mdash; which is why it is more '
  'expensive per iteration, not less. What it gains is memory independent of \\(N\\), and a '
  'reusable operator rather than a single vector. What it risks is bias: the pullback keeps '
  'only \\(r\\) directions, so any part of the true gradient lying outside them is discarded '
  'rather than merely perturbed.</div>')

# ── 2.4 building the approximation ───────────────────────────────────────
A('<h3>2.4 How the approximation is built</h3>')

A('<h4>Why a plain rank-\\(r\\) model is not enough</h4>')
A('<p>The obvious idea is to compress the sensitivity directly, '
  '\\(J_t\\approx U_tS_tV_t^{\\top}\\). This fails at the very first step. The flow map of a '
  'smooth ODE is locally invertible, so \\(J_t\\) is generally of full rank, and at '
  '\\(t=0\\)</p>')
A('<div class="math-block">\\[J_0=I,\\]</div>')
A('<p>because the state <em>is</em> the source point there, so perturbing \\(z\\) perturbs '
  '\\(x_0\\) one-for-one. No rank-\\(r\\) matrix approximates the identity when '
  '\\(r\\ll d\\).</p>')
A('<p>The assumption that <em>is</em> justified is weaker: not that \\(J_t\\) is '
  'rank-deficient, but that its <em>action</em> is effectively low-rank &mdash; the part that '
  'matters here, \\(J_t^{\\top}g_x\\), is concentrated in a few directions. That is the same '
  'intuition that makes D-Flow work at all, namely that differentiating through the flow '
  'projects the terminal gradient onto dominant data-manifold directions. Hence a low-rank '
  '<em>plus baseline</em> model: the scalar carries the full-rank part the factorization '
  'structurally cannot, and the rank-\\(r\\) residual carries the directional structure.</p>')

A('<h4>The scalar</h4>')
A('<p>Starting from \\(\\alpha_0=1,\\ Y_0=0\\) reproduces \\(J_0=I\\) exactly, so the residual '
  'only has to describe how the sensitivity <em>deviates</em> from a multiple of the identity. '
  'Holding \\(\\alpha_t\\equiv1\\) gives a poor sensitivity action on these flows, which are '
  'strongly contractive; instead \\(\\alpha_t\\) is the scalar that best matches \\(J_t\\) in '
  'Frobenius norm, which is the mean diagonal entry</p>')
A('<div class="math-block">\\[\\alpha_t=\\frac{\\operatorname{tr}(J_t)}{d}.\\]</div>')
A('<p>The trace is not directly available, so it is estimated stochastically: draw \\(k_p\\) '
  'fixed \\(\\pm1\\) probes \\(p_i\\), carry them along the flow by the same sensitivity '
  'equation so that \\(q_i(t)=J_tp_i\\), and take</p>')
A('<div class="math-block">\\[\\alpha_t\\approx\\frac{1}{k_pd}\\sum_{i=1}^{k_p}'
  'p_i^{\\top}q_i(t),\\qquad\\dot\\alpha_t\\approx\\frac{1}{k_pd}\\sum_{i=1}^{k_p}'
  'p_i^{\\top}A_tq_i(t),\\]</div>')
A('<p>unbiased because \\(\\mathbb{E}\\,p^{\\top}Mp=\\operatorname{tr}(M)\\) for such probes. '
  'Propagating the probes costs one extra Jacobian product per flow step, and '
  '\\(\\dot\\alpha_t\\) comes free from that same product.</p>')

A('<h4>What the residual has to satisfy</h4>')
A('<p>Differentiating the flow ODE with respect to \\(z\\) shows that the sensitivity obeys its '
  'own matrix-valued ODE, the variational equation,</p>')
A('<div class="math-block">\\[\\dot J_t=A_tJ_t,\\qquad J_0=I,\\qquad '
  'A_t=\\partial_xv_\\theta(t,x_t),\\]</div>')
A('<p>where \\(A_t\\) is the Jacobian of the velocity field along the current trajectory. At '
  '250&times;250, \\(d=62{,}500\\), so \\(A_t\\) and \\(J_t\\) are each '
  '\\(62{,}500\\times62{,}500\\) and neither can be formed &mdash; \\(A_t\\) is a neural '
  'network Jacobian and is never available as an array. What <em>is</em> cheap is its action on '
  'a few vectors: \\(A_tQ\\) by a forward-mode derivative (JVP) and \\(A_t^{\\top}Q\\) by a '
  'reverse-mode one (VJP), for skinny \\(Q\\). Everything below uses only those two '
  'products.</p>')
A('<p>Substituting the surrogate into the variational equation gives the dynamics the residual '
  'must follow:</p>')
A('<div class="math-block">\\[\\dot Y_t=F(t,Y_t):=A_t\\big(\\alpha_tI+Y_t\\big)'
  '-\\dot\\alpha_tI.\\]</div>')

A('<h4>Keeping the residual on the rank-\\(r\\) manifold</h4>')
A('<p>\\(F\\) generally points off the rank-\\(r\\) manifold \\(\\mathcal{M}_r\\), so the '
  'Koch&ndash;Lubich construction takes the velocity <em>within</em> the tangent space closest '
  'to the true one,</p>')
A('<div class="math-block">\\[\\dot Y_t=P_{T_{Y_t}\\mathcal{M}_r}F(t,Y_t),\\qquad '
  'P_Y(Z)=ZVV^{\\top}-UU^{\\top}ZVV^{\\top}+UU^{\\top}Z.\\]</div>')
A('<p>Expanding this factor by factor, with the gauge conditions \\(U^{\\top}\\dot U=0\\) and '
  '\\(V^{\\top}\\dot V=0\\) that make the representation unique and preserve orthonormality, '
  'gives</p>')
A('<div class="math-block">\\[\\dot S=U^{\\top}ZV,\\qquad\\dot U=(I-UU^{\\top})ZVS^{-1},'
  '\\qquad\\dot V=(I-VV^{\\top})Z^{\\top}US^{-\\top},\\qquad Z:=F(t,Y).\\]</div>')
A('<div class="note warn"><strong>Why these are not integrated directly.</strong> Both factor '
  'equations contain \\(S^{-1}\\), and the core is ill-conditioned exactly when the '
  'approximation is nearly of lower rank &mdash; the normal situation here, and precisely the '
  'situation at \\(t=0\\) where \\(Y_0=0\\). A direct integrator would need a step size tied to '
  'the smallest singular value of \\(S\\).</div>')
A('<p>The Lubich&ndash;Oseledets projector-splitting integrator removes \\(S^{-1}\\) entirely '
  'by splitting the projection into its three terms and integrating each in turn over one flow '
  'step, with \\(A_t\\) and \\(\\alpha_t\\) frozen:</p>')
A('<div class="math-block">\\[P_Y(Z)=\\underbrace{ZVV^{\\top}}_{\\text{K-step}}'
  '\\underbrace{-\\,UU^{\\top}ZVV^{\\top}}_{\\text{S-step}}'
  '+\\underbrace{UU^{\\top}Z}_{\\text{L-step}}\\]</div>')
A('<p>The K-step finds the new column space, the L-step the new row space, and the S-step runs '
  '<em>backwards</em> between them to remove the double counting. For the scalar-plus-low-rank '
  'model the three substeps are</p>')
A('<div class="math-block">\\[\\dot K=\\alpha_tA_tV+A_tK-\\dot\\alpha_tV,\\qquad '
  '\\dot S=-\\alpha_tU_1^{\\top}A_tV-(U_1^{\\top}A_tU_1)S+\\dot\\alpha_tU_1^{\\top}V,\\]</div>')
A('<div class="math-block">\\[\\dot L=\\alpha_tA_t^{\\top}U_1'
  '+L\\,(U_1^{\\top}A_t^{\\top}U_1)-\\dot\\alpha_tU_1,\\]</div>')
A('<p>each started from the current factors and closed by a thin QR factorization. Every '
  'occurrence of \\(A_t\\) acts on a matrix with only \\(r\\) columns.</p>')

A('<h4>Starting the recursion</h4>')
A('<p>One difficulty remains at \\(t=0\\): the residual starts at \\(Y_0=0\\), which has no '
  'singular vectors, so the factorization is undefined there. Seeding \\(S_0=\\varepsilon I\\) '
  'on a random basis fails badly &mdash; with \\(S\\approx0\\) the tangent projection is '
  'degenerate and the basis never rotates toward the directions that carry the gradient. The '
  'implementation instead defers initialization by one Euler step and forms a randomized sketch '
  'of the first genuinely nonzero residual,</p>')
A('<div class="math-block">\\[Y_h\\approx hA_0+(1-\\alpha_h)I,\\]</div>')
A('<p>then at each subsequent step augments the bases with fresh forcing directions before '
  'recompressing back to rank \\(r\\), which keeps the basis from going stale as the trajectory '
  'turns. This single change reduced the relative gradient error at 64&times;64 from 0.85 to '
  '0.13.</p>')

# ── 2.5 the inner loop ───────────────────────────────────────────────────
A('<h3>2.5 One flow step of the factor evolution</h3>')
A('<p>This is step (a) of Algorithm 2, expanded. It is what runs 100 times per outer '
  'iteration.</p>')
A('<div class="algo">')
A('<div class="algo-title">Algorithm 3 &mdash; factor evolution, one step from \\(t\\) to '
  '\\(t+h\\)</div>')
A('<div class="algo-io"><strong>Have:</strong> state \\(x\\), factors \\((U,S,V)\\), scalar '
  '\\(\\alpha\\), probes \\(Q\\)<br>'
  '<strong>Use only:</strong> \\(A_tQ\\) (JVP) and \\(A_t^{\\top}Q\\) (VJP)</div>')
A('<ol>')
A('<li><em>K-step.</em> \\(K_1=US+h\\big(\\alpha A_tV+A_tUS-\\dot\\alpha V\\big)\\), then '
  '\\(U_1\\hat S=\\operatorname{qr}(K_1)\\). <span class="cmt">new column space</span></li>')
A('<li><em>S-step.</em> with \\(M=U_1^{\\top}A_tU_1\\), '
  '\\(\\tilde S=\\hat S+h\\big(-\\alpha U_1^{\\top}A_tV-M\\hat S'
  '+\\dot\\alpha U_1^{\\top}V\\big)\\). <span class="cmt">backward substep</span></li>')
A('<li><em>L-step.</em> \\(L_1=V\\tilde S^{\\top}+h\\big(\\alpha A_t^{\\top}U_1'
  '+V\\tilde S^{\\top}M^{\\top}-\\dot\\alpha U_1\\big)\\), then '
  '\\(V_1S_1^{\\top}=\\operatorname{qr}(L_1)\\). <span class="cmt">new row space</span></li>')
A('<li><em>Enrich and recompress.</em> Augment \\(U_1,V_1\\) with \\(p\\) fresh forcing '
  'directions and truncate the augmented core back to rank \\(r\\) by a small SVD.</li>')
A('<li><em>Advance.</em> \\(x\\leftarrow x+h\\,v_\\theta(t,x)\\); propagate the probes and '
  'update \\(\\alpha,\\dot\\alpha\\) by the Hutchinson estimates above.</li>')
A('</ol>')
A('<p class="small" style="margin:4px 0 0">Per step: \\(O(r+k_p+p)\\) Jacobian products through '
  'the velocity network, two thin QR factorizations of \\(d\\times r\\) matrices, and '
  '\\(r\\times r\\) core algebra. Storage \\(O(dr+r^2)\\), independent of \\(N\\).</p>')
A('</div>')
A('<p><strong>Settings used in the experiment below:</strong> \\(r=8\\), \\(N=100\\), '
  '\\(K=100\\), \\(k_p=8\\) probes, \\(p=4\\) enrichment directions, \\(\\lambda=10^{-3}\\), '
  'LBFGS learning rate 1.0 with history 100. The terminal scalar is set to \\(\\alpha_1=0\\) in '
  'the pullback: these flows are strongly contractive, \\(\\alpha_t\\) decays from 1 to order '
  '\\(10^{-3}\\) by \\(t=1\\), and there \\(J_1\\) is numerically low rank so the identity term '
  'is mostly error. A diagnostic this week qualifies that default &mdash; at iterates that have '
  'drifted off the data distribution the spectrum of \\(J_1\\) flattens, and retaining '
  '\\(\\alpha_1\\) halves the gradient action error (relative error 0.23 &rarr; 0.10, cosine '
  '0.976 &rarr; 0.995) at no extra cost, since \\(\\alpha_t\\) is already tracked.</p>')

# ── 2.6 how good is the surrogate + honest framing ───────────────────────
A('<h3>2.6 How close the surrogate gradient is, and what the method claims</h3>')
A('<p>The quantity to check is not how well \\(\\alpha_1I+Y_1\\) approximates \\(J_1\\) as a '
  'matrix, but how well its action reproduces the gradient. On fresh factors at rank 8 the '
  'relative action error</p>')
A('<div class="math-block">\\[\\frac{\\big\\|J_1^{\\top}g_x-\\big(\\alpha_1g_x'
  '+V_1S_1^{\\top}U_1^{\\top}g_x\\big)\\big\\|_2}{\\|J_1^{\\top}g_x\\|_2}\\]</div>')
A('<p>is 0.066 with cosine 0.998 against the exact gradient, falling to 0.030 and 0.9997 at '
  'rank 16. The direction is therefore accurate where the factors are fresh, which is why they '
  'are rebuilt from \\(t=0\\) at every outer iteration: measurements show the alignment decays '
  'within about two iterations if they are reused.</p>')
A('<div class="note warn"><strong>The comparison to keep in mind.</strong> For a single scalar '
  'objective and a single gradient evaluation, the classical adjoint equation '
  '\\(\\dot a_t=-A_t^{\\top}a_t,\\ a_1=g_x\\), already returns the exact gradient without ever '
  'forming \\(J_t\\) &mdash; and that is precisely what reverse-mode differentiation implements '
  'in Algorithm 1. DLR-D-Flow should therefore not be advertised as automatically cheaper than '
  'one exact adjoint gradient when \\(r>1\\). Its computational role is more specific: it '
  'removes the full-Jacobian and repeated-SVD bottleneck, and it produces a <em>reusable</em> '
  'low-rank sensitivity operator, so several gradient or trial-step applications can be made '
  'cheaply once the factors exist. The experiment below measures it against the single-vector '
  'adjoint, which is the hardest baseline for this method and the one that matters in '
  'practice.</div>')
A('<div class="note interpretation"><strong>What it does and does not guarantee.</strong> The '
  'method removes the full-SVD bottleneck; it does <em>not</em> make the exact flow Jacobian '
  'low rank. It is useful when the update only needs the dominant sensitivity directions, and '
  'if the forward operator needs information in directions the rank-\\(r\\) model discards the '
  'update becomes biased. The factors are therefore best treated as an approximate gradient '
  'mechanism or a structured preconditioner, and an objective-decrease check must stay in the '
  'solver &mdash; which is the safeguard in step (d) of Algorithm 2.</div>')

# ── 2.7 numerical experiment ───────────────────────────────────────────────────────
A('<h3>2.7 Numerical Experiment: Per-Iteration Cost against D-Flow</h3>')
A('<p><strong>Protocol.</strong> One fixed configuration at every resolution &mdash; no '
  'per-resolution tuning, no adaptive mechanisms, no stopping criterion. Both methods solve the '
  'same problem from the same initial source point with the same optimizer, the same 100-step '
  'Euler objective and the same prior, and both run a fixed 100 outer iterations so that '
  'iteration counts and wall times are directly comparable. The two arms differ only in how the '
  'gradient is obtained: exact reverse-mode differentiation through the flow (D-Flow) against '
  'the rank-8 dynamic low-rank surrogate (DLR-D-Flow). Runs were sequential on a single idle '
  'GPU. D-Flow uses activation checkpointing above 128&times;128, without which its autograd '
  'tape does not fit.</p>')
A('<p><strong>Table 4. Per-iteration cost and time to the best iterate, one sample per '
  'resolution.</strong> "It. to best" is the iteration at which the measurement objective was '
  'lowest, out of 100. The better value of each pair is shown in bold.</p>')
A('<table><tr><th rowspan="2">Resolution</th><th rowspan="2">\\(d\\)</th>'
  '<th class="ge grp-head" colspan="4">D-Flow (exact gradient)</th>'
  '<th class="gr grp-head" colspan="4">DLR-D-Flow (rank 8)</th>'
  '<th rowspan="2">Cost ratio</th></tr>'
  '<tr><th class="ge">It. to best</th><th class="ge">s / iter</th>'
  '<th class="ge">Wall to best (s)</th><th class="ge">Total wall (s)</th>'
  '<th class="gr">It. to best</th><th class="gr">s / iter</th>'
  '<th class="gr">Wall to best (s)</th><th class="gr">Total wall (s)</th></tr>')
for res in RES_B:
    d, l = bench[f"{res}_dflow"], bench[f"{res}_dlr"]
    A(f'<tr><td>{res}&times;{res}</td><td>{res*res:,}</td>'
      f'<td class="ge">{d["it_best"]}</td><td class="ge">{d["s_it"]:.2f}</td>'
      f'<td class="ge">{d["t_best"]:.0f}</td><td class="ge">{d["t_total"]:.0f}</td>'
      f'<td class="gr">{l["it_best"]}</td><td class="gr">{l["s_it"]:.1f}</td>'
      f'<td class="gr">{l["t_best"]:.0f}</td><td class="gr">{l["t_total"]:.0f}</td>'
      f'<td>{l["s_it"]/d["s_it"]:.0f}&times;</td></tr>')
A('</table>')

A('<p><strong>Table 5. Accuracy at the best iterate, and peak memory.</strong> Same runs as '
  'Table 4.</p>')
A('<table><tr><th rowspan="2">Resolution</th>'
  '<th class="ge grp-head" colspan="3">D-Flow</th>'
  '<th class="gr grp-head" colspan="3">DLR-D-Flow</th></tr>'
  '<tr><th class="ge">Best meas. rel. (%)</th><th class="ge">Image rel. (%)</th>'
  '<th class="ge">Peak mem (GB)</th>'
  '<th class="gr">Best meas. rel. (%)</th><th class="gr">Image rel. (%)</th>'
  '<th class="gr">Peak mem (GB)</th></tr>')
for res in RES_B:
    d, l = bench[f"{res}_dflow"], bench[f"{res}_dlr"]
    dw = d["best_meas"] < l["best_meas"]
    dm = f'<span class="win">{100*d["best_meas"]:.2f}</span>' if dw else f'{100*d["best_meas"]:.2f}'
    lm = f'{100*l["best_meas"]:.2f}' if dw else f'<span class="win">{100*l["best_meas"]:.2f}</span>'
    di = d["image_rel"] < l["image_rel"]
    dii = f'<span class="win">{100*d["image_rel"]:.2f}</span>' if di else f'{100*d["image_rel"]:.2f}'
    lii = f'{100*l["image_rel"]:.2f}' if di else f'<span class="win">{100*l["image_rel"]:.2f}</span>'
    ck = "&nbsp;<span class='small'>(ckpt)</span>" if d.get("checkpointing") else ""
    A(f'<tr><td>{res}&times;{res}</td>'
      f'<td class="ge">{dm}</td><td class="ge">{dii}</td>'
      f'<td class="ge">{d["peak_mem_gb"]:.2f}{ck}</td>'
      f'<td class="gr">{lm}</td><td class="gr">{lii}</td>'
      f'<td class="gr">{l["peak_mem_gb"]:.2f}</td></tr>')
A('</table>')
A('<p class="small"><strong>(ckpt)</strong> &mdash; above 128&times;128 the exact gradient no '
  'longer fits: the reverse pass needs an activation tape for all 100 flow steps, which grows '
  '4.9 &rarr; 25.7 GB over the first three rows and overflows the 47 GB card at 160. Those runs '
  'therefore use <em>activation checkpointing</em>, keeping only the state at each step '
  'boundary and recomputing each step\'s forward during the backward pass. That is why peak '
  'memory drops to 0.5 GB at 160 instead of continuing to climb &mdash; at the price of running '
  'every forward step twice, which is part of the jump from 1.21 to 2.47 s per iteration in '
  'Table 4. DLR-D-Flow never needs it, since it holds no tape.</p>')

nwin = sum(1 for r in RES_B if bench[f"{r}_dlr"]["best_meas"] < bench[f"{r}_dflow"]["best_meas"])
iwin = sum(1 for r in RES_B if bench[f"{r}_dlr"]["image_rel"] < bench[f"{r}_dflow"]["image_rel"])
A(f'<div class="note"><strong>Outcome.</strong> The cost gap is structural and large: '
  f'DLR-D-Flow needs 19&ndash;41&times; longer per outer iteration and 26&ndash;170&times; '
  f'longer to reach its best iterate. In exchange it reaches a <em>lower</em> measurement '
  f'residual in {nwin} of the 6 resolutions and a lower image error in {iwin} of 6, with the '
  f'margin widening as the grid grows &mdash; at 250&times;250 it reaches 3.93% residual and '
  f'13.74% image error against D-Flow\'s 6.04% and 20.71%. At 64 and 200 its best iterate was '
  f'the last one, so it had not finished descending when the budget ran out.</div>')
A('<div class="note warn"><strong>Where the cost comes from, and the memory caveat.</strong> '
  'The ratio is arithmetic, not implementation slack. Reverse-mode differentiation produces the '
  'full-rank action \\(J_1^{\\top}g_x\\) for about two forward passes, roughly 300 network '
  'evaluations per iteration, by storing an activation tape for all 100 steps. The dynamic '
  'low-rank evolution stores nothing and instead applies \\(A_t\\) to roughly 60 columns at '
  'every one of the 100 flow steps, about 6,000 evaluations. That 20&times; is what Table 4 '
  'measures. Profiling confirmed the implementation is at the convolution roofline at high '
  'resolution: a CUDA-graph rewrite removed a kernel-dispatch bottleneck and gave 4.9&times; at '
  '64&times;64, but mixed precision, channels-last layout, kernel autotuning and a fused '
  'GroupNorm JVP were each measured and gave nothing. The memory advantage also needs '
  'qualifying: DLR-D-Flow uses 14&ndash;17&times; less memory than D-Flow at '
  '\\(\\le128\\), but against the checkpointed D-Flow of the larger grids '
  '(0.5&ndash;2.4 GB against 4&ndash;22 GB) the memory argument does not hold.</div>')

A('<p class="small">Data and scripts: '
  '<code>outputs/reports/2026-09-09_assets/</code> holds the raw results '
  '(<code>noise_rk4.json</code>, <code>final_bench.json</code>) '
  'and every figure; <code>scripts/build_dlr_dflow_report_2026_09_09.py</code> and '
  '<code>scripts/make_report_figures_2026_09_09.py</code> regenerate this page.</p>')
A('</main></body></html>')

OUT.write_text("\n".join(h))
print("wrote", OUT, f"({OUT.stat().st_size/1024:.0f} KB)")
