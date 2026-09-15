"""Build the 2026-09-01 DLR-D-Flow report (noise study + algorithm)."""
from __future__ import annotations
import csv, glob, json, re, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "outputs/reports"
OUT = REPORTS / "dlr_dflow_report_2026-09-01.html"

# ── data ─────────────────────────────────────────────────────────────────
noise_rows = []
for f in sorted(glob.glob(str(ROOT / "outputs/dflow_sparse_ct_no_line_search_1200_noise_sweep/noise_*/res*.log"))):
    lvl = int(re.search(r"noise_(\d+)", f).group(1))
    for line in open(f):
        m = re.match(r"Completed res=(\d+) sample=(\d+) method=(\S+): "
                     r"image_rel=(\S+) measurement_rel=(\S+) time=(\S+)s", line)
        if m:
            noise_rows.append(dict(noise=lvl, res=int(m.group(1)), sample=int(m.group(2)),
                                   method=m.group(3), image_rel=float(m.group(4)),
                                   meas_rel=float(m.group(5)), time=float(m.group(6))))

bench = list(csv.DictReader(open(ROOT / "outputs/dlr_dflow_sparse_ct_benchmark/results.csv")))
bench_rel = json.load(open(ROOT / "outputs/dlr_dflow_sparse_ct_benchmark/results_rel_l2.json"))
noise_picks = json.load(open(ROOT / "outputs/reports/2026-09-01_assets/noise_picks.json"))
speed_path = ROOT / "outputs/dlr_dflow_noise_benchmark/iteration_speed.json"
speed = json.load(open(speed_path)) if speed_path.exists() else []

RANK_ACC = {64: {4: (.1386, .9918), 8: (.0861, .9966), 16: (.0724, .9975), 32: (.0584, .9984)},
            96: {4: (.2093, .9734), 8: (.1353, .9917), 16: (.1017, .9953), 32: (.0859, .9964)},
            128: {4: (.2340, .9723), 8: (.1848, .9852), 16: (.1670, .9854), 32: (.1450, .9887)},
            160: {4: (.2198, .9786), 8: (.1956, .9821), 16: (.1820, .9841), 32: (.1555, .9879)}}

def esc(s): return str(s)

# ── HTML pieces ──────────────────────────────────────────────────────────
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
.samp-grid{display:grid;grid-template-columns:76px repeat(3,minmax(0,1fr));gap:12px;align-items:start;margin:18px 0 30px}
.samp-head{font-size:13px;font-weight:650;color:#273442;background:var(--soft);border:1px solid var(--line);border-radius:6px;padding:8px;text-align:center}
.samp-res{font-weight:700;color:#273442;padding-top:16px;font-size:13px}
.samp-grid figure{margin:0}
.samp-grid figcaption{font-size:11.5px;line-height:1.45;margin-top:6px}
@media(max-width:900px){.samp-grid{grid-template-columns:1fr}.samp-head{display:none}.samp-res{grid-column:1/-1;border-top:1px solid var(--line);padding-top:16px}}
th.ge,td.ge{background:#e8f1f6}
th.gr,td.gr{background:#f9efe7}
th.ge,th.gr{color:#273442}
.settings-table td:nth-child(2),.settings-table th:nth-child(2){text-align:left}
figure img{width:100%;height:auto;display:block;border-radius:4px}
.card{border:1px solid var(--line);border-radius:8px;padding:12px;background:white;margin:0 0 22px}
figcaption{color:var(--muted);font-size:13px;line-height:1.55;margin-top:10px}
.small{color:var(--muted);font-size:13px}
.math-block{overflow-x:auto;margin:18px 0;text-align:center;line-height:1.8}
.grid-2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}
@media(max-width:850px){main{padding:24px 16px 44px}.status-grid,.grid-2{grid-template-columns:1fr}table{font-size:12px;display:block;overflow-x:auto}.compact-table{display:table}}"""

HEAD = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>DLR-D-Flow: Noise Study and Algorithm</title>'
        '<script>window.MathJax={tex:{inlineMath:[["\\\\(","\\\\)"]],displayMath:[["\\\\[","\\\\]"]],'
        'processEscapes:true},svg:{fontCache:"global"},options:{skipHtmlTags:["script","noscript","style","textarea","pre","code"]}};</script>'
        '<script defer src="vendor/node_modules/mathjax/es5/tex-svg.js"></script>'
        f'<style>{CSS}</style></head><body><main>')

h = [HEAD]
A = h.append

A('<h1>DLR-D-Flow: Noise Study and Algorithm</h1>')
A('<p class="subtitle">Sparse-view CT on scale-relative single-circle datasets. '
  'Noise robustness at 128&times;128 and 160&times;160, and the dynamic low-rank '
  'sensitivity algorithm with its accuracy and per-iteration cost.</p>')
A('<section class="status-grid">'
  '<div class="status"><strong>Noise study</strong><span>1%, 5%, 10% at 128/160</span></div>'
  '<div class="status"><strong>Time integrator</strong><span>100 Euler steps</span></div>'
  '<div class="status"><strong>Circle radii</strong><span>Scale-relative, fixed</span></div>'
  '<div class="status"><strong>DLR sensitivity</strong><span>Rank 8, cos &ge; 0.98</span></div>'
  '</section>')

# ── 1. setup ─────────────────────────────────────────────────────────────
A('<h2>1. Changes to the Dataset and Flow-Matching Models</h2>')
A('<ul>'
  '<li>Backgrounds are 0 and circles are 1.</li>'
  '<li><strong>Radius sampling.</strong> Radii are drawn as a fixed fraction of the image '
  'side \\(S\\) rather than in absolute pixels: '
  '\\(r\\sim\\mathcal{U}(0.10\\,S,\\ 0.25\\,S)\\), with a boundary gap of '
  '\\(\\tfrac{4}{64}S\\). Circles therefore cover the same fraction of the field of view at '
  'every resolution, so only the discretization changes across the study.</li>'
  '<li><strong>Time integrator.</strong> The flow ODE uses 100 time steps.</li>'
  '</ul>')
A('<table class="compact-table"><tr><th>Resolution</th><th>Radius (px)</th>'
  '<th>Border gap (px)</th></tr>')
for res in [64, 96, 128, 160]:
    A(f'<tr><td>{res}&times;{res}</td><td>{0.10*res:.1f} &ndash; {0.25*res:.1f}</td>'
      f'<td>{round(res*4/64)}</td></tr>')
A('</table>')

# ── 2. noise test
A('<h2>2. Noise Test: RK4 versus Forward Euler</h2>')
A('<p><strong>Goal.</strong> With the number of time steps matched between integrators, we hope '
  'to see that RK4 yields better reconstructions than forward Euler. RK4 uses four velocity '
  'evaluations per step against Euler\'s one, so at matched step count it costs roughly four '
  'times as much; the question is whether the reduced time-discretization error buys a '
  'corresponding gain in reconstruction quality.</p>')

A('<h3>2.1 Experimental Setup</h3>')
A('<p>Sparse-view CT: recover \\(x_\\star\\) from 15 parallel-beam projections with additive '
  'Gaussian measurement noise, by optimizing the source point of the pretrained flow.</p>')
A('<table class="compact-table settings-table"><tr><th>Setting</th><th>Value</th></tr>'
  '<tr><td>Forward operator</td><td>Parallel-beam Radon, 15 angles over \\([0,\\pi)\\)</td></tr>'
  '<tr><td>Resolutions</td><td>128&times;128, 160&times;160</td></tr>'
  '<tr><td>Noise levels \\(\\delta\\)</td><td>1%, 5%, 10% relative \\(L_2\\)</td></tr>'
  '<tr><td>Integrators</td><td>Forward Euler and RK4, both 100 time steps</td></tr>'
  '<tr><td>Velocity evaluations per solve</td><td>100 (Euler), 400 (RK4)</td></tr>'
  '<tr><td>Optimizer</td><td>LBFGS on the source point \\(z\\)</td></tr>'
  '<tr><td>Line search</td><td>None</td></tr>'
  '<tr><td>LBFGS max_iter / history</td><td>5 / 100</td></tr>'
  '<tr><td>Outer iterations</td><td>1200</td></tr>'
  '<tr><td>Selection rule</td><td>Best measurement-objective iterate</td></tr>'
  '<tr><td>Samples per cell</td><td>5 (matched across integrators)</td></tr>'
  '<tr><td>Error metric</td><td>Image relative \\(L_2\\), '
  '\\(\\|\\hat{x}-x_\\star\\|_2/\\|x_\\star\\|_2\\)</td></tr>'
  '<tr><td>Total reconstructions</td><td>60</td></tr></table>')

A('<h3>2.2 Numerical Results</h3>')
A('<p><strong>Table 1. Image relative \\(L_2\\) error, as a percentage, and per-iteration cost.</strong> '
  'Best, median and worst are taken over the five matched samples in each cell; time per '
  'iteration is the mean over those samples. The lower median of each pair is shown in bold.</p>')
A('<table><tr><th rowspan="2">Noise</th><th rowspan="2">Resolution</th>'
  '<th class="ge grp-head" colspan="4">Forward Euler</th>'
  '<th class="gr grp-head" colspan="4">RK4</th></tr>'
  '<tr><th class="ge">Best rel. err. (%)</th><th class="ge">Median rel. err. (%)</th>'
  '<th class="ge">Worst rel. err. (%)</th><th class="ge">s / iter</th>'
  '<th class="gr">Best rel. err. (%)</th><th class="gr">Median rel. err. (%)</th>'
  '<th class="gr">Worst rel. err. (%)</th><th class="gr">s / iter</th></tr>')
for lvl in [1, 5, 10]:
    for res in [128, 160]:
        cells = {}
        for meth in ("forward_euler", "rk4"):
            g = [r for r in noise_rows if r["noise"] == lvl and r["res"] == res and r["method"] == meth]
            if not g:
                cells[meth] = None
                continue
            v = sorted(r["image_rel"] for r in g)
            cells[meth] = (v[0], st.median(v), v[-1], st.mean([r["time"] for r in g]) / 1200)
        e, k = cells["forward_euler"], cells["rk4"]
        if not (e and k):
            continue
        e_win = e[1] <= k[1]
        em = f"<strong>{100*e[1]:.2f}</strong>" if e_win else f"{100*e[1]:.2f}"
        km = f"{100*k[1]:.2f}" if e_win else f"<strong>{100*k[1]:.2f}</strong>"
        A(f'<tr><td>{lvl}%</td><td>{res}&times;{res}</td>'
          f'<td class="ge">{100*e[0]:.2f}</td><td class="ge">{em}</td>'
          f'<td class="ge">{100*e[2]:.2f}</td><td class="ge">{e[3]:.2f}</td>'
          f'<td class="gr">{100*k[0]:.2f}</td><td class="gr">{km}</td>'
          f'<td class="gr">{100*k[2]:.2f}</td><td class="gr">{k[3]:.2f}</td></tr>')
A('</table>')

wins = sum(1 for lvl in [1,5,10] for res in [128,160]
           for a, b in [(sorted(r["image_rel"] for r in noise_rows
                                if r["noise"]==lvl and r["res"]==res and r["method"]=="rk4"),
                         sorted(r["image_rel"] for r in noise_rows
                                if r["noise"]==lvl and r["res"]==res and r["method"]=="forward_euler"))]
           if a and b and st.median(a) < st.median(b))
A(f'<div class="note"><strong>Outcome.</strong> RK4 did not deliver the hoped-for improvement. '
  f'It achieved the lower median error in {wins} of the 6 noise/resolution cells, while costing '
  f'roughly 4&times; as much per iteration (10.9&ndash;12.9 s against 3.1&ndash;3.2 s). '
  f'Median error for both integrators stays in the range 2.7&ndash;6.5% across every cell, '
  f'showing little sensitivity to either the integrator or a tenfold change in noise.</div>')
A('<h3>2.3 Reconstructions</h3>')
A('<p>Best, median and worst of the five samples in each noise/resolution cell, ranked by image '
  'relative \\(L_2\\), with forward Euler on the top row and RK4 below it. Each panel is the '
  'standard diagnostic layout produced by the sweep.</p>')
for lvl in [1, 5, 10]:
    for res in [128, 160]:
        cell = [q for q in noise_picks if q["noise"] == lvl and q["res"] == res]
        if not cell:
            continue
        A(f'<h4>{lvl}% noise, {res}&times;{res}</h4>')
        A('<div class="nz-grid"><div></div>'
          '<div class="samp-head">Best</div><div class="samp-head">Median</div>'
          '<div class="samp-head">Worst</div>')
        for meth, label in [("forward_euler", "Forward Euler"), ("rk4", "RK4")]:
            A(f'<div class="nz-meth">{label}</div>')
            for want in ("best", "median", "worst"):
                q = next((x for x in cell if x["method"] == meth and x["label"] == want), None)
                if q is None:
                    A('<div></div>')
                    continue
                A(f'<figure class="card"><img src="2026-09-01_assets/{q["file"]}" '
                  f'alt="{label} {want} at {lvl}% noise, {res}x{res}">'
                  f'<figcaption><strong>Sample {q["sample"]}</strong><br>'
                  f'image rel. \\(L_2\\) = {100*q["image_rel"]:.2f}%<br>'
                  f'measurement rel. \\(L_2\\) = {100*q["meas_rel"]:.2f}%</figcaption></figure>')
        A('</div>')

# ── 3. algorithm ─────────────────────────────────────────────────────────
A('<h2>3. The DLR-D-Flow Algorithm</h2>')
A('<h3>3.1 Setting and Notation</h3>')

A('<h4>The generative flow</h4>')
A('<p>A pretrained flow-matching model defines an ordinary differential equation that transports '
  'a noise sample into an image:</p>')
A('<div class="math-block">\\[\\dot x_t = v_\\theta(t,x_t),\\qquad x_0=z,'
  '\\qquad t\\in[0,1].\\]</div>')
A('<h4>The inverse problem</h4>')
A('<p>We observe noisy, incomplete measurements of an unknown true image and want to recover it.</p>')
A('<ul>'
  '<li>\\(x_\\star\\) is the unknown ground-truth image.</li>'
  '<li>\\(\\mathcal{A}\\) is the forward operator.</li>'
  '<li>\\(\\eta\\) is measurement noise and \\(\\delta=\\|\\eta\\|_2/'
  '\\|\\mathcal{A}(x_\\star)\\|_2\\) is its relative size.</li>'
  '<li>\\(y_\\delta = \\mathcal{A}(x_\\star)+\\eta\\) is the <em>observed data</em>. '
  'The subscript records that it carries noise at level \\(\\delta\\); it is not an index.</li>'
  '</ul>')
A('<p>D-Flow does not optimize the image directly. It optimizes the source point, so that every '
  'candidate is by construction an image the flow can generate &mdash; this is what enforces the '
  'prior:</p>')
A('<div class="math-block">\\[\\min_{z}\\ \\Phi(z),\\qquad '
  '\\Phi(z)=\\tfrac12\\big\\|\\mathcal{A}(x_1(z))-y_\\delta\\big\\|_2^2 .\\]</div>')
A('<p>\\(\\Phi\\) is the objective: half the squared mismatch between the measurements '
  'predicted from the current sample and those actually observed.</p>')

A('<h4>The gradient and why it is hard</h4>')
A('<p>Minimizing \\(\\Phi\\) needs \\(\\nabla_z\\Phi\\), its gradient with respect to '
  'the source point. The chain rule splits this into two pieces:</p>')
A('<div class="math-block">\\[\\nabla_z\\Phi \\;=\\; J_1^{\\top}g_x,\\qquad '
  'g_x \\;=\\; D\\mathcal{A}(x_1)^{*}\\big(\\mathcal{A}(x_1)-y_\\delta\\big).\\]</div>')
A('<ul>'
  '<li>\\(g_x\\in\\mathbb{R}^{d}\\) is the <em>terminal gradient</em>: the derivative of '
  '\\(\\Phi\\) with respect to the generated image \\(x_1\\), holding the flow fixed. '
  'Here \\(D\\mathcal{A}(x_1)\\) is the derivative of the forward operator and '
  '\\(D\\mathcal{A}(x_1)^{*}\\) its adjoint, which pushes a measurement-space residual back '
  'into image space. \\(g_x\\) is cheap &mdash; it involves no flow.</li>'
  '<li>\\(J_t=\\partial x_t/\\partial z\\in\\mathbb{R}^{d\\times d}\\) is the '
  '<em>flow sensitivity</em>: it records how a perturbation of the source point changes the state '
  'at flow time \\(t\\). \\(J_1\\) is its value at the endpoint, so \\(J_1\\) says how '
  'the generated image responds to a change in \\(z\\).</li>'
  '<li>\\(J_1^{\\top}\\) is its transpose. Applying it to \\(g_x\\) pulls the '
  'image-space gradient back through the whole flow to become a gradient in source space. This is '
  'the expensive object, and the entire method exists to approximate its action.</li>'
  '</ul>')
A('<p>Differentiating the flow ODE with respect to \\(z\\) shows that \\(J_t\\) satisfies '
  'its own matrix-valued ODE, the <em>variational</em> or sensitivity equation:</p>')
A('<div class="math-block">\\[\\dot J_t = A_tJ_t,\\qquad J_0=I,\\qquad '
  'A_t=\\partial_x v_\\theta(t,x_t).\\]</div>')
A('<p>\\(A_t\\in\\mathbb{R}^{d\\times d}\\) is the Jacobian of the velocity field with '
  'respect to its state argument, evaluated along the current trajectory. The initial condition '
  '\\(J_0=I\\) is the identity because at \\(t=0\\) the state <em>is</em> the source point, '
  'so perturbing \\(z\\) perturbs \\(x_0\\) one-for-one.</p>')
A('<div class="note warn"><strong>The obstacle.</strong> At 160&times;160, \\(d=25{,}600\\), so '
  '\\(J_t\\) and \\(A_t\\) are each \\(25{,}600\\times25{,}600\\) &mdash; and '
  '\\(A_t\\) is the Jacobian of a neural network, never available as an array. Neither matrix '
  'can be formed or stored. What <em>is</em> available cheaply is the action of \\(A_t\\) on a '
  'few vectors: \\(A_tQ\\) by a forward-mode derivative (JVP) and \\(A_t^{\\top}Q\\) by a '
  'reverse-mode one (VJP), for a skinny \\(Q\\) with \\(r\\ll d\\) columns. Every step of '
  'the algorithm is built from those two products.</div>')

A('<h3>3.2 Deriving the \\(U_t\\), \\(S_t\\), \\(V_t\\) Equations</h3>')
A('<p>Model the sensitivity as a scalar baseline plus a rank-\\(r\\) residual,</p>')
A('<div class="math-block">\\[J_t \\approx \\alpha_t I + Y_t,\\qquad Y_t = U_tS_tV_t^{\\top},'
  '\\qquad U_t,V_t\\in\\mathbb{R}^{d\\times r},\\ S_t\\in\\mathbb{R}^{r\\times r},\\]</div>')
A('<p>with orthonormal factors \\(U_t^{\\top}U_t=V_t^{\\top}V_t=I_r\\). Here '
  '\\(\\alpha_t\\in\\mathbb{R}\\) is a single scalar &mdash; the <em>baseline</em> &mdash; '
  'and \\(Y_t\\) is the rank-\\(r\\) <em>residual</em> that corrects it.</p>')

A('<h4>What \\(\\alpha_t\\) is and why it is there</h4>')
A('<p>A pure rank-\\(r\\) model cannot represent the initial condition. The sensitivity starts '
  'at \\(J_0=I\\), which has full rank \\(d\\), so no \\(USV^{\\top}\\) with '
  '\\(r\\ll d\\) can approximate it. Splitting off a multiple of the identity fixes this: at '
  '\\(t=0\\) we take \\(\\alpha_0=1\\) and \\(Y_0=0\\), so the model reproduces '
  '\\(J_0=I\\) exactly, and the low-rank part only has to describe how the sensitivity '
  '<em>deviates</em> from a scalar multiple of the identity as the flow proceeds.</p>')
A('<p>The natural choice for that scalar is the one that best matches \\(J_t\\) in the '
  'Frobenius norm. Minimizing \\(\\|J_t-\\alpha I\\|_F\\) over \\(\\alpha\\) gives '
  'the mean diagonal entry,</p>')
A('<div class="math-block">\\[\\alpha_t \\;=\\; \\frac{\\operatorname{tr}(J_t)}{d},\\]</div>')
A('<p>so \\(\\alpha_t\\) measures the average factor by which the flow stretches or contracts '
  'a perturbation of the source point, ignoring any directional structure.</p>')
A('<p>The trace of \\(J_t\\) cannot be read off directly, so it is estimated stochastically. '
  'We draw \\(k\\) fixed random probe vectors \\(p_1,\\dots,p_k\\) with entries '
  '\\(\\pm1\\), carry them along the flow by the same sensitivity equation, '
  '\\(q_i(t)=J_tp_i\\), and use the Hutchinson estimator</p>')
A('<div class="math-block">\\[\\alpha_t \\;\\approx\\; '
  '\\frac{1}{k\\,d}\\sum_{i=1}^{k} p_i^{\\top}q_i(t),\\qquad '
  '\\dot\\alpha_t \\;\\approx\\; \\frac{1}{k\\,d}\\sum_{i=1}^{k} '
  'p_i^{\\top}A_tq_i(t),\\]</div>')
A('<p>which is unbiased because \\(\\mathbb{E}\\,p^{\\top}Mp=\\operatorname{tr}(M)\\) '
  'for such probes. The implementation uses \\(k=32\\). Propagating the probes costs one extra '
  'product \\(A_tQ\\) per flow step; \\(\\dot\\alpha_t\\) comes free from the same '
  'product, and both feed the factor updates below.</p>')
A('<div class="note warn"><strong>In practice the baseline contributes very little.</strong> '
  'The trained flows here are strongly contractive, so \\(\\alpha_t\\) decays from 1 to '
  'roughly 0.008 by \\(t=1\\): almost all of the sensitivity is carried by the low-rank '
  'residual, not by the scalar. Setting the terminal baseline to zero changes the relative '
  'gradient error only from 0.1287 to 0.1288. The baseline is therefore doing real work near '
  '\\(t=0\\), where \\(J_t\\) is still close to the identity, and essentially none by the '
  'time the gradient is read off. Note also that integrating the log-volume relation '
  '\\(\\tfrac{\\mathrm{d}}{\\mathrm{d}t}\\log\\alpha_t=\\operatorname{tr}(A_t)/d\\) '
  'would give \\(\\det(J_t)^{1/d}\\), the <em>geometric</em> mean of the singular values, '
  'which is not the Frobenius-optimal scalar and is substantially smaller; the implementation '
  'uses \\(\\operatorname{tr}(J_t)/d\\) above.</div>')

A('<p>Substituting the model into the sensitivity equation gives the residual dynamics</p>')
A('<div class="math-block">\\[\\dot Y_t = F(t,Y_t) := A_t\\big(\\alpha_tI+Y_t\\big)-\\dot\\alpha_t I.\\]</div>')
A('<p>The right-hand side \\(F\\) generally leaves the rank-\\(r\\) manifold '
  '\\(\\mathcal{M}_r\\). The Koch&ndash;Lubich construction imposes a Galerkin condition: '
  'choose the velocity <em>within</em> the tangent space that is closest to the true one,</p>')
A('<div class="math-block">\\[\\dot Y_t \\in T_{Y_t}\\mathcal{M}_r \\quad\\text{such that}\\quad '
  '\\big\\|\\dot Y_t - F(t,Y_t)\\big\\|_F \\ \\text{is minimal}'
  '\\qquad\\Longleftrightarrow\\qquad \\dot Y_t = P_{T_{Y_t}\\mathcal{M}_r}F(t,Y_t).\\]</div>')
A('<p>Every tangent vector at \\(Y=USV^{\\top}\\) has the form</p>')
A('<div class="math-block">\\[\\delta Y = \\delta U\\,S\\,V^{\\top} + U\\,\\delta S\\,V^{\\top} '
  '+ U\\,S\\,\\delta V^{\\top},\\]</div>')
A('<p>and this representation is not unique &mdash; the decomposition admits rotations that leave '
  '\\(Y\\) unchanged. Uniqueness is fixed by the <em>gauge conditions</em></p>')
A('<div class="math-block">\\[U^{\\top}\\dot U = 0,\\qquad V^{\\top}\\dot V = 0,\\]</div>')
A('<p>which also preserve orthonormality, since \\(\\frac{d}{dt}(U^{\\top}U)=\\dot U^{\\top}U+U^{\\top}\\dot U=0\\). '
  'With these conditions the orthogonal projection onto the tangent space is</p>')
A('<div class="math-block">\\[P_Y(Z) = ZVV^{\\top} - UU^{\\top}ZVV^{\\top} + UU^{\\top}Z.\\]</div>')
A('<p>Now write \\(\\dot Y = \\dot U S V^{\\top} + U\\dot S V^{\\top} + U S\\dot V^{\\top}\\) '
  'and extract each factor equation by projecting.</p>')
A('<p><strong>Core.</strong> Multiply on the left by \\(U^{\\top}\\) and on the right by \\(V\\). '
  'The gauge conditions kill the outer terms, \\(U^{\\top}\\dot U=0\\) and \\(\\dot V^{\\top}V=0\\), leaving</p>')
A('<div class="math-block">\\[U^{\\top}\\dot Y V = \\dot S \\qquad\\Longrightarrow\\qquad '
  '\\boxed{\\ \\dot S = U^{\\top} Z V\\ }\\qquad (Z:=F(t,Y)).\\]</div>')
A('<p><strong>Left factor.</strong> Multiply on the right by \\(V\\) only: '
  '\\(\\dot Y V = \\dot U S + U\\dot S\\). Applying \\(I-UU^{\\top}\\) removes the second term, '
  'and \\((I-UU^{\\top})\\dot U = \\dot U\\) by the gauge condition, so</p>')
A('<div class="math-block">\\[\\dot U S = (I-UU^{\\top})ZV \\qquad\\Longrightarrow\\qquad '
  '\\boxed{\\ \\dot U = (I-UU^{\\top})\\,Z\\,V\\,S^{-1}\\ }.\\]</div>')
A('<p><strong>Right factor.</strong> Symmetrically, multiplying on the left by \\(U^{\\top}\\) '
  'and applying \\(I-VV^{\\top}\\) to the transpose gives</p>')
A('<div class="math-block">\\[\\boxed{\\ \\dot V = (I-VV^{\\top})\\,Z^{\\top}\\,U\\,S^{-\\top}\\ }.\\]</div>')
A('<div class="note warn"><strong>Why these equations are not integrated directly.</strong> '
  'Both factor equations contain \\(S^{-1}\\). The core \\(S_t\\) becomes ill-conditioned exactly '
  'when the approximation is nearly of lower rank &mdash; which is the normal situation here, and '
  'is the situation at \\(t=0\\) where the residual starts at \\(Y_0=0\\). A direct integrator '
  'would need a step size tied to the smallest singular value of \\(S\\).</div>')

A('<h3>3.3 Projector Splitting</h3>')
A('<p>The Lubich&ndash;Oseledets integrator avoids \\(S^{-1}\\) entirely by splitting the '
  'projection into its three terms and integrating each in turn. Over one flow step of length '
  '\\(h\\) with \\(A_t\\) and \\(\\alpha_t\\) frozen:</p>')
A('<div class="math-block">\\[P_Y(Z) = \\underbrace{ZVV^{\\top}}_{\\text{K-step}} '
  '\\underbrace{-\\,UU^{\\top}ZVV^{\\top}}_{\\text{S-step}} + \\underbrace{UU^{\\top}Z}_{\\text{L-step}}\\]</div>')
A('<ul>'
  '<li><strong>K-step.</strong> Freeze \\(V=V_0\\), set \\(K=US\\) and integrate '
  '\\(\\dot K = F(t,KV_0^{\\top})V_0\\). A thin QR factorization \\(K(t_1)=U_1\\hat S\\) '
  'produces the new column space.</li>'
  '<li><strong>S-step.</strong> Freeze both bases and integrate '
  '\\(\\dot S = -U_1^{\\top}F(t,U_1SV_0^{\\top})V_0\\). The minus sign makes this substep run '
  '<em>backwards</em>; it removes the double counting between the other two.</li>'
  '<li><strong>L-step.</strong> Freeze \\(U=U_1\\), set \\(L=VS^{\\top}\\) and integrate '
  '\\(\\dot L = F(t,U_1L^{\\top})^{\\top}U_1\\), then \\(L(t_1)=V_1S_1^{\\top}\\) by QR.</li>'
  '</ul>')
A('<p>For the scalar-baseline model \\(F(t,Y)=A_t(\\alpha_tI+Y)-\\dot\\alpha_tI\\) these become</p>')
A('<div class="math-block">\\[\\dot K = \\alpha_tA_tV_0 + A_tK - \\dot\\alpha_tV_0,\\qquad '
  '\\dot S = -\\alpha_tU_1^{\\top}A_tV_0 - (U_1^{\\top}A_tU_1)S + \\dot\\alpha_tU_1^{\\top}V_0,\\]</div>')
A('<div class="math-block">\\[\\dot L = \\alpha_tA_t^{\\top}U_1 + L\\,(U_1^{\\top}A_t^{\\top}U_1) '
  '- \\dot\\alpha_tU_1.\\]</div>')
A('<p>Every occurrence of \\(A_t\\) is applied to a matrix with only \\(r\\) columns. Those '
  'products are computed as batched JVPs and VJPs through the velocity network, so the Jacobian '
  'is never assembled. Per flow step the cost is \\(O(r)\\) network evaluations, \\(O(dr^2)\\) '
  'for the thin QRs, and \\(O(r^3)\\) for the core algebra; storage is \\(O(dr)\\) rather than \\(O(d^2)\\).</p>')
A('<div class="note interpretation"><strong>Initialization.</strong> The residual begins at '
  '\\(Y_0=0\\), which has no singular vectors, so the factorization is undefined at \\(t=0\\). '
  'Seeding \\(S_0=\\varepsilon I\\) on a random basis fails badly: with \\(S\\approx0\\) the '
  'tangent projection is degenerate and the basis never rotates toward the directions that carry '
  'the gradient. The implementation instead defers initialization by one Euler step and forms a '
  'randomized sketch of the first genuinely nonzero residual '
  '\\(Y_h\\approx hA_0+(1-\\alpha_h)I\\), then augments the bases with fresh forcing directions '
  'at each step before recompressing to rank \\(r\\). This single change reduced relative '
  'gradient error at 64&times;64 from 0.85 to 0.13.</div>')

A('<h3>3.4 Algorithm</h3>')
A('<div class="algo">')
A('<div class="algo-title">Algorithm 1 &mdash; DLR-D-Flow</div>')
A('<div class="algo-io"><strong>Input:</strong> observation \\(y\\); operator '
  '\\(\\mathcal{A}\\); velocity field \\(v_\\theta\\); initial source \\(z_0\\); '
  'rank \\(r\\); flow steps \\(N\\) with \\(h=1/N\\); outer steps \\(K\\); '
  'step size \\(\\eta\\)<br>'
  '<strong>Output:</strong> the iterate with the smallest measurement misfit</div>')
A('<ol>')
A('<li><span class="step">outer</span> <strong>for</strong> \\(k=1,\\dots,K\\):'
  '<ol>')

A('<li><span class="step">init</span> Set \\(x\\leftarrow z\\), \\(\\alpha\\leftarrow1\\). '
  'Take one Euler step and sketch the first nonzero residual, since \\(Y_0=0\\) has no '
  'singular vectors:'
  '<div class="math-block">\\[(U,S,V)\\leftarrow\\operatorname{rand-svd}_r'
  '\\big(Z\\mapsto hA_0Z+(1-\\alpha_h)Z\\big),\\qquad '
  'x\\leftarrow x+h\\,v_\\theta(0,x).\\]</div></li>')

A('<li><span class="step">flow</span> <strong>for</strong> \\(n=1,\\dots,N-1\\) with '
  '\\(t=nh\\), using only \\(A_tQ\\) (JVP) and \\(A_t^{\\top}Q\\) (VJP):'
  '<ol>')
A('<li><em>K-step.</em> '
  '\\(K_1=US+h\\big(\\alpha A_tV+A_t US-\\dot\\alpha V\\big)\\), '
  'then \\(U_1\\hat S=\\operatorname{qr}(K_1)\\). '
  '<span class="cmt">new column space</span></li>')
A('<li><em>S-step.</em> With \\(M=U_1^{\\top}A_tU_1\\), '
  '\\(\\tilde S=\\hat S+h\\big(-\\alpha U_1^{\\top}A_tV-M\\hat S'
  '+\\dot\\alpha U_1^{\\top}V\\big)\\). '
  '<span class="cmt">runs backwards; removes double counting</span></li>')
A('<li><em>L-step.</em> '
  '\\(L_1=V\\tilde S^{\\top}+h\\big(\\alpha A_t^{\\top}U_1'
  '+V\\tilde S^{\\top}M^{\\top}-\\dot\\alpha U_1\\big)\\), '
  'then \\(V_1S_1^{\\top}=\\operatorname{qr}(L_1)\\). '
  '<span class="cmt">new row space</span></li>')
A('<li><em>Enrich and recompress.</em> Augment the bases with fresh forcing directions and '
  'truncate back to rank \\(r\\):'
  '\\(\\ (U,S,V)\\leftarrow\\operatorname{trunc}_r'
  '\\big([U_1\\,|\\,A_tU_1],\\,S_1,\\,[V_1\\,|\\,A_t^{\\top}V_1]\\big)\\). '
  '<span class="cmt">keeps the basis from going stale</span></li>')
A('<li><em>Advance.</em> \\(x\\leftarrow x+h\\,v_\\theta(t,x)\\), and update '
  '\\(\\alpha\\) from the propagated probes.</li>')
A('</ol></li>')

A('<li><span class="step">grad</span> Terminal gradient and its low-rank pullback:'
  '<div class="math-block">\\[g_x=D\\mathcal{A}(x)^{*}\\big(\\mathcal{A}(x)-y\\big),'
  '\\qquad \\widehat{\\nabla_z\\Phi}=\\alpha g_x+VS^{\\top}U^{\\top}g_x.\\]</div>'
  '<span class="cmt">three small mat-vecs; \\(J_1\\) is never formed</span></li>')

A('<li><span class="step">step</span> '
  '\\(z\\leftarrow\\operatorname{adam}\\big(z,\\widehat{\\nabla_z\\Phi},\\eta\\big)\\), '
  'then regenerate \\(x_k=\\operatorname{flow}(z)\\) and retain it if '
  '\\(\\|\\mathcal{A}(x_k)-y\\|^2\\) is the smallest seen.</li>')
A('</ol></li>')
A('</ol>')
A('<p class="small" style="margin:4px 0 0">Per flow step: \\(O(r)\\) network evaluations, '
  'two thin QR factorizations of \\(d\\times r\\) matrices, and \\(r\\times r\\) core '
  'algebra. Storage is \\(O(dr)\\).</p>')
A('</div>')

# ── 3.5 accuracy ─────────────────────────────────────────────────────────
A('<h3>3.5 Accuracy</h3>')
A('<p>Noiseless sparse-view CT with 15 angles, rank 8, five test images per resolution, '
  'measured as image relative \\(L_2\\) error '
  '\\(\\|\\hat{x}-x_\\star\\|_2/\\|x_\\star\\|_2\\).</p>')
A('<h4>Best, median and worst reconstruction per resolution</h4>')
A('<p>Each panel is the standard diagnostic layout: ground truth, DLR-D-Flow reconstruction and '
  'image error on the top row; observed \\(Ax\\), reconstructed \\(Ax\\) and measurement '
  'residual on the bottom. Columns are the best, median and worst of the five samples at that '
  'resolution, ranked by image relative \\(L_2\\).</p>')
A('<div class="samp-grid"><div></div>'
  '<div class="samp-head">Best</div><div class="samp-head">Median</div>'
  '<div class="samp-head">Worst</div>')
for res in [64, 96, 128, 160]:
    g = sorted((r for r in bench_rel if r["resolution"] == res), key=lambda r: r["image_rel_l2"])
    picks = [("best", g[0]), ("median", g[len(g) // 2]), ("worst", g[-1])]
    A(f'<div class="samp-res">{res}&times;{res}</div>')
    for label, row in picks:
        A(f'<figure class="card"><img src="2026-09-01_assets/recon_res{res}_{label}.png" '
          f'alt="{label} DLR-D-Flow reconstruction at {res}x{res}">'
          f'<figcaption><strong>Sample {row["test_index"]}</strong><br>'
          f'solution rel. \\(L_2\\) = {100*row["image_rel_l2"]:.2f}%<br>'
          f'measurement rel. \\(L_2\\) = {100*row["meas_rel_l2"]:.2f}%</figcaption></figure>')
A('</div>')
A('<div class="note interpretation"><strong>Interpretation.</strong> The worst column shows what '
  'failure looks like: a clean, correctly sized circle placed at the wrong position, with the '
  'characteristic crescent pair in the image-error panel and a coherent dipole rather than '
  'scatter in the measurement residual. The flow prior is working in every case &mdash; each '
  'iterate is a valid circle &mdash; so what fails is localization, not regularization.</div>')

A('<p><strong>Table 2. DLR-D-Flow reconstruction error by resolution, as a percentage.</strong> '
  'Best, median and worst are taken over the five samples at each resolution; the last column is '
  'the median achieved measurement relative error.</p>')
A('<table class="compact-table"><tr><th>Resolution</th><th>\\(d\\)</th><th>n</th>'
  '<th>Best rel. err. (%)</th><th>Median rel. err. (%)</th><th>Worst rel. err. (%)</th>'
  '<th>Median meas. rel. err. (%)</th></tr>')
for res in [64, 96, 128, 160]:
    g = [r for r in bench_rel if r["resolution"] == res]
    v = sorted(r["image_rel_l2"] for r in g)
    meas_med = st.median([r["meas_rel_l2"] for r in g])
    A(f'<tr><td>{res}&times;{res}</td><td>{res*res:,}</td><td>{len(g)}</td>'
      f'<td>{100*v[0]:.2f}</td><td>{100*st.median(v):.2f}</td><td>{100*v[-1]:.2f}</td>'
      f'<td>{100*meas_med:.2f}</td></tr>')
A('</table>')

allp = sorted(r["image_rel_l2"] for r in bench_rel)
nsucc = sum(1 for v in allp if v < 0.25)
A(f'<div class="note"><strong>Outcome.</strong> Median relative error is between 5.7% and 10.8% '
  f'at every resolution and does not degrade systematically with grid size. Outcomes are bimodal '
  f'rather than graded: {nsucc} of {len(allp)} runs reached 1.4&ndash;3.9% and the remaining '
  f'{len(allp)-nsucc} landed at 64&ndash;145%, with nothing in between. An error above 100% means '
  f'the reconstruction is further from the truth than a blank image, which is what happens when a '
  f'correctly sized circle is placed at the wrong position: the error counts both the missing '
  f'circle and the spurious one.</div>')
A('</main></body></html>')

OUT.write_text("\n".join(h))
print("wrote", OUT, f"({OUT.stat().st_size/1024:.0f} KB)")
