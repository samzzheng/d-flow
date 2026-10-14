"""Build the dated HTML report from audited final single-circle reconstructions."""
from __future__ import annotations

import argparse
import base64
import csv
from dataclasses import dataclass
from datetime import date
import hashlib
from html import escape
import json
from pathlib import Path
import re
import shutil
import statistics


METHODS = ("D-Flow", "P-Flow", "DLR D-Flow")
NOISES = (1, 5, 10, 20)
RESOLUTIONS = (64, 96, 128, 160, 200, 250)
CaseKey = tuple[int, str, int, int]


@dataclass(frozen=True)
class Case:
    noise: int
    problem: str
    resolution: int
    sample: int
    errors: tuple[float, ...]
    statuses: tuple[str, ...]
    attempts: tuple[int, ...]
    source: Path

    @property
    def gap(self) -> float:
        return self.errors[0] - self.errors[2]

    @property
    def name(self) -> str:
        return f"noise{self.noise:02d}_{self.problem}_r{self.resolution}_sample{self.sample:03d}"


def panel(case: Case, label: str, assets: Path) -> str:
    target = assets / f"{case.name}.png"
    if not target.exists():
        shutil.copyfile(case.source / "comparison.png", target)
    title = f"{label} · {case.problem.upper()} · {case.resolution}² · Sample {case.sample}"
    metrics = " · ".join(f"{name}: <strong>{value:.2f}%</strong>"
                         for name, value in zip(METHODS, case.errors))
    stopping = "; ".join(f"{name}: {status.replace('_', ' ')} at attempt {attempt}"
                         for name, status, attempt in zip(METHODS, case.statuses, case.attempts))
    return (f'<figure class="card"><figcaption><strong>{escape(title)}</strong><br>{metrics}'
            f'<br><span class="small">{escape(stopping)}</span></figcaption>'
            f'<img src="{assets.name}/{target.name}" loading="lazy" width="2240" height="1120" '
            f'alt="{escape(title)}; paired GT, D-Flow, P-Flow and DLR images and measurements"></figure>')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("outputs/five_step_single_circle_results_20261014"))
    parser.add_argument("--out", type=Path, default=Path("outputs/reports"))
    parser.add_argument("--date", type=date.fromisoformat, default=date.today())
    args = parser.parse_args()
    summary = json.loads((args.data / "summary.json").read_text())
    roots = {int(study["noise_percent"]): Path(study["root"]) for study in summary["studies"]}
    grouped: dict[CaseKey, dict[str, dict[str, str]]] = {}
    with (args.data / "runs.csv").open() as handle:
        for row in csv.DictReader(handle):
            key = (int(float(row["noise_percent"])), row["problem"], int(row["resolution"]), int(row["sample"]))
            assert row["method"] not in grouped.setdefault(key, {})
            grouped[key][row["method"]] = row
    cases = []
    for (noise, problem, resolution, sample), records in sorted(grouped.items()):
        assert set(records) == set(METHODS)
        acquisition = "ct10" if problem == "ct" else "mri4"
        cases.append(Case(noise, problem, resolution, sample,
            tuple(float(records[method]["image_percent"]) for method in METHODS),
            tuple(records[method]["status"] for method in METHODS),
            tuple(int(records[method]["attempts"]) for method in METHODS),
            roots[noise] / "production" / f"r{resolution}_n1_{acquisition}" / f"sample_{sample:03d}"))
    assert len(cases) == 1440
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"single_circle_noise_test_{args.date.isoformat()}"
    assets = args.out / f"{stem}_assets"
    assets.mkdir(exist_ok=True)
    previous = (args.out / "dlr_dflow_report_2026-09-16.html").read_text()
    style = re.search(r"<style>(.*?)</style>", previous, flags=re.S).group(1)
    style += """
nav{display:flex;gap:18px;flex-wrap:wrap;margin:20px 0}a{color:var(--accent)}
details{border:1px solid var(--line);border-radius:8px;margin:12px 0;padding:0 16px}
summary{cursor:pointer;padding:15px 0;font-weight:650;line-height:1.5}
details[open]>summary{border-bottom:1px solid var(--line);margin-bottom:14px}
.card figcaption{margin:0 0 12px;color:var(--ink)}.card img{margin-top:12px}
.empty{padding:12px 16px;background:var(--soft);color:var(--muted)}
.tables{display:grid;grid-template-columns:1fr 1fr;gap:28px}.tables h3{margin-top:0}
button{font:inherit;background:var(--soft);color:var(--accent);border:1px solid var(--line);border-radius:5px;padding:8px 12px;cursor:pointer}
@media(max-width:850px){.tables{grid-template-columns:1fr}.card{padding:6px}.card figcaption{font-size:12px}}
@media print{nav,button{display:none}main{max-width:none;padding:0}details{break-inside:avoid}figure{break-inside:avoid}}
"""
    title = f"Single Circle Noise Test ({args.date.strftime('%B %d, %Y')})"
    html = [f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{title}</title><style>{style}</style></head><body><main>',
            f'<h1>{title}</h1>',
            '<p class="subtitle">Completed single-circle study · 1,440 paired cases · 4,320 reconstructions</p>',
            '<div class="note interpretation"><strong>DLR retains low typical reconstruction error as measurement noise increases.</strong> '
            'From 1% to 20% noise, its median image error rises from 1.91% to 5.94% for CT and from 2.04% to 2.40% for MRI. '
            'The median advantage over D-Flow and P-Flow is clearest at high noise. '
            'D-Flow remains best for MRI at 1% noise; this is not a uniform advantage on every sample.</div>',
            '<nav><a href="#setup">Setup</a><a href="#results">Median results</a>'
            '<a href="#representatives">Best / median / worst</a><a href="#gaps">Largest DLR advantages</a></nav>',
            '<h2 id="setup">1. Inverse problems and protocol</h2>',
            '<p><strong>CT:</strong> sparse parallel-beam projections at 10 uniformly spaced angles over [0°, 180°), '
            'using a rotate-and-average Radon operator. <strong>MRI:</strong> single-coil, zero-phase circle phantoms '
            'with a centered orthonormal Fourier transform and variable-density Cartesian sampling at nominal 4× acceleration '
            '(approximately one quarter of phase-encoding lines, including DC).</p>',
            '<p><strong>Measurements:</strong> y = A(x<sub>GT</sub>) + ε, with ‖ε‖₂ / ‖A(x<sub>GT</sub>)‖₂ '
            'equal to 1%, 5%, 10%, or 20%. Gaussian noise is normalized to the specified norm; MRI noise is complex '
            'and added only at acquired coefficients. Truths, masks, initial sources and noise seeds are paired across methods and noise levels.</p>',
            '<p><strong>Shared settings:</strong> resolutions 64², 96², 128², 160², 200², 250²; 30 samples per acquisition/resolution/noise level; '
            'epoch-200 non-attention flow-matching UNets; <strong>5 Euler steps</strong>; FP32; '
            'LBFGS (learning rate 1, history 100) with Armijo geometric halving (c₁ = 10⁻⁴, at most 30 trials), '
            'first-attempt initial scaling, and source displacement cap 0.01√d. '
            'Up to 1,000 optimizer attempts, stopping after two failed searches. '
            'DLR uses the fast rank-8 configuration, α(t) = 1 − t, zero enrichment and previous-V sketch reuse. '
            'All methods minimize ½‖A(G(z)) − y‖₂² without source projection, image clipping or extra regularization.</p>',
            '<h2 id="results">2. Median relative image error</h2>',
            '<p>100 × ‖x̂ − x<sub>GT</sub>‖₂ / ‖x<sub>GT</sub>‖₂, from the <strong>final saved iterate</strong>. '
            'Each entry pools 180 samples (30 at each resolution); lower is better. All outcomes are included. '
            'Bold indicates the lowest median within a row.</p><div class="tables">']
    for problem in ("ct", "mri"):
        html.append(f'<section><h3>{"CT · 10 angles" if problem == "ct" else "MRI · 4× acceleration"}</h3>'
                    '<table><thead><tr><th>Noise</th><th>D-Flow</th><th>P-Flow</th><th>DLR D-Flow</th></tr></thead><tbody>')
        for noise in NOISES:
            subset = [case for case in cases if case.problem == problem and case.noise == noise]
            medians = [statistics.median(case.errors[index] for case in subset) for index in range(3)]
            cells = [f'<td>{"<strong>" if value == min(medians) else ""}{value:.2f}%'
                     f'{"</strong>" if value == min(medians) else ""}</td>' for value in medians]
            html.append(f'<tr><td>{noise}%</td>{"".join(cells)}</tr>')
        html.append('</tbody></table></section>')
    html += ['</div><p class="small">Medians describe typical performance, not reliability on every sample. '
             'Large-error outcomes remain included and are shown below; their cause has not been established as optimizer failure. '
             'A stalled line search is not proof of convergence. The methods also stop at different attempts, '
             'so these results do not isolate low-rank regularization from optimization and stopping effects.</p>',
             '<h2 id="representatives">3. Best, median and worst by DLR image error</h2>',
             '<p>For each noise level and inverse problem, rank all 180 final DLR reconstructions across resolutions. '
             'Show the best, the lower of the two central samples (rank 90/180), and the worst, with all methods on that same case. '
             'The median representative need not equal the table median. Sample selection is retrospective; checkpoint selection is always final.</p>',
             '<p class="small">Panels reuse the shared reconstruction figures: GT, D-Flow, P-Flow, DLR D-Flow. '
             'Images use a fixed [0,1] display scale; CT measurements use [0,1] and MRI log(1 + magnitude) uses [0,5]. '
             'Display saturation does not affect errors. Measurement errors compare with clean A(x<sub>GT</sub>), not noisy y.</p>']
    selections = []
    for noise in NOISES:
        html.append(f'<details class="gallery"><summary>{noise}% noise · best / median / worst</summary>')
        for problem in ("ct", "mri"):
            ordered = sorted((case for case in cases if case.noise == noise and case.problem == problem),
                             key=lambda case: (case.errors[2], case.resolution, case.sample))
            html.append(f'<details class="gallery"><summary>{problem.upper()} · 180 cases</summary>')
            for label, position in (("Best", 0), ("Median representative", 89), ("Worst", 179)):
                case = ordered[position]
                html.append(panel(case, label, assets))
                selections.append({"section": "representatives", "label": label, "case": case.name,
                                   "rank": position + 1, "source": str(case.source)})
            html.append('</details>')
        html.append('</details>')
    html += ['<h2 id="gaps">4. Cases with the largest DLR advantage over D-Flow</h2>',
             '<p>For each noise level, resolution and inverse problem, show the three largest positive gaps '
             '<strong>Δ = D-Flow error − DLR error</strong> (percentage points), among cases with DLR image error ≤10%. '
             'This explicit gallery cutoff identifies relatively accurate DLR reconstructions; it does not filter the summary tables. '
             'Positive gaps can be small at low noise. Missing qualifying examples are marked rather than replaced with DLR losses. '
             'These selected examples illustrate differences, not their frequency.</p>',
             '<button type="button" onclick="document.querySelectorAll(\'details.gallery\').forEach(d=>d.open=true)">Expand all examples</button> '
             '<button type="button" onclick="document.querySelectorAll(\'details.gallery\').forEach(d=>d.open=false)">Collapse all examples</button>']
    for noise in NOISES:
        html.append(f'<details class="gallery"><summary>{noise}% noise · DLR advantage by resolution</summary>')
        for resolution in RESOLUTIONS:
            html.append(f'<details class="gallery"><summary>Resolution: {resolution}²</summary>')
            for problem in ("ct", "mri"):
                subset = [case for case in cases if (case.noise, case.resolution, case.problem) == (noise, resolution, problem)]
                assert len(subset) == 30
                ordered = sorted((case for case in subset if case.gap > 0 and case.errors[2] <= 10),
                                 key=lambda case: (-case.gap, case.sample))
                html.append(f'<details class="gallery"><summary>{problem.upper()} · '
                            f'{len(ordered)}/30 qualifying cases · showing {min(3, len(ordered))}</summary>')
                for index in range(3):
                    if index < len(ordered):
                        case = ordered[index]
                        label = f"Example {index + 1} · Δ = {case.gap:.2f} percentage points"
                        html.append(panel(case, label, assets))
                        selections.append({"section": "advantages", "label": label, "case": case.name,
                                           "gap_percentage_points": case.gap, "source": str(case.source)})
                    else:
                        html.append(f'<p class="empty">Example {index + 1}: unavailable — '
                                    'no additional case has DLR error ≤10% and lower error than D-Flow.</p>')
                html.append('</details>')
            html.append('</details>')
        html.append('</details>')
    html += ['<p class="small">Provenance: completed single-circle audit, 14 October 2026. '
             'The accompanying selection manifest records the source directory for every example; '
             'all 4,320 final metric records are retained in the report assets.</p>', '</main></body></html>']
    document = '\n'.join(html)
    (args.out / f"{stem}.html").write_text(document)
    standalone = document
    for path in sorted(assets.glob('*.png')):
        encoded = base64.b64encode(path.read_bytes()).decode('ascii')
        standalone = standalone.replace(f'src="{assets.name}/{path.name}"', f'src="data:image/png;base64,{encoded}"')
    (args.out / f"{stem}_standalone.html").write_text(standalone)
    shutil.copyfile(args.data / 'runs.csv', assets / 'runs.csv')
    shutil.copyfile(args.data / 'summary.json', assets / 'summary.json')
    metadata = {"title": title, "data": str(args.data), "run_count": len(cases) * 3,
                "runs_sha256": hashlib.sha256((args.data / 'runs.csv').read_bytes()).hexdigest(),
                "gallery_rule": "DLR <=10%, positive D-Flow minus DLR gap; top 3 per noise/resolution/problem",
                "selections": selections}
    (assets / 'selection_manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({"report": str(args.out / f'{stem}.html'), "selections": len(selections),
                      "unique_images": len({item['case'] for item in selections})}))


if __name__ == "__main__":
    main()
