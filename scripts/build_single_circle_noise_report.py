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
    title = rf"{label} · {case.problem.upper()} · \({case.resolution}^2\) · Sample {case.sample}"
    metrics = " · ".join(f"{name}: <strong>{value:.2f}%</strong>"
                         for name, value in zip(METHODS, case.errors))
    return (f'<figure class="card"><figcaption><strong>{escape(title)}</strong><br>{metrics}'
            '</figcaption>'
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
            roots[noise] / "production" / f"r{resolution}_n1_{acquisition}" / f"sample_{sample:03d}"))
    assert len(cases) == 1440
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"single_circle_noise_test_{args.date.isoformat()}"
    assets = args.out / f"{stem}_assets"
    assets.mkdir(exist_ok=True)
    previous = (args.out / "dlr_dflow_report_2026-09-16.html").read_text()
    style = re.search(r"<style>(.*?)</style>", previous, flags=re.S).group(1)
    style += """
a{color:var(--accent)}
details{border:1px solid var(--line);border-radius:8px;margin:12px 0;padding:0 16px}
summary{cursor:pointer;padding:15px 0;font-weight:650;line-height:1.5}
details[open]>summary{border-bottom:1px solid var(--line);margin-bottom:14px}
.card figcaption{margin:0 0 12px;color:var(--ink)}.card img{margin-top:12px}
.empty{padding:12px 16px;background:var(--soft);color:var(--muted)}
.tables{display:grid;grid-template-columns:1fr 1fr;gap:28px}.tables h3{margin-top:0}
button{font:inherit;background:var(--soft);color:var(--accent);border:1px solid var(--line);border-radius:5px;padding:8px 12px;cursor:pointer}
@media(max-width:850px){.tables{grid-template-columns:1fr}.card{padding:6px}.card figcaption{font-size:12px}}
@media print{button{display:none}main{max-width:none;padding:0}details{break-inside:avoid}figure{break-inside:avoid}}
"""
    title = f"Single Circle Noise Test ({args.date.strftime('%B %d, %Y')})"
    mathjax_config = r'''<script>window.MathJax={tex:{inlineMath:[["\\(","\\)"]],displayMath:[["\\[","\\]"]]},svg:{fontCache:"local"}};</script>'''
    mathjax_script = '<script defer src="vendor/node_modules/mathjax/es5/tex-svg.js"></script>'
    html = [f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{title}</title>{mathjax_config}{mathjax_script}<style>{style}</style></head><body><main>',
            f'<h1>{title}</h1>',
            '<p class="subtitle">Completed single-circle study · 1,440 paired cases · 4,320 reconstructions</p>',
            '<h2 id="setup">1. Inverse problems and protocol</h2>',
            r'<ul><li><strong>CT:</strong> 10 equally spaced parallel-beam projection angles over \([0,\pi)\).</li>'
            r'<li><strong>MRI:</strong> single-coil Fourier measurements with variable-density Cartesian sampling at nominal \(4\times\) acceleration.</li>'
            r'<li><strong>Noise:</strong> Gaussian, with \(\|\varepsilon\|_2/\|A(x_{\mathrm{GT}})\|_2\) equal to 1%, 5%, 10% or 20%; complex noise on acquired MRI coefficients.</li>'
            r'<li><strong>Resolutions:</strong> \(64^2,96^2,128^2,160^2,200^2,250^2\).</li>'
            '<li><strong>Samples:</strong> 30 per resolution, inverse problem and noise level; paired across methods.</li>'
            '<li><strong>Settings:</strong> non-attention flow-matching UNets, 5 Euler steps, LBFGS with learning rate 1, '
            r'DLR rank \(r=8\), no additional regularization.</li></ul>',
            '<h2 id="results">2. Median relative image error</h2>',
            r'<p>Image error (%): \(100\|\hat{x}-x_{\mathrm{GT}}\|_2/\|x_{\mathrm{GT}}\|_2\). '
            'Medians include all 180 final reconstructions per row, pooled across resolutions. Best median in bold.</p><div class="tables">']
    for problem in ("ct", "mri"):
        acquisition_title = "CT · 10 angles" if problem == "ct" else r"MRI · \(4\times\) acceleration"
        html.append(f'<section><h3>{acquisition_title}</h3>'
                    '<table><thead><tr><th>Noise</th><th>D-Flow</th><th>P-Flow</th><th>DLR D-Flow</th></tr></thead><tbody>')
        for noise in NOISES:
            subset = [case for case in cases if case.problem == problem and case.noise == noise]
            medians = [statistics.median(case.errors[index] for case in subset) for index in range(3)]
            cells = [f'<td>{"<strong>" if value == min(medians) else ""}{value:.2f}%'
                     f'{"</strong>" if value == min(medians) else ""}</td>' for value in medians]
            html.append(f'<tr><td>{noise}%</td>{"".join(cells)}</tr>')
        html.append('</tbody></table></section>')
    html += ['</div>',
             '<h2 id="representatives">3. Best, median and worst by DLR image error</h2>',
             '<p>Within each resolution, noise level and inverse problem: DLR ranks 1, 15 and 30 of 30 samples '
             '(best, lower median and worst). All methods show the same case at their final iterate.</p>',
             r'<p class="small">Fixed display scales; errors use original arrays. Measurement errors reference clean \(A(x_{\mathrm{GT}})\).</p>']
    selections = []
    for noise in NOISES:
        html.append(f'<details class="gallery"><summary>{noise}% noise · best / median / worst</summary>')
        for resolution in RESOLUTIONS:
            html.append(rf'<details class="gallery"><summary>Resolution: \({resolution}^2\)</summary>')
            for problem in ("ct", "mri"):
                ordered = sorted((case for case in cases if (case.noise, case.resolution, case.problem) == (noise, resolution, problem)),
                                 key=lambda case: (case.errors[2], case.sample))
                assert len(ordered) == 30
                html.append(f'<details class="gallery"><summary>{problem.upper()} · 30 cases</summary>')
                for label, position in (("Best", 0), ("Median representative", 14), ("Worst", 29)):
                    case = ordered[position]
                    html.append(panel(case, label, assets))
                    selections.append({"section": "representatives", "label": label, "case": case.name,
                                       "rank": position + 1, "source": str(case.source)})
                html.append('</details>')
            html.append('</details>')
        html.append('</details>')
    html += ['<h2 id="gaps">4. Cases with the largest DLR advantage over D-Flow</h2>',
             r'<p>At 5%, 10% and 20% noise, show the three largest positive gaps \(\Delta=e_{\mathrm{D\text{-}Flow}}-e_{\mathrm{DLR}}\) '
             r'(percentage points), with DLR image error \(\leq10\%\), per resolution and inverse problem. '
             'This selection does not filter the tables above.</p>',
             '<button type="button" onclick="document.querySelectorAll(\'details.gallery\').forEach(d=>d.open=true)">Expand all examples</button> '
             '<button type="button" onclick="document.querySelectorAll(\'details.gallery\').forEach(d=>d.open=false)">Collapse all examples</button>']
    for noise in (5, 10, 20):
        html.append(f'<details class="gallery"><summary>{noise}% noise · DLR advantage by resolution</summary>')
        for resolution in RESOLUTIONS:
            html.append(rf'<details class="gallery"><summary>Resolution: \({resolution}^2\)</summary>')
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
                        label = rf"Example {index + 1} · \(\Delta = {case.gap:.2f}\) percentage points"
                        html.append(panel(case, label, assets))
                        selections.append({"section": "advantages", "label": label, "case": case.name,
                                           "gap_percentage_points": case.gap, "source": str(case.source)})
                    else:
                        html.append(f'<p class="empty">Example {index + 1}: unavailable — '
                                    r'no additional case has DLR error \(\leq10\%\) and lower error than D-Flow.</p>')
                html.append('</details>')
            html.append('</details>')
        html.append('</details>')
    html += ['</main></body></html>']
    document = '\n'.join(html)
    (args.out / f"{stem}.html").write_text(document)
    mathjax_bundle = (args.out / 'vendor/node_modules/mathjax/es5/tex-svg.js').read_text()
    standalone = document.replace(mathjax_script, '<script>' + mathjax_bundle.replace('</script', r'<\/script') + '</script>')
    for name in sorted({item['case'] for item in selections}):
        path = assets / f'{name}.png'
        encoded = base64.b64encode(path.read_bytes()).decode('ascii')
        standalone = standalone.replace(f'src="{assets.name}/{path.name}"', f'src="data:image/png;base64,{encoded}"')
    (args.out / f"{stem}_standalone.html").write_text(standalone)
    shutil.copyfile(args.data / 'runs.csv', assets / 'runs.csv')
    shutil.copyfile(args.data / 'summary.json', assets / 'summary.json')
    metadata = {"title": title, "data": str(args.data), "run_count": len(cases) * 3,
                "runs_sha256": hashlib.sha256((args.data / 'runs.csv').read_bytes()).hexdigest(),
                "representative_rule": "DLR ranks 1/15/30 within each noise/resolution/problem",
                "gallery_rule": "Noise 5/10/20%; DLR <=10%, positive D-Flow minus DLR gap; top 3 per noise/resolution/problem",
                "selections": selections}
    (assets / 'selection_manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({"report": str(args.out / f'{stem}.html'), "selections": len(selections),
                      "unique_images": len({item['case'] for item in selections})}))


if __name__ == "__main__":
    main()
