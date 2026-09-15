"""Update the HTML report after the DLR-D-Flow sweep completes."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "outputs/reports/dflow_dlr_completion_report.html"
SWEEP = ROOT / "dlr_dflow/outputs/sweep/dlr_dflow_sweep_results.json"


def fmt(value: float) -> str:
    return f"{value:.4e}"


def main() -> None:
    rows = json.loads(SWEEP.read_text())
    best = min(rows, key=lambda row: row["image_mse"])
    note = (
        '<div class="note"><strong>Outcome.</strong> The DLR-D-Flow sweep selected '
        f'init=<code>{best["init"]}</code>, seed=<code>{best["seed"]}</code>, '
        f'rank=<code>{best["rank"]}</code>, lr=<code>{best["lr"]}</code>, '
        f'steps=<code>{best["steps"]}</code>. Best data-misfit MSE '
        f'<code>{fmt(best["best_misfit"])}</code> at step '
        f'<code>{best["best_step"]}</code>; best-solution image MSE '
        f'<code>{fmt(best["image_mse"])}</code>. The sweep output is '
        '<code>dlr_dflow/outputs/sweep/dlr_dflow_sweep_results.json</code>.<br>'
        '<strong>Interpretation.</strong> The DLR inverse solve is sensitive to '
        'initialization and optimizer settings. The sweep result should be treated as '
        'the current best toy demonstration, while the action-error diagnostic remains '
        'the primary check that the low-rank sensitivity approximation is behaving '
        'coherently.</div>'
    )
    row = (
        '<tr><td>Selected deblurring sweep</td>'
        f'<td>init=<code>{best["init"]}</code>, rank <code>{best["rank"]}</code>, '
        f'lr <code>{best["lr"]}</code>: best data-misfit MSE '
        f'<code>{fmt(best["best_misfit"])}</code>; image MSE '
        f'<code>{fmt(best["image_mse"])}</code></td></tr>'
    )

    text = REPORT.read_text()
    text = re.sub(
        r'<div class="note"><strong>Outcome\.</strong> The best completed rerun.*?</div>',
        note,
        text,
        count=1,
        flags=re.S,
    )
    text = re.sub(
        r'<tr><td>Selected deblurring (?:rerun|sweep)</td>.*?</tr>',
        row,
        text,
        count=1,
        flags=re.S,
    )
    REPORT.write_text(text)
    print(f"Updated {REPORT} with sweep best: {best}")


if __name__ == "__main__":
    main()
