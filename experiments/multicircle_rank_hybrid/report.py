"""Refresh descriptive reports from atomic worker records, including missing work."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
from typing import Any

from .hybrid import SCHEDULES


def report(output: Path) -> None:
    rank_records = [json.loads(path.read_text()) for path in sorted((output / "ranks").glob("*.json"))]
    lines = ["# Independent Euler-Jacobian effective ranks", "",
             "Epoch-200 non-attention models, FP32 with TF32 disabled, 100 Euler steps. "
             "These are ranks of the source-to-image Jacobian at specified sources, not ranks of image arrays.", "",
             "Energy means sum of squared singular values. 99%, 99.9%, 99.99% retention corresponds "
             "to 10%, 3.16%, 1% relative Frobenius truncation error. Tail energy uses 24 independent "
             "Gaussian products; unresolved means the width/refinement budget did not establish the rank. "
             "A dominant singular value can produce a small energy rank while important reconstruction "
             "directions remain outside that rank. Initial and optimized sources are reported separately.", "",
             "| Resolution | Circles | Initial sources resolved / measured | r99 median [range] | r99.9 median [range] | r99.99 median [range] |",
             "| --- | --- | --- | --- | --- | --- |"]
    rows: list[dict[str, Any]] = []
    for resolution in (64, 96, 128, 160, 200, 250):
        for circles in (1, 2, 3, 4):
            selected = [r for r in rank_records if r["resolution"] == resolution and
                        r["circles"] == circles and r["source_kind"] == "initial"]
            resolved = [r for r in selected if r["status"] == "resolved_estimate"]
            available = Path(f"outputs/fm_unet/circles-{resolution}-noattn-ch16-b8-e200/n{circles}/checkpoints/epoch_0200.pt").exists()
            count = f"{len(resolved)}/{len(selected)}" if available else "checkpoint unavailable"
            cells = []
            for target in ("0.99", "0.999", "0.9999"):
                values = [r["latest"]["effective_ranks"][target] for r in resolved]
                cells.append(f"{statistics.median(values):g} [{min(values)}, {max(values)}]" if values else "pending/unresolved")
            lines.append(f"| {resolution}² | {circles} | {count} | " + " | ".join(cells) + " |")
    for record in rank_records:
        latest = record["latest"]
        rows.append({**{key: record[key] for key in ("resolution", "circles", "sample", "source_kind", "source_path", "status")},
                     **{f"rank_{key}": value for key, value in latest["effective_ranks"].items()},
                     **{f"count_sigma_ratio_{key}": value for key, value in latest["singular_threshold_counts"].items()},
                     "width": latest["width"], "tail_energy_fraction": latest["omitted_energy_fraction"],
                     "triplet_residual": latest["relevant_triplet_residual"]})
    lines += ["", "Individual source estimates, threshold counts, and unresolved results are in ranks.csv and ranks/*.json. "
              "Threshold counts are for computed singular values only; counts at the width limit are censored. "
              "The report does not select a production DLR rank."]
    (output / "rank_report.md").write_text("\n".join(lines) + "\n")
    if rows:
        with (output / "ranks.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# Paired P-Flow / DLR alternation pilot", "",
             "Rank 8 retained as the existing control; no new rank selected. Noiseless CT10/MRI4, "
             "100 Euler steps, FP32, up to 100 optimization attempts, same input files and LBFGS/Armijo settings. "
             "Histories and cached gradients are cleared at estimator switches without repeating first-step scaling. "
             "The 1:1 hybrid therefore uses no accumulated curvature. Pure controls stop after two failures; "
             "hybrids stop after two entirely failed schedule cycles. Ratios refer to attempts, not accepted updates.", "",
             "Only completed four-way matched cases enter the summary. Final checkpoints, not best-image selection.", "",
             "| Case | P-Flow image % | DLR image % | 1P:1DLR image % | 3P:1DLR image % |",
             "| --- | --- | --- | --- | --- |"]
    paired = []
    for case in sorted((output / "hybrid").glob("r*/sample_*")):
        paths = [case / f"{variant}.json" for variant in SCHEDULES]
        if not all(path.exists() for path in paths):
            continue
        items = [json.loads(path.read_text()) for path in paths]
        if not all(item["status"] in {"step_limit", "stalled"} for item in items):
            continue
        assert all(item["protocol"] == items[0]["protocol"] for item in items)
        errors = [100 * item["history"][-1]["image_relative_l2"] for item in items]
        lines.append(f"| {case.parent.name}/{case.name} | " + " | ".join(f"{value:.3f}" for value in errors) + " |")
        paired.append(errors)
    lines += ["", f"Completed paired cases: {len(paired)}."]
    if paired:
        lines += ["Median final image errors (P / DLR / 1:1 / 3:1): " +
                  " / ".join(f"{statistics.median(column):.3f}%" for column in zip(*paired)) + "."]
    (output / "hybrid_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    report(parser.parse_args().output)
