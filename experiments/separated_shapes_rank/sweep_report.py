"""Compare completed reconstructions on the same samples across DLR ranks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.ct_mri_sweep.run import TERMINAL, write_json


def report(root: Path) -> None:
    plan = json.loads((root / "plan.json").read_text())
    ranks = list(map(int, plan["rank_roots"]))
    lines = ["# Noiseless separated-shapes rank sweep", "",
             "Five paired samples (0–4) per resolution and inverse problem. CT10 / Cartesian MRI4×; five Euler steps; unchanged LBFGS and geometric Armijo, up to 1,000 attempts with the existing safeguards. Analytical alpha=1-t, FP32, no added regularization. Rank-specific source directories and reused runs are recorded in plan.json.", "",
             "Medians below use only samples completed at every rank in that row. Pending runs are not counted as failures. Image and measurement metrics use final checkpoints. Rank selection on these five samples is exploratory and needs confirmation on other samples. A winner at the largest tested rank does not establish a sufficient rank.", "",
             "| Resolution / problem | Common samples | " + " | ".join(f"Rank {rank} image error %" for rank in ranks) + " | D-Flow % | P-Flow % |",
             "| --- | ---: | " + " | ".join(["---:"] * (len(ranks) + 2)) + " |"]
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    figure, axes = plt.subplots(2, 6, figsize=(18, 6), constrained_layout=True)
    for column, resolution in enumerate(plan["resolutions"]):
        for row_index, problem in enumerate(("ct10", "mri4")):
            case = f"r{resolution}_shapes_{problem}"
            completed: dict[int, dict[int, dict[str, Any]]] = {}
            for rank in ranks:
                completed[rank] = {}
                for sample in plan["samples"]:
                    path = Path(plan["rank_roots"][str(rank)]) / "production" / case / f"sample_{sample:03d}" / "dlr_d_flow.json"
                    if not path.exists():
                        continue
                    data = json.loads(path.read_text())
                    if data["status"] not in TERMINAL:
                        continue
                    last = data["history"][-1]
                    item = {"resolution": resolution, "problem": problem, "sample": sample, "rank": rank,
                            "image_percent": 100 * last["image_relative_l2"],
                            "measurement_percent": 100 * last["measurement_relative_l2_noisy"],
                            "attempts": last["step"], "status": data["status"], "path": str(path)}
                    completed[rank][sample] = item
                    records.append(item)
            common = sorted(set.intersection(*(set(completed[rank]) for rank in ranks)))
            image_medians = [statistics.median(completed[rank][sample]["image_percent"] for sample in common)
                             if common else None for rank in ranks]
            residual_medians = [statistics.median(completed[rank][sample]["measurement_percent"] for sample in common)
                                if common else None for rank in ranks]
            controls = []
            for method in ("d_flow", "p_flow"):
                values = []
                for sample in common:
                    path = Path(plan["rank_roots"]["8"]) / "production" / case / f"sample_{sample:03d}" / f"{method}.json"
                    if path.exists():
                        data = json.loads(path.read_text())
                        if data["status"] in TERMINAL:
                            values.append(100 * data["history"][-1]["image_relative_l2"])
                controls.append(statistics.median(values) if values and len(values) == len(common) else None)
            lines.append(f"| {resolution}² / {problem} | {len(common)}/5 | " + " | ".join(
                "pending" if value is None else f"{value:.3f}" for value in image_medians + controls) + " |")
            rows.append({"case": case, "common_samples": common,
                         "completed_by_rank": {str(rank): len(completed[rank]) for rank in ranks},
                         "median_image_percent": dict(zip(map(str, ranks), image_medians)),
                         "median_measurement_percent": dict(zip(map(str, ranks), residual_medians)),
                         "control_image_percent": dict(zip(("D-Flow", "P-Flow"), controls))})
            axis = axes[row_index, column]
            if common:
                axis.plot(ranks, image_medians, marker="o", label="DLR")
                for label, value, color in zip(("D-Flow", "P-Flow"), controls, ("tab:orange", "tab:green")):
                    if value is not None:
                        axis.axhline(value, color=color, linestyle="--", label=label)
                axis.legend(fontsize=7)
            axis.set(xscale="log", title=f"{resolution}² {problem} (n={len(common)})", xlabel="Rank", ylabel="Median image error %")
            axis.set_xticks(ranks, labels=list(map(str, ranks)))
            axis.grid(alpha=.2)
    figure.savefig(root / "rank_comparison.png", dpi=140)
    plt.close(figure)
    lines += ["", "| Case | " + " | ".join(f"Completed rank {rank}" for rank in ranks) + " |",
              "| --- | " + " | ".join(["---:"] * len(ranks)) + " |",
              *[f"| {row['case']} | " + " | ".join(str(row["completed_by_rank"][str(rank)]) for rank in ranks) + " |" for row in rows], "",
              "Per-sample image errors, measurement residuals, attempt counts and termination reasons are saved in summary.json. Standard four-column reconstruction figures are saved inside each rank's production directory; unavailable controls are marked explicitly."]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    write_json(root / "summary.json", {"rows": rows, "records": records})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    report(parser.parse_args().out)
