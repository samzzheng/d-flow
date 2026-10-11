"""Summarize terminal noisy results; compare ranks only on completed five-rank sets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.ct_mri_sweep.run import TERMINAL, Settings, write_json


def report(out: Path) -> None:
    lines = ["# Paired 1%-noise sweeps", "",
             "100-step Euler, full FP32, shared LBFGS/geometric Armijo, up to100 attempts. "
             "Two failed searches stop a run. Step limit is not a convergence claim. "
             "Final saved checkpoints; no truth-based selection. "
             "Noise is1% relative to clean measurement norm; residuals below use observed y as denominator.", "",
             "| Study | Terminal runs | Stalled | Median final image error % | Median observed residual % |",
             "| --- | --- | --- | --- | --- |"]
    summaries: Settings = {}
    for name in ("comparison", "rank_2", "rank_4", "rank_8", "rank_12", "rank_16"):
        for method in (("d_flow", "p_flow", "dlr_d_flow") if name == "comparison" else ("dlr_d_flow",)):
            records = [json.loads(path.read_text()) for path in (out / name).glob(f"r*/sample_*/{method}.json")]
            records = [item for item in records if item["status"] in TERMINAL]
            summary = {"terminal": len(records), "stalled": sum(item["status"] == "stalled" for item in records),
                "median_image_percent": statistics.median(100 * item["history"][-1]["image_relative_l2"] for item in records) if records else None,
                "median_residual_percent": statistics.median(100 * item["history"][-1]["measurement_relative_l2_noisy"] for item in records) if records else None}
            summaries[f"{name}/{method}"] = summary
            values = [str(summary[key]) if summary[key] is not None else "n/a" for key in summary]
            lines.append(f"| {name}/{method} | " + " | ".join(values) + " |")
    lines += ["", "Partial medians above can cover different cases and must not be used to rank methods.", "",
              "| Acquisition | Completed five-rank cases | Median final image error %, ranks2/4/8/12/16 |", "| --- | --- | --- |"]
    paired: Settings = {}
    for acquisition in sorted((out / "rank_4").glob("r*")):
        rows = []
        for case in sorted(acquisition.glob("sample_*")):
            paths = [out / f"rank_{rank}" / acquisition.name / case.name / "dlr_d_flow.json" for rank in (2, 4, 8, 12, 16)]
            if not all(path.exists() for path in paths):
                continue
            records = [json.loads(path.read_text()) for path in paths]
            if not all(record["status"] in TERMINAL for record in records):
                continue
            keys = ("checkpoint_sha256", "test_subset_sha256", "job", "sample", "source_seed", "noise_seed", "dlr_seed", "operator")
            assert all(all(record["protocol"][key] == records[0]["protocol"][key] for key in keys) for record in records)
            rows.append({"sample": case.name, "ranks": [{"rank": rank, "status": record["status"],
                         "final": record["history"][-1],
                         "best_retrospective": min(record["history"], key=lambda item: item["image_relative_l2"])}
                         for rank, record in zip((2, 4, 8, 12, 16), records)]})
        paired[acquisition.name] = rows
        if not rows:
            continue
        medians = [statistics.median(100 * row["ranks"][index]["final"]["image_relative_l2"] for row in rows) for index in range(5)]
        lines.append(f"| {acquisition.name} | {len(rows)}/10 | " + " / ".join(f"{value:.3f}" for value in medians) + " |")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        for row in rows:
            for field, style in (("final", "-o"), ("best_retrospective", "--x")):
                errors = [100 * item[field]["image_relative_l2"] for item in row["ranks"]]
                residuals = [100 * item[field]["measurement_relative_l2_noisy"] for item in row["ranks"]]
                axes[0].plot((2, 4, 8, 12, 16), errors, style, alpha=.6)
                axes[1].plot(residuals, errors, style, alpha=.6)
        axes[0].set(xlabel="Rank", ylabel="Relative image error (%)", xticks=[2, 4, 8, 12, 16])
        axes[1].set(xlabel="Residual against observed y (%)", ylabel="Relative image error (%)")
        fig.suptitle(f"{acquisition.name}: {len(rows)} paired samples, 1% noise\nSolid: final; dashed: retrospective best-image checkpoint and its own residual")
        fig.savefig(out / f"rank_summary_{acquisition.name}.png", dpi=140)
        plt.close(fig)
    write_json(out / "summary.json", {"partial_counts": summaries, "paired_rank_cases": paired})
    (out / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    report(parser.parse_args().out)
