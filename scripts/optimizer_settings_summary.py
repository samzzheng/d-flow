"""Summarize historical optimizer sweeps and the CPU scaling-policy audit."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/optimizer_settings_2026_09_27"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = []
    for root in (ROOT / "outputs/optimizer_diagnosis", ROOT / "outputs/source_prior_sweep"):
        for path in sorted(root.rglob("samples.csv")):
            settings = json.loads((path.parent / "settings.json").read_text())
            with path.open() as handle:
                samples = list(csv.DictReader(handle))
            with (path.parent / "iterations.csv").open() as handle:
                iterations = list(csv.DictReader(handle))
            for sample in samples:
                rows = [row for row in iterations if all(row[key] == sample[key]
                        for key in ("resolution", "sample_index", "method"))]
                accepted = [int(row.get("function_evaluations") or 0) for row in rows
                            if int(row["closure_evaluations"]) > 0]
                failed = [int(row.get("function_evaluations") or 0) for row in rows
                          if int(row["closure_evaluations"]) == 0]
                inventory.append({"source": str(path.relative_to(ROOT)), **sample,
                                  "settings": settings,
                                  "accepted_search_max_trials": max(accepted, default=0),
                                  "accepted_search_median_trials": statistics.median(accepted) if accepted else 0,
                                  "steps_with_no_gradient": len(failed),
                                  "trials_at_steps_with_no_gradient": sum(failed)})
    (OUT / "historical_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    paths = [("legacy_s1.json", "Existing: rescale whenever history is empty"),
             ("once_s1.json", "Candidate: rescale only on first iteration")]
    if not all((OUT / name).exists() for name, _ in paths):
        print("Historical inventory written; paired solve results are not both present yet.")
        return
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), layout="constrained")
    summary = []
    for name, label in paths:
        data = json.loads((OUT / name).read_text())
        run = data["runs"][0]
        steps = run["steps"]
        indices = [row["step"] for row in steps]
        axes[0].plot(indices, [100 * row["residual"] for row in steps], "o-", label=label)
        axes[1].plot(indices, [100 * row["image_relative_l2"] for row in steps], "o-")
        axes[2].semilogy(indices, [row["accepted_step_norm"] for row in steps], "o-")
        summary.append({"source": name, "sample": run["sample"], "policy": run["policy"],
                        "final": steps[-1], "protocol": data["protocol"]})
    axes[0].set_ylabel("Measurement residual (%)")
    axes[1].set_ylabel("Clamped image relative L2 error (%)")
    axes[2].set_ylabel("Accepted latent step L2 norm")
    axes[2].axhline(2.5, color="0.5", linestyle="--", linewidth=1)
    for ax in axes:
        ax.set_xlabel("Optimizer iteration")
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Exact D-Flow, 100 Euler steps, non-attention 250², sample 1, 1% noise; CPU, 4 threads")
    fig.savefig(OUT / "scaling_policy_s1.png", dpi=180)
    fig.savefig(OUT / "scaling_policy_s1.pdf")
    plt.close(fig)
    (OUT / "paired_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    heldout_path = OUT / "once_s0.json"
    baseline_dir = ROOT / "outputs/accuracy_fast_euler_2026_09_27/exact_from_z0"
    if heldout_path.exists() and (baseline_dir / "results.json").exists():
        heldout = json.loads(heldout_path.read_text())["runs"][0]
        baseline = json.loads((baseline_dir / "results.json").read_text())["runs"][0]
        reference = {row["step"]: row for row in baseline["steps"]}
        metric_keys = ["residual", "image_relative_l2", "raw_image_relative_l2", "source_norm_over_sqrt_d"]
        equal_metrics = all(row["objective"] == reference[row["step"]]["true_objective"]
                            and all(row[key] == reference[row["step"]][key] for key in metric_keys)
                            for row in heldout["steps"])
        control = {"sample": 0, "steps_compared": len(heldout["steps"]),
                   "all_reported_metrics_exactly_equal": equal_metrics,
                   "candidate_final": heldout["steps"][-1],
                   "baseline_final": reference[heldout["steps"][-1]["step"]]}
        if "final_source" in heldout:
            candidate_source = torch.load(heldout["final_source"], map_location="cpu", weights_only=False)["source"]
            baseline_sources = torch.load(baseline_dir / "sources_sample0_exact.pt", map_location="cpu", weights_only=False)["sources"]
            control["final_source_bitwise_equal"] = torch.equal(candidate_source, baseline_sources[heldout["steps"][-1]["step"]])
        (OUT / "heldout_control.json").write_text(json.dumps(control, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
