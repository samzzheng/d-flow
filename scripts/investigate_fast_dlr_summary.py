"""Summarize the saved fast-DLR accuracy study without model evaluations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_result(path):
    """Read one immutable view; tolerate files not yet emitted by a worker."""
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def paired_tail_summary(result, control_rows):
    """Compare absolute attempts, including costs added since the shared fork."""
    fork = result["intervention"]["original_step"]
    control = {row["step"]: row for row in control_rows}
    tail = {row["step"]: row for row in result.get("steps", [])}
    baseline = control.get(fork)
    metrics = ("objective", "residual", "residual_over_noise", "image_relative_l2", "raw_image_relative_l2")
    costs = ("gradient_evaluations", "true_forward_evaluations", "line_search_trials",
             "gradient_seconds", "elapsed_seconds", "failed_searches")
    pairs = []
    for attempt in range(fork, fork + result["settings"]["steps"] + 1):
        tail_row, control_row = tail.get(attempt), control.get(attempt)
        pairs.append({"attempt": attempt, "tail": tail_row, "original_sketch32": control_row,
            "tail_minus_original": ({key: tail_row[key] - control_row[key] for key in metrics}
                                    if tail_row is not None and control_row is not None else None),
            "tail_added_costs": ({key: tail_row.get(f"added_{key}") for key in costs}
                                 if tail_row is not None else None),
            "original_added_costs": ({key: control_row[key] - baseline[key] for key in costs
                                      if key in control_row and key in baseline}
                                     if control_row is not None and baseline is not None else None)})
    paired = [row for row in pairs if row["tail"] is not None and row["original_sketch32"] is not None]
    return {"control_run": "fresh_sketch32_s0", "fork_attempt": fork,
        "baseline_metrics_identical": (all(tail[fork][key] == baseline[key] for key in metrics)
                                       if fork in tail and baseline is not None else None),
        "latest_paired_attempt": paired[-1]["attempt"] if paired else None,
        "cost_scope": "Additional work after the saved fork; each arm retains the same original totals. "
                      "Elapsed times reflect separate CPU executions and are not a controlled speed comparison.",
        "pairs": pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
        default=Path("outputs/fast_accuracy_recovery_2026_09_27"))
    args = parser.parse_args()
    root = args.root
    summary = {"point_checks": {}, "reconstructions": {}, "gpu_timing": None,
               "tail_interventions": {}, "endpoint_diagnostics": {}}
    for name in ("points_step20_seed0", "points_rank12", "nested_seed0", "nested_seed1",
                 "heldout_sample1_step44"):
        path = root / name / "results.json"
        result = read_result(path)
        if result is not None:
            summary["point_checks"][name] = {
                "settings": result["settings"], "protocol": result["protocol"],
                "initial": result["initial"], "runs": result["runs"]}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), constrained_layout=True)
    closeup, close_axes = plt.subplots(1, 2, figsize=(10, 4.1), constrained_layout=True)
    reconstruction_rows = {}
    for folder, label in (("fresh_fast_s0", "Fast DLR, oversampling 8"),
                          ("fresh_sketch32_s0", "Fast DLR, oversampling 32"),
                          ("fresh_exact_s0", "Exact D-Flow")):
        path = root / folder / "results.json"
        result = read_result(path)
        if result is None:
            continue
        if "runs" in result:
            if not result["runs"]:
                continue
            run = result["runs"][0]
            rows = run["steps"]
            stop_reason = run.get("stop_reason")
            step_key = "step"
        else:
            rows = result["history"]
            stop_reason = result.get("stop_reason")
            step_key = "attempt"
        if not rows:
            continue
        reconstruction_rows[folder] = rows
        summary["reconstructions"][folder] = {
            "label": label, "protocol": result["protocol"],
            "armijo_progress_policy": result["protocol"].get("armijo_progress_policy", "legacy_rounded_rhs"),
            "last": rows[-1], "stop_reason": stop_reason,
            "at_iteration": {str(n): next((r for r in rows if r[step_key] == n), None)
                             for n in (10, 20, 30, 40)},
            "first_crossings": {str(tau): next((r for r in rows
                if r["residual_over_noise"] <= tau), None) for tau in (2, 1.1, 1.05, 1)}}
        xs = [r[step_key] for r in rows]
        line, = axes[0].semilogy(xs, [100*r["image_relative_l2"] for r in rows], label=label)
        axes[1].semilogy(xs, [r["residual_over_noise"] for r in rows], label=label)
        close_rows = [r for r in rows if 25 <= r[step_key] <= 40]
        if close_rows:
            close_x = [r[step_key] for r in close_rows]
            close_axes[0].semilogy(close_x, [r["residual_over_noise"] for r in close_rows],
                                  color=line.get_color(), marker=".", markersize=3, label=label)
            close_axes[1].semilogy(close_x, [100*r["image_relative_l2"] for r in close_rows],
                                  color=line.get_color(), marker=".", markersize=3, label=label)

    for folder in ("tail_reset_from32", "tail_rank12_from32"):
        result = read_result(root / folder / "results.json")
        if result is None:
            continue
        rows = result.get("steps", [])
        fork = result["intervention"]["original_step"]
        intervention_label = "Curvature reset" if result["settings"]["mode"] == "reset" else "Rank 12 + curvature reset"
        label = f"{intervention_label}, fork at {fork}"
        summary["tail_interventions"][folder] = {
            "label": label, "status": "complete" if "stop_reason" in result else "in_progress",
            "settings": result["settings"], "protocol": result["protocol"],
            "armijo_progress_policy": result["protocol"].get("armijo_progress_policy", "legacy_rounded_rhs"),
            "intervention": result["intervention"], "last": rows[-1] if rows else None,
            "stop_reason": result.get("stop_reason"), "original_counts": result["original_counts"],
            "original_elapsed_seconds": result.get("original_elapsed_seconds"),
            "same_attempt_control": paired_tail_summary(result, reconstruction_rows.get("fresh_sketch32_s0", []))}
        if not rows:
            continue
        xs = [row["step"] for row in rows]
        color = "#9467bd" if result["settings"]["mode"] == "reset" else "#8c564b"
        style = dict(color=color, ls="--", lw=2, marker="o", markersize=3, label=label)
        axes[0].semilogy(xs, [100*row["image_relative_l2"] for row in rows], **style)
        axes[1].semilogy(xs, [row["residual_over_noise"] for row in rows], **style)
        close_axes[0].semilogy(xs, [row["residual_over_noise"] for row in rows], **style)
        close_axes[1].semilogy(xs, [100*row["image_relative_l2"] for row in rows], **style)
        for ax in close_axes:
            ax.axvline(fork, color=color, ls=":", lw=1, alpha=.6)

    diagnostic_paths = {root / "plateau_sketch32_step32" / "results.json"}
    diagnostic_paths.update(root.glob("endpoint*/results.json"))
    for path in sorted(diagnostic_paths):
        result = read_result(path)
        if result is not None:
            summary["endpoint_diagnostics"][path.parent.name] = result
    summary["optimizer_acceptance_replay"] = read_result(root / "armijo_roundoff_replay.json")
    axes[0].set(ylabel="Clamped image relative L2 error (%)", xlabel="Optimizer attempt")
    axes[1].set(ylabel="Raw measurement residual / noise norm", xlabel="Optimizer attempt")
    axes[1].axhline(1, color="black", ls="--", lw=1, label="Noise norm")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("250 × 250 non-attention model · sample 0 · Euler 100 · 1% noise")
    for ext in ("png", "pdf"):
        fig.savefig(root / f"reconstruction_comparison.{ext}", dpi=180)
    plt.close(fig)
    close_axes[0].set(ylabel="Raw measurement residual / noise norm", xlabel="Optimizer attempt", xlim=(25, 40))
    close_axes[1].set(ylabel="Clamped image relative L2 error (%)", xlabel="Optimizer attempt", xlim=(25, 40))
    close_axes[0].axhline(1, color="black", ls="--", lw=1, label="Noise norm")
    for ax in close_axes:
        ax.set_yscale("linear")
        ax.grid(alpha=.2)
    close_axes[0].legend(fontsize=8, loc="upper right")
    closeup.suptitle("250 × 250 · sample 0 · Euler 100 · late optimization"
                     + ("\nDashed intervention curves fork from saved attempt 32"
                        if summary["tail_interventions"] else ""))
    for ext in ("png", "pdf"):
        closeup.savefig(root / f"reconstruction_tail_closeup.{ext}", dpi=180)
    plt.close(closeup)

    timing_path = root / "gpu_sketch_timing_chunked.json"
    result = read_result(timing_path)
    if result is not None:
        summary["gpu_timing"] = result
        if result.get("status") == "complete":
            summary["gpu_timing"]["oversampling32_overhead_fraction"] = (
                result["medians"]["32"] / result["medians"]["8"] - 1)
    summary["derivative_column_budget"] = {
        "fast": 36 + 99 * 20,
        "sketch32": 84 + 99 * 20,
        "trace8": 40 + 99 * 24,
        "rank12": 44 + 99 * 28,
        "rank12_sketch32": 92 + 99 * 28,
        "scope": "100 Euler steps; count scalar JVP/VJP columns, not wall time"}
    temporary = root / "summary.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(root / "summary.json")
    print(json.dumps({name: {"last": run["last"], "stop_reason": run["stop_reason"]}
                      for name, run in summary["reconstructions"].items()}, indent=2))


if __name__ == "__main__":
    main()
