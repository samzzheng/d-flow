"""CPU-only paired reports for DLR Gradient Flow reconstruction pilots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from scripts.reconstruction_figures import RESIDUAL_VMAX, plot_reconstruction_comparison

Record = dict[str, Any]
LABELS = {"exact": "D-Flow", "proxy": "P-Flow", "fresh": "DLR D-Flow",
          "secant": "DLR Gradient Flow (corrected)",
          "secant_transport": "DLR Gradient Flow (Jacobian secant)",
          "transport": "DLR Gradient Flow (literal)",
          "finite_difference": "DLR Gradient Flow (finite difference)"}
BASELINE_FILES = {"exact": "d_flow", "proxy": "p_flow", "fresh": "dlr_d_flow"}


def relative_percent(estimate: torch.Tensor, reference: torch.Tensor) -> float | None:
    """Measure original arrays, subtracting complex data before magnitudes."""
    dtype = torch.complex128 if reference.is_complex() else torch.float64
    target, value = reference.to(dtype), estimate.to(dtype)
    norm = float(target.norm())
    return 100 * float((value - target).norm()) / norm if norm else None


def save_panels(destination: Path, inputs: Record, records: dict[str, Record],
                problem: str, title: str, *, experimental_mode: str | None = None) -> None:
    """Use the shared renderer, explicitly relabeling the experimental column."""
    panels = {LABELS[mode]: (record["image"], record["measurement"])
              for mode, record in records.items() if mode in BASELINE_FILES}
    histories = {LABELS[mode]: record["history"]
                 for mode, record in records.items() if mode in BASELINE_FILES}
    figure = plot_reconstruction_comparison(
        inputs["truth"], inputs["clean"], panels, observation=inputs["observed"],
        histories=histories, residual_vmax=RESIDUAL_VMAX[problem], title=title)
    if experimental_mode is not None:
        for axis in figure.axes:
            axis.set_title(axis.get_title().replace("DLR D-Flow", LABELS[experimental_mode]))
        figure.supxlabel(figure.get_supxlabel() + "\n"
                         f"Experimental fourth column: {LABELS[experimental_mode]}; "
                         "other columns retain their D-Flow / P-Flow baselines.", fontsize=9)
    for extension in ("png", "pdf"):
        figure.savefig(destination.with_suffix(f".{extension}"), dpi=140)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path,
                        default=Path("outputs/ct_mri_sweep_fp32_20261009"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    output = args.root / "report"
    output.mkdir(parents=True, exist_ok=True)
    baseline_settings = json.loads((args.baseline / "manifest.json").read_text())["settings"]
    rows: list[Record] = []
    pairing: list[Record] = []
    pending: list[str] = []
    for protocol_path in sorted((args.root / "recon").rglob("protocol.json")):
        protocol = json.loads(protocol_path.read_text())
        case = protocol["case"]
        input_path = args.baseline / case / "inputs.pt"
        inputs = torch.load(input_path, map_location="cpu", weights_only=True)
        if hashlib.sha256(input_path.read_bytes()).hexdigest() != protocol["inputs_sha256"]:
            raise ValueError(f"Paired input hash mismatch: {case}")
        if inputs["protocol"] != protocol["source_protocol"]:
            raise ValueError(f"Source/checkpoint/acquisition protocol mismatch: {case}")
        if protocol["settings"] != baseline_settings:
            raise ValueError(f"Base reconstruction settings differ: {case}")
        checkpoint = Path(baseline_settings["run_template"].format(**inputs["protocol"]["job"])) / (
            f"checkpoints/epoch_{baseline_settings['epoch']:04d}.pt")
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != inputs["protocol"]["checkpoint_sha256"]:
            raise ValueError(f"Current model checkpoint hash mismatch: {checkpoint}")
        pairing.append({"case": case, "inputs_sha256": protocol["inputs_sha256"],
                        "checkpoint_sha256": protocol["source_protocol"]["checkpoint_sha256"],
                        "source_seed": protocol["source_protocol"]["source_seed"],
                        "passed": True})
        baseline: dict[str, Record] = {}
        reruns: dict[str, Record] = {}
        for mode, filename in BASELINE_FILES.items():
            path = args.baseline / case / f"{filename}.pt"
            if path.exists():
                baseline[mode] = torch.load(path, map_location="cpu", weights_only=True)
                if baseline[mode]["protocol"] != inputs["protocol"]:
                    raise ValueError(f"Saved baseline pairing mismatch: {path}")
        for path in sorted(protocol_path.parent.glob("*/state.pt")):
            mode = path.parent.name
            state = torch.load(path, map_location="cpu", weights_only=True)
            if state["case"] != case or state["mode"] != mode:
                raise ValueError(f"Reconstruction case/mode mismatch: {path}")
            # Use one atomic state snapshot for image, measurement, and history;
            # result.json can be replaced just before the next state snapshot.
            reruns[mode] = state
        if not reruns:
            pending.append(case)
        for origin, records in (("baseline", baseline), ("rerun", reruns)):
            for mode, record in records.items():
                history = record["history"]
                final, best = history[-1], min(history, key=lambda item: item["image_relative_l2"])
                accepted = sum(point.get("source_step_norm", 0) > 0 for point in history[1:])
                rows.append({
                    "case": case, "method": LABELS[mode], "mode": mode, "origin": origin,
                    "status": record["status"], "optimizer": record.get("optimizer", "lbfgs")
                    if origin == "rerun" else "lbfgs",
                    "attempts": final["step"], "accepted_updates": accepted,
                    "failed_attempts": final["step"] - accepted,
                    "image_percent": relative_percent(record["image"], inputs["truth"]),
                    "clean_measurement_percent": relative_percent(record["measurement"], inputs["clean"]),
                    "observed_residual_percent": relative_percent(record["measurement"], inputs["observed"]),
                    "final_objective": final["phi"],
                    "best_seen_image_percent_retrospective": 100 * best["image_relative_l2"],
                    "best_seen_attempt_retrospective": best["step"],
                    "residual_at_best_image_percent_retrospective":
                        100 * best["measurement_relative_l2_noisy"],
                    "clean_residual_at_best_image_percent_retrospective":
                        100 * best["measurement_relative_l2_clean"],
                })
        case_output = output / case
        case_output.mkdir(parents=True, exist_ok=True)
        problem = inputs["protocol"]["job"]["problem"]
        save_panels(case_output / "baseline", inputs, baseline, problem,
                    f"{case} | paired FP32 baselines | final saved iterates")
        for mode, record in reruns.items():
            if mode not in BASELINE_FILES:
                companion = {**baseline, "fresh": record}
                save_panels(case_output / mode, inputs, companion, problem,
                            f"{case} | {LABELS[mode]} | final saved iterates",
                            experimental_mode=mode)
        selected = {**baseline, **reruns}
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
        for mode, record in selected.items():
            history = record["history"]
            label = LABELS[mode] + (" [rerun]" if mode in reruns else " [baseline]")
            for axis, key in zip(axes, ("image_relative_l2", "measurement_relative_l2_noisy")):
                line, = axis.plot([point["step"] for point in history],
                                  [100 * point[key] for point in history], label=label)
                if record["status"] == "stalled":
                    axis.plot(history[-1]["step"], 100 * history[-1][key], "x", color=line.get_color())
        for axis, label in zip(axes, ("Relative image L2 error (%)", "Observed measurement residual (%)")):
            axis.set(xlabel="Optimizer attempts (including rejected searches)", ylabel=label, ylim=(0, None))
            axis.grid(alpha=.2)
            axis.legend(fontsize=7)
        figure.suptitle(f"{case} | paired FP32 | crosses indicate stalled termination")
        for extension in ("png", "pdf"):
            figure.savefig((case_output / "convergence").with_suffix(f".{extension}"), dpi=160)
        plt.close(figure)

    benchmarks: list[Record] = []
    for path in sorted(args.root.glob("benchmark*/benchmark.json")):
        record = json.loads(path.read_text())
        for mode, measurements in record["methods"].items():
            if measurements:
                benchmarks.append({"case": record["case"], "mode": mode, "method": LABELS[mode],
                    "file": str(path), "n": len(measurements),
                    "gradient_seconds_median": statistics.median(point["seconds"] for point in measurements),
                    "relative_gradient_error_percent_median": 100 * statistics.median(
                        point["relative_gradient_error"] for point in measurements),
                    "cosine_median": statistics.median(point["cosine"] for point in measurements),
                    "displacement_norm": record["displacement_norm"]})
    summary = {"pairing": pairing, "reconstructions": rows, "benchmarks": benchmarks,
               "pending_cases": pending,
               "unfinished_runs": [f"{row['case']}/{row['mode']}" for row in rows
                                   if row["origin"] == "rerun"
                                   and row["status"] not in {"step_limit", "stalled", "converged"}],
               "selection_note":
               "First two samples per selected acquisition form a small pilot, not a population estimate. "
               "Best image errors are retrospective selections from recorded histories; "
               "their own residuals are paired, but figures always use final saved iterates.",
               "benchmark_scope": "Synchronized cached-objective gradient wall time. "
               "Excludes preceding canonical objective, optimizer, line search, and warmup; "
               "exact gradient includes its necessary differentiable Euler forward/backward."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if rows:
        with (output / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# DLR Gradient Flow pilot", "", summary["selection_note"], "",
             "Corrected endpoint KSL is a secant-plus-correction variant. Literal Section 11 "
             "uses directional second derivatives. Jacobian-secant KSL instead uses the difference "
             "between the old and new true Jacobians. These are distinct algorithms.", "",
             f"Paired input, source/acquisition protocol, and checkpoint hashes passed for {len(pairing)} cases.", "",
             "Final metrics use original arrays from a single saved checkpoint. Runs may stop after "
             "two failed searches before their attempt limit; accepted updates are reported separately.", "",
             "| Case | Method | Origin | Status | Attempts / accepted | Image L2 % | Residual % |",
             "|---|---|---|---|---:|---:|---:|"]
    for row in rows:
        image = "n/a" if row["image_percent"] is None else f"{row['image_percent']:.3f}"
        residual = "n/a" if row["observed_residual_percent"] is None else f"{row['observed_residual_percent']:.3f}"
        lines.append(f"| {row['case']} | {row['method']} | {row['origin']} | {row['status']} | "
                     f"{row['attempts']} / {row['accepted_updates']} | {image} | {residual} |")
    lines += ["", "## Fixed-source gradient benchmarks", "", summary["benchmark_scope"], "",
              "These are local gradient-accuracy measurements, not reconstruction success rates. "
              "Gradient errors are measured against the native 100-step Euler gradient at the same source.", "",
              "| Case | Method | Median seconds | Gradient error % | Repeats |",
              "|---|---|---:|---:|---:|"]
    for row in benchmarks:
        lines.append(f"| {row['case']} | {row['method']} | {row['gradient_seconds_median']:.4f} | "
                     f"{row['relative_gradient_error_percent_median']:.4f} | {row['n']} |")
    lines += ["", "Figures use fixed measurement-residual color scales: CT 0–1, MRI 0–25, `amp`. "
              "Companion figures explicitly replace only the fourth column with the named experimental method. "
              "CSV/JSON include retrospective best-image metrics and residuals at those same iterates."]
    if pending:
        lines += ["", "No new reconstruction snapshot yet: " + ", ".join(pending) + "."]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(output), "paired_cases": len(pairing),
                      "metric_rows": len(rows), "benchmark_rows": len(benchmarks), "pending": pending}))


if __name__ == "__main__":
    main()
