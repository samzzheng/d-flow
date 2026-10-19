"""Summarize the bounded 250px execution experiments; never run a GPU job."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import benchmark_dflow_dlr_sparse_ct_circles_gpu as bench
import test_parallel_bug_integrator as helper


def difference(value, reference):
    value, reference = value.flatten().double(), reference.flatten().double()
    return {"relative_l2": float((value - reference).norm() / reference.norm().clamp_min(1e-30)),
            "cosine": float(torch.dot(value, reference) /
                            (value.norm() * reference.norm()).clamp_min(1e-30)),
            "bitwise_equal": torch.equal(value, reference)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path,
                        default=ROOT / "outputs/dlr_execution_2026_09_28")
    args = parser.parse_args()
    torch.set_num_threads(2)
    directory = args.directory
    names = ("idle_native", "idle_optimized", "idle_dflow_uncheckpointed",
             "idle_dflow_checkpointed", "idle_optimized_tuned", "idle_optimized_chunks3_2")
    result = {"scope": "sample0, 250x250 non-attention epoch100; 100 Euler steps; no optimizer/Armijo",
              "timing": {}, "accuracy": {}}
    outputs = {}
    for name in names:
        raw = json.loads((directory / f"{name}.json").read_text())
        if raw["status"] != "complete":
            raise ValueError(f"Incomplete run: {name}")
        assert raw["protocol"]["steps"] == 100
        assert raw["protocol"]["source_iterate"] == 20
        result["timing"][name] = {
            "median_seconds": raw["median_seconds"],
            "samples_seconds": [row["seconds"] for row in raw["calls"] if not row["warmup"]],
            "cold_seconds": raw["calls"][0]["seconds"],
            "peak_reserved_MiB": max(row["peak_reserved_MiB"] for row in raw["calls"]),
        }
        outputs[name] = torch.load(directory / f"{name}.pt", weights_only=True, map_location="cpu")
    data = torch.load(ROOT / "data/circles-250-n1.pt", weights_only=True, map_location="cpu", mmap=True)["test"]
    truth = bench.normalize_images(data[:1])
    angles = torch.linspace(0, torch.pi, 16)[:-1]
    _, target = helper.make_problem(250, 0, data, angles, torch.device("cpu"))

    def metrics(image):
        image = image.float()
        clipped = image.clamp(0, 1)
        _, residual_sq = bench.data_objective(bench.sparse_radon(image, angles), target, 250, "relative_scaled")
        return {"image_relative_l2": float((clipped - truth).norm() / truth.norm()),
                "raw_image_relative_l2": float((image - truth).norm() / truth.norm()),
                "psnr_db_clipped_range_1": float(-10 * torch.log10((clipped - truth).square().mean())),
                "measurement_relative_residual": float(residual_sq.sqrt())}

    def compare(output, baseline, exact):
        return {"image_vs_old_dlr": difference(output["image"], baseline["image"]),
                "gradient_vs_old_dlr": difference(output["source_gradient"], baseline["source_gradient"]),
                "gradient_vs_dflow": difference(output["source_gradient"], exact["source_gradient"]),
                "image_metrics_at_fixed_source": metrics(output["image"])}

    baseline, exact = outputs["idle_native"], outputs["idle_dflow_uncheckpointed"]
    result["accuracy"]["source20"] = {name: compare(output, baseline, exact)
        for name, output in outputs.items()}
    interleaved = json.loads((directory / "idle_matched_all.json").read_text())
    if interleaved["status"] != "complete":
        raise ValueError("Incomplete interleaved confirmation")
    result["interleaved_timing"] = {
        "medians_seconds": interleaved["medians"],
        "calls": [{key: row[key] for key in ("method", "warmup", "seconds")}
                  for row in interleaved["calls"]],
        "memory_note": "Use isolated runs for per-method memory; this process shares allocator pools.",
    }
    matched_outputs = torch.load(directory / "idle_matched_all.pt", weights_only=True, map_location="cpu")
    result["accuracy"]["interleaved_source20"] = {
        name: compare(output, baseline, matched_outputs["dflow-uncheckpointed"])
        for name, output in matched_outputs.items()}
    initial = torch.load(directory / "matched_initial.pt", weights_only=True, map_location="cpu")
    initial_baseline = torch.load(directory / "native_initial.pt", weights_only=True, map_location="cpu")
    result["accuracy"]["source0"] = {
        "old_dlr": compare(initial_baseline, initial_baseline, initial["dflow-checkpointed"]),
        "optimized_dlr": compare(initial["dlr"], initial_baseline, initial["dflow-checkpointed"]),
        "dflow": {"image_metrics_at_fixed_source": metrics(initial["dflow-checkpointed"]["image"])},
    }
    native_time = result["timing"]["idle_native"]["median_seconds"]
    optimized_time = result["timing"]["idle_optimized"]["median_seconds"]
    result["speedup_vs_previous_dlr"] = native_time / optimized_time
    result["optimized_over_uncheckpointed_time"] = optimized_time / result["timing"]["idle_dflow_uncheckpointed"]["median_seconds"]
    destination = directory / "summary.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"timing": result["timing"],
                      "interleaved_medians_seconds": result["interleaved_timing"]["medians_seconds"],
                      "speedup_vs_previous_dlr": result["speedup_vs_previous_dlr"],
                      "optimized_over_uncheckpointed_time": result["optimized_over_uncheckpointed_time"],
                      "summary": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
