"""Measure fitting of the known noise using saved 64px terminal measurements."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median

import torch


def main() -> None:
    torch.set_num_threads(1)
    destination = Path("outputs/noise_regularization_analysis_20261014/noise_alignment")
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int | float]] = []
    for noise in (1, 5, 10, 20):
        root = Path(f"outputs/five_step_noise{noise:03d}_n124_20261013/production")
        for problem in ("ct10", "mri4"):
            for sample in range(30):
                case = root / f"r64_n1_{problem}" / f"sample_{sample:03d}"
                inputs = torch.load(case / "inputs.pt", map_location="cpu", weights_only=False, mmap=True)
                dtype = torch.complex128 if inputs["clean"].is_complex() else torch.float64
                clean = inputs["clean"].flatten().to(dtype)
                epsilon = inputs["observed"].flatten().to(dtype) - clean
                noise_squared = torch.vdot(epsilon, epsilon).real
                for method in ("d_flow", "p_flow", "dlr_d_flow"):
                    saved = torch.load(case / f"{method}.pt", map_location="cpu", weights_only=False, mmap=True)
                    error = saved["measurement"].flatten().to(dtype) - clean
                    coefficient = torch.vdot(epsilon, error).real / noise_squared
                    perpendicular = error - coefficient * epsilon
                    residual = error - epsilon
                    decomposed = (coefficient - 1).square() + perpendicular.abs().square().sum() / noise_squared
                    residual_squared = residual.abs().square().sum() / noise_squared
                    identity_error = float(abs(decomposed - residual_squared) / residual_squared.clamp_min(1e-30))
                    assert identity_error < 1e-10, identity_error
                    image = saved["image"].double()
                    truth = inputs["truth"].double()
                    rows.append({
                        "noise_percent": noise, "resolution": 64, "problem": problem,
                        "sample": sample, "method": method,
                        "image_error_percent": 100 * float((image - truth).norm() / truth.norm()),
                        "noise_fit_coefficient": float(coefficient),
                        "perpendicular_error_over_noise": float(perpendicular.norm() / epsilon.norm()),
                        "residual_over_noise": float(residual.norm() / epsilon.norm()),
                        "decomposition_relative_error": identity_error,
                    })
    with (destination / "runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries = []
    for noise in (1, 5, 10, 20):
        for problem in ("ct10", "mri4"):
            for method in ("d_flow", "p_flow", "dlr_d_flow"):
                group = [row for row in rows if (row["noise_percent"], row["problem"], row["method"]) == (noise, problem, method)]
                summaries.append({"noise_percent": noise, "problem": problem, "method": method,
                                  "samples": len(group), **{
                                      "median_" + key: median(float(row[key]) for row in group)
                                      for key in ("noise_fit_coefficient", "perpendicular_error_over_noise", "residual_over_noise", "image_error_percent")}})
    result = {
        "definition": "c=Re<epsilon,A(x_hat)-A(x_GT)>/||epsilon||^2; c=1 fits the noise component; c=0 fits none of that component.",
        "scope": "All 30 samples at 64px, single circles, CT10 and MRI4, each of 1/5/10/20% noise; saved terminal measurements only. No clipping or new model evaluations.",
        "max_decomposition_relative_error": max(float(row["decomposition_relative_error"]) for row in rows),
        "groups": summaries,
    }
    (destination / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
