"""Test residual/noise geometry at all30 saved64px DLR endpoints per problem."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any

import torch

from dlr_dflow.api import load_flow_model, make_terminal_flow_products, sensitivity_factors
from dlr_dflow.implementation import enriched
from experiments.ct_mri_sweep.operators import CT, MRI
from experiments.ct_mri_sweep.run import write_json
from sampler.ode_samplers import euler_flow


def real_measurement(value: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if value.is_complex():
        value = value[mask] if mask is not None else value.flatten()
        return torch.view_as_real(value).flatten().double()
    return value.flatten().double()


def main() -> None:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    enriched.SPECIALIZE_ZERO_ENRICHMENT = True
    root = Path("outputs/five_step_noise020_n124_20261013/production")
    output = Path("outputs/noise_regularization_analysis_20261014/flow_time/subspace_residual.json")
    settings = json.loads((root / "manifest.json").read_text())["settings"]
    model_path = Path("outputs/fm_unet/circles-64-noattn-ch16-b8-e200/n1")
    model, _ = load_flow_model(64, 1, 200, "cpu", model_path)
    checkpoint_hash = hashlib.sha256((model_path / "checkpoints/epoch_0200.pt").read_bytes()).hexdigest()
    rows: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "device": "cpu", "threads": 2, "checkpoint_sha256": checkpoint_hash,
        "noise_percent": 20, "resolution": 64, "circles": 1, "flow_steps": 5,
        "limitations": ["New factor evolution seeded by saved terminal V, not original last-gradient replay",
                        "Effective observation-subspace rank uses singular values above1e-6 of maximum",
                        "Projection formula is a local restricted-model diagnostic, not a theorem for rotating-basis LBFGS",
                        "All30 samples/problem are included; image<=10% grouping is retrospective"],
        "rows": rows,
    }
    for problem, suffix in (("ct", "ct10"), ("mri", "mri4")):
        for sample in range(30):
            directory = root / f"r64_n1_{suffix}" / f"sample_{sample:03d}"
            inputs = torch.load(directory / "inputs.pt", map_location="cpu", weights_only=True, mmap=True)
            saved = torch.load(directory / "dlr_d_flow.pt", map_location="cpu", weights_only=True, mmap=True)
            assert inputs["protocol"]["checkpoint_sha256"] == checkpoint_hash
            operator: CT | MRI = CT(64, 10, torch.device("cpu")) if problem == "ct" else MRI(
                64, 4, inputs["protocol"]["operator"]["mask_seed"], torch.device("cpu"))
            source = saved["source"]
            _, (u, _, v), alpha = sensitivity_factors(model, source, 5, **settings["dlr"],
                seed=inputs["protocol"]["dlr_seed"], seed_basis=saved["previous_v"])
            assert alpha == 0
            j, _ = make_terminal_flow_products(model, source, 5)
            noise = real_measurement(inputs["observed"] - inputs["clean"], inputs["mask"])
            endpoint = euler_flow(model, source, 5).detach()
            residual = real_measurement(operator(endpoint) - inputs["observed"], inputs["mask"])
            image_error = 100 * float((endpoint - inputs["truth"]).norm() / inputs["truth"].norm())
            for label, image_directions in (("DLR_left_U", u), ("actual_JV", j(v))):
                observed_directions = torch.stack([real_measurement(operator(column.reshape_as(source)), inputs["mask"])
                                                   for column in image_directions.T], dim=1)
                left, singular, _ = torch.linalg.svd(observed_directions, full_matrices=False)
                rank = int((singular > 1e-6 * singular[0]).sum())
                q = left[:, :rank]
                noise_projection = q @ (q.T @ noise)
                ideal_residual = -(noise - noise_projection)
                rows.append({
                    "problem": problem, "sample": sample, "space": label,
                    "image_error_percent": image_error, "effective_observed_rank": rank,
                    "real_measurement_dimension": noise.numel(),
                    "predicted_isotropic_residual_rms_ratio": (1 - rank / noise.numel()) ** .5,
                    "realized_ideal_residual_over_noise": float(ideal_residual.norm() / noise.norm()),
                    "actual_residual_over_noise": float(residual.norm() / noise.norm()),
                    "residual_fraction_in_fitted_space": float((q.T @ residual).norm() / residual.norm()),
                    "actual_vs_ideal_residual_cosine": float(residual @ ideal_residual / (residual.norm() * ideal_residual.norm())),
                    "actual_vs_ideal_residual_difference_over_noise": float((residual - ideal_residual).norm() / noise.norm()),
                    "realized_noise_energy_in_fitted_space": float(noise_projection.square().sum() / noise.square().sum()),
                    "observed_direction_singular_values": singular.tolist(),
                })
            write_json(output, result)
            print(json.dumps({"problem": problem, "sample": sample, "image_error": image_error}), flush=True)
    summaries = []
    for problem in ("ct", "mri"):
        for space in ("DLR_left_U", "actual_JV"):
            for selection in ("all", "image_error_at_most10_percent", "image_error_above10_percent"):
                selected = [row for row in rows if row["problem"] == problem and row["space"] == space and
                            (selection == "all" or (row["image_error_percent"] <= 10) == (selection == "image_error_at_most10_percent"))]
                summaries.append({"problem": problem, "space": space, "selection": selection, "samples": len(selected),
                                  **{"median_" + key: median(row[key] for row in selected) if selected else None
                                     for key in ("effective_observed_rank", "predicted_isotropic_residual_rms_ratio",
                                        "realized_ideal_residual_over_noise", "actual_residual_over_noise",
                                        "residual_fraction_in_fitted_space", "actual_vs_ideal_residual_cosine",
                                        "actual_vs_ideal_residual_difference_over_noise", "realized_noise_energy_in_fitted_space")}})
    result["summaries"] = summaries
    write_json(output, result)


if __name__ == "__main__":
    main()
