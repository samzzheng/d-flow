"""CPU precision and directional diagnostics at frozen GPU DLR sources.

CPU replays are different numerical maps from CUDA. Report their endpoint
drift explicitly; this experiment cannot certify a CUDA line-search failure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import torch

from dlr_dflow.api import load_flow_model
from sampler.ode_samplers import euler_flow
from .operators import CT, MRI


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/ct_mri_sweep_geometric_v_20261008"))
    parser.add_argument("--output", type=Path, default=Path("outputs/dlr_failure_tests_20261008/precision"))
    parser.add_argument("--cases", nargs="+", default=[
        "r128_n3_ct10/sample_008", "r128_n2_mri4/sample_003"])
    parser.add_argument("--gradient-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    args.output.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for specification in args.cases:
        started = time.monotonic()
        case_path = args.root / specification
        saved = torch.load(case_path / "dlr_d_flow.pt", map_location="cpu", weights_only=True)
        inputs = torch.load(case_path / "inputs.pt", map_location="cpu", weights_only=True)
        protocol = saved["protocol"]
        job = protocol["job"]
        resolution = job["resolution"]
        model_path = Path(f"outputs/fm_unet/circles-{resolution}-noattn-ch16-b8-e200/n{job['circles']}")
        model, _ = load_flow_model(resolution, job["circles"], 200, "cpu", model_path)
        operator = (CT(resolution, job["level"], torch.device("cpu")) if job["problem"] == "ct" else
                    MRI(resolution, job["level"], protocol["operator"]["mask_seed"], torch.device("cpu")))
        if isinstance(operator, MRI):
            assert torch.equal(operator.mask, inputs["mask"])
        result: dict[str, Any] = {
            "case": specification, "saved_status": saved["status"],
            "saved_image_error_pct": 100 * saved["history"][-1]["image_relative_l2"],
            "saved_phi": saved["history"][-1]["phi"],
            "displacement_cap": .01 * resolution,
            "interpretation": "CPU shadow maps; endpoint drift precludes treating these as CUDA replays.",
            "precisions": [],
        }
        previous_exact: torch.Tensor | None = None
        previous_image: torch.Tensor | None = None
        for dtype in (torch.float32, torch.float64):
            model.to(dtype=dtype)

            def cast_embedding(module: torch.nn.Module, arguments: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
                return (arguments[0].to(dtype=dtype),)

            # Timestep embedding intentionally retains the model's float32
            # sinusoid arithmetic; only its linear-layer input is promoted.
            hook = model.time_proj.register_forward_pre_hook(cast_embedding)
            if isinstance(operator, CT):
                operator.grids = tuple(grid.to(dtype=dtype) for grid in operator.grids)
            observed = inputs["observed"].to(
                dtype=(torch.complex128 if dtype == torch.float64 else torch.complex64)
                if inputs["observed"].is_complex() else dtype)
            source = saved["source"].to(dtype=dtype).detach().requires_grad_(True)
            image = euler_flow(model, source, 100)
            loss = .5 * (operator(image) - observed).abs().square().sum(dtype=torch.float64)
            exact, = torch.autograd.grad(loss, source)
            stored = saved["optimizer"]["g"].to(dtype=dtype).reshape_as(source)
            value = float(loss.detach())
            entry: dict[str, Any] = {
                "dtype": str(dtype), "phi": value,
                "endpoint_relative_drift_from_gpu": float((image.detach() - saved["image"]).norm() / saved["image"].norm()),
                "exact_gradient_norm": float(exact.norm()),
                "saved_gradient_norm": float(stored.norm()),
                "saved_vs_cpu_exact_cosine": float(torch.nn.functional.cosine_similarity(stored.flatten(), exact.flatten(), dim=0)),
                "saved_vs_cpu_exact_relative_error": float((stored - exact).norm() / exact.norm()),
                "directions": {},
            }
            if previous_exact is not None and previous_image is not None:
                entry["cpu32_vs_cpu64_gradient_relative_difference"] = float((previous_exact - exact).norm() / exact.norm())
                entry["cpu32_vs_cpu64_gradient_cosine"] = float(torch.nn.functional.cosine_similarity(previous_exact.to(dtype=dtype).flatten(), exact.flatten(), dim=0))
                entry["cpu32_vs_cpu64_endpoint_relative_difference"] = float((previous_image - image.detach()).norm() / image.detach().norm())
            print(json.dumps({"case": specification, "stage": "exact_gradient", **entry}), flush=True)
            if dtype == torch.float64 and not args.gradient_only:
                for name, gradient in (("saved_dlr", stored), ("exact", exact)):
                    direction = -gradient / gradient.norm()
                    slope = float((exact * direction).sum())
                    trials: list[dict[str, Any]] = []
                    for fraction in (1., 1e-3, 1e-5, 1e-7, 1e-9):
                        displacement = fraction * .01 * resolution
                        with torch.no_grad():
                            candidate = source + displacement * direction
                            trial_image = euler_flow(model, candidate, 100)
                            trial_value = float(.5 * (operator(trial_image) - observed).abs().square().sum(dtype=torch.float64))
                        change = trial_value - value
                        trial = {
                            "displacement": displacement,
                            "actual_displacement": float((candidate - source.detach()).norm()),
                            "changed_source_coordinates_float64": int(torch.count_nonzero(candidate != source)),
                            "changed_source_coordinates_if_float32": int(torch.count_nonzero(candidate.float() != source.float())),
                            "objective_change": change,
                            "linear_prediction": displacement * slope,
                            "change_over_linear_prediction": change / (displacement * slope) if slope != 0 else None,
                            "armijo_with_cpu_true_slope": slope < 0 and change < 0 and change <= 1e-4 * displacement * slope,
                        }
                        trials.append(trial)
                        print(json.dumps({"case": specification, "direction": name, **trial}), flush=True)
                    entry["directions"][name] = {"true_directional_derivative": slope, "trials": trials}
            previous_exact, previous_image = exact.detach(), image.detach()
            result["precisions"].append(entry)
            hook.remove()
            del image, loss, source, exact
        result["seconds"] = time.monotonic() - started
        results.append(result)
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps({"case": specification, "stage": "complete", "seconds": result["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
