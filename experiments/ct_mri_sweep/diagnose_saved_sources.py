"""CPU shadow diagnostics at saved sources; never resume or change the GPU sweep.

CPU and CUDA trajectories need not coincide. Endpoint drift is reported, so
these checks do not certify the directional derivatives of the CUDA objective.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import torch

from dlr_dflow.api import load_flow_model
from p_flow import endpoint_gradient
from sampler.ode_samplers import euler_flow
from .operators import CT, MRI


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", required=True,
                        help="job/sample_XXX/method_stem, relative to root")
    parser.add_argument("--float64-only", action="store_true")
    parser.add_argument("--relative-displacements", nargs="+", type=float,
                        default=[.01, .0001, .000001], help="Source L2 displacement divided by resolution")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    args.output.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for specification in args.cases:
        started = time.monotonic()
        path = args.root / (specification + ".pt")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        inputs = torch.load(path.parent / "inputs.pt", map_location="cpu", weights_only=True)
        protocol = saved["protocol"]
        job = protocol["job"]
        resolution = job["resolution"]
        model_path = Path(f"outputs/fm_unet/circles-{resolution}-noattn-ch16-b8-e200/n{job['circles']}")
        model, _ = load_flow_model(resolution, job["circles"], 200, "cpu", model_path)
        operator = (CT(resolution, job["level"], torch.device("cpu")) if job["problem"] == "ct" else
                    MRI(resolution, job["level"], protocol["operator"]["mask_seed"], torch.device("cpu")))
        if isinstance(operator, MRI):
            assert torch.equal(operator.mask, inputs["mask"])
        result: dict[str, Any] = {"case": specification, "status": saved["status"],
                                  "gpu_image_pct": 100 * saved["history"][-1]["image_relative_l2"],
                                  "saved_phi": saved["history"][-1]["phi"],
                                  "saved_gradient_norm": float(saved["optimizer"]["g"].norm()),
                                  "saved_lbfgs_pair_count": len(saved["optimizer"]["S"]),
                                  "cpu": []}
        for dtype in ((torch.float64,) if args.float64_only else (torch.float32, torch.float64)):
            model.to(dtype=dtype)
            # The model's sinusoidal embedding is explicitly float32; cast at
            # its linear layer for this isolated float64 diagnostic only.
            hook = model.time_proj.register_forward_pre_hook(lambda module, arguments: (arguments[0].to(dtype=dtype),))
            if isinstance(operator, CT):
                operator.grids = tuple(grid.to(dtype=dtype) for grid in operator.grids)
            observed = inputs["observed"].to(dtype=torch.complex128 if inputs["observed"].is_complex() and dtype == torch.float64 else
                                             torch.complex64 if inputs["observed"].is_complex() else dtype)
            source = saved["source"].to(dtype=dtype).detach().requires_grad_(True)
            image = euler_flow(model, source, 100)
            loss = .5 * (operator(image) - observed).abs().square().sum(dtype=torch.float64)
            exact, = torch.autograd.grad(loss, source)
            _, proxy = endpoint_gradient(image.detach(), operator, observed)
            stored = saved["optimizer"]["g"].to(dtype=dtype).reshape_as(source)
            value = float(loss.detach())
            entry: dict[str, Any] = {"dtype": str(dtype), "phi": value,
                                     "endpoint_relative_drift_from_gpu": float((image.detach() - saved["image"]).norm() / saved["image"].norm()),
                                     "exact_gradient_norm": float(exact.norm()),
                                     "saved_vs_cpu_exact_cosine": float(torch.nn.functional.cosine_similarity(stored.flatten(), exact.flatten(), dim=0)),
                                     "saved_vs_cpu_exact_relative_error": float((stored-exact).norm()/exact.norm()),
                                     "directions": {}}
            # CPU32 checks endpoint/gradient sensitivity; CPU64 adds true
            # directional and finite displacement tests on its own objective.
            if dtype == torch.float64:
                for name, gradient in (("saved_gradient", stored), ("exact", exact), ("p_flow", proxy)):
                    direction = -gradient / gradient.norm()
                    slope = float((exact * direction).sum())
                    trials = []
                    for displacement in map(lambda fraction: fraction * resolution, args.relative_displacements):
                        with torch.no_grad():
                            trial_image = euler_flow(model, source + displacement * direction, 100)
                            trial_value = float(.5 * (operator(trial_image) - observed).abs().square().sum(dtype=torch.float64))
                        trials.append({"displacement": displacement, "objective_change": trial_value - value,
                                       "armijo_with_cpu_true_slope": trial_value < value and trial_value - value <= 1e-4 * displacement * slope})
                    entry["directions"][name] = {"true_directional_derivative": slope, "trials": trials}
            result["cpu"].append(entry)
            hook.remove()
            del image, loss, source, exact
        result["seconds"] = time.monotonic() - started
        results.append(result)
        (args.output / "shadow_diagnostics.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
