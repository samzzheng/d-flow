"""Bounded GPU replay of stalled directions; no changes to the running sweep."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import torch

from dlr_dflow.api import approx_source_gradient, load_flow_model, sensitivity_factors
from dlr_dflow.implementation import enriched, tangent_unet
from dlr_dflow.implementation.graphed_products import GraphedAProducts
from p_flow import endpoint_gradient
from sampler.ode_samplers import euler_flow
from .operators import CT, MRI
from .run import metrics, save_tensor, write_json

Record = dict[str, Any]


def reverse_euler_gradient(model: torch.nn.Module, source: torch.Tensor,
                           operator: CT | MRI, observed: torch.Tensor,
                           steps: int = 100) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact discrete adjoint with saved states and one-step recomputation."""
    states = [source.detach()]
    with torch.no_grad():
        for index in range(steps):
            t = torch.full((source.shape[0],), index / steps, device=source.device, dtype=source.dtype)
            states.append(states[-1] + (1. / steps) * model(states[-1], t))
    _, gx = endpoint_gradient(states[-1], operator, observed)
    gradient = gx
    for index in reversed(range(steps)):
        variable = states[index].detach().requires_grad_(True)
        t = torch.full((source.shape[0],), index / steps, device=source.device, dtype=source.dtype)
        updated = variable + (1. / steps) * model(variable, t)
        gradient, = torch.autograd.grad(updated, variable, gradient)
    return states[-1], gx, gradient.detach()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/ct_mri_sweep_geometric_v_20261008"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--tf32", choices=("on", "off"), default="on")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--factor-tests", action="store_true")
    parser.add_argument("--directions", nargs="+")
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    assert device == torch.device("cuda:1"), "GPU 0 remains excluded"
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.06, device)
    torch.backends.cudnn.allow_tf32 = args.tf32 == "on"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    enriched.SPECIALIZE_ZERO_ENRICHMENT = True
    enriched.SKIP_UNTRUNCATED_SVD = True
    tangent_unet.GN_STATS_MODE = "manual"
    tangent_unet.GN_ACTION_BACKEND = "triton"
    tangent_unet.GN_PRIMAL_BACKEND = "full"
    destination = args.output / args.case / f"tf32_{args.tf32}"
    destination.mkdir(parents=True, exist_ok=True)
    saved = torch.load(args.root / args.case / "dlr_d_flow.pt", map_location="cpu", weights_only=True)
    inputs = torch.load(args.root / args.case / "inputs.pt", map_location="cpu", weights_only=True)
    protocol = saved["protocol"]
    job = protocol["job"]
    settings = json.loads((args.root / "manifest.json").read_text())["settings"]
    model_directory = Path(settings["run_template"].format(**job))
    checkpoint = model_directory / f"checkpoints/epoch_{settings['epoch']:04d}.pt"
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == protocol["checkpoint_sha256"]
    model, _ = load_flow_model(job["resolution"], job["circles"], settings["epoch"], str(device), model_directory)
    operator = (CT(job["resolution"], job["level"], device) if job["problem"] == "ct" else
                MRI(job["resolution"], job["level"], protocol["operator"]["mask_seed"], device))
    observed = inputs["observed"].to(device)
    source = saved["source"].to(device)
    image, gx, exact = reverse_euler_gradient(model, source, operator, observed)

    def objective(value: torch.Tensor) -> float:
        with torch.no_grad():
            prediction = operator(euler_flow(model, value, 100))
            return float(.5 * (prediction - observed).abs().square().sum(dtype=torch.float64))

    value = float(.5 * (operator(image) - observed).abs().square().sum(dtype=torch.float64))
    repeated = objective(source)
    gradients = {"saved_dlr": saved["optimizer"]["g"].to(device).reshape_as(source), "exact": exact, "proxy": gx}
    result: Record = {"case": args.case, "tf32": args.tf32, "saved_phi": saved["history"][-1]["phi"],
                      "checkpoint_sha256": protocol["checkpoint_sha256"],
                      "source_checkpoint_sha256": hashlib.sha256((args.root / args.case / "dlr_d_flow.pt").read_bytes()).hexdigest(),
                      "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      "phi": value, "repeated_phi": repeated,
                      "saved_image_error_pct": 100 * saved["history"][-1]["image_relative_l2"],
                      "endpoint_drift_from_saved": float((image.cpu() - saved["image"]).norm() / saved["image"].norm()),
                      "current_metrics": metrics(image, operator(image), inputs["truth"].to(device), inputs["clean"].to(device), observed),
                      "directions": {}}
    print(json.dumps({key: item for key, item in result.items() if key != "directions"}), flush=True)
    write_json(destination / "result.json", result)
    if args.factor_tests:
        products = GraphedAProducts(model, source, share_linearization=True, full_step_graphs=True,
                                   max_jvp_columns=20, max_vjp_columns=16,
                                   max_step_jvp_columns=6, max_step_vjp_columns=4)
        factor_endpoint, factors, alpha = sensitivity_factors(model, source, 100, **settings["dlr"], seed=protocol["dlr_seed"],
                                               seed_basis=saved["previous_v"].to(device), products=products)
        assert alpha == 0
        result["factor_endpoint_relative_difference"] = float((factor_endpoint - image).norm() / image.norm())
        gradients["refreshed_v8"] = approx_source_gradient(factors, gx, alpha)
        core_u, singular, core_vh = torch.linalg.svd(factors[1], full_matrices=False)
        # Diagnostic preconditioning, not an exact-Jacobian gradient claim.
        clipped = singular.clamp(max=singular.median())
        gradients["spectral_cap"] = (factors[2] @ core_vh.T @ (clipped[:, None] * (core_u.T @ factors[0].T @ gx.reshape(-1, 1)))).reshape_as(source)
        result["core_singular_values"] = singular.cpu().tolist()
        result["spectral_cap_rule"] = "clip singular values to their median; changes the gradient weighting"
    cap = .01 * math.sqrt(source.numel())
    for name, gradient in gradients.items():
        if args.directions and name not in args.directions:
            continue
        direction = -gradient
        norm = float(direction.norm())
        rate = min(1., cap / norm) if norm else 0.
        approximate_slope = -float(gradient.square().sum())
        true_slope = float((exact * direction).sum())
        entry: Record = {"norm": norm, "cosine_exact": float(torch.nn.functional.cosine_similarity(gradient.flatten(), exact.flatten(), dim=0)),
                         "relative_error_exact": float((gradient - exact).norm() / exact.norm()),
                         "true_slope": true_slope, "trials": [], "first_armijo_trial": None}
        for trial in range(1, 61):
            candidate = source + rate * direction
            changed = int((candidate != source).count_nonzero())
            if changed == 0:
                entry["unchanged_at_trial"] = trial
                break
            proposed = objective(candidate)
            armijo = proposed < value and proposed - value <= 1e-4 * rate * approximate_slope
            entry["trials"].append({"trial": trial, "rate": rate, "changed_coordinates": changed,
                                    "displacement": float((candidate - source).norm()), "phi": proposed,
                                    "decrease": value - proposed, "armijo": armijo})
            if armijo:
                entry["first_armijo_trial"] = trial
                with torch.no_grad():
                    candidate_image = euler_flow(model, candidate, 100)
                    entry["accepted_metrics"] = metrics(candidate_image, operator(candidate_image), inputs["truth"].to(device),
                                                        inputs["clean"].to(device), observed)
                break
            rate *= .5
        result["directions"][name] = entry
        write_json(destination / "result.json", result)
        print(json.dumps({"case": args.case, "tf32": args.tf32, "direction": name,
                          **{key: item for key, item in entry.items() if key != "trials"}}), flush=True)
    result["seconds"] = time.monotonic() - started
    result["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
    result["complete"] = True
    save_tensor(destination / "gradients.pt", {"source": source, "image": image, "gradients": gradients})
    write_json(destination / "result.json", result)


if __name__ == "__main__":
    main()
