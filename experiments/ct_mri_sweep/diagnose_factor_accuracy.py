"""Same-source CPU DLR accuracy and fixed-cost initialization ablations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import torch

from dlr_dflow.api import approx_source_gradient, load_flow_model, sensitivity_factors
from dlr_dflow.implementation import enriched
from p_flow import endpoint_gradient
from sampler.ode_samplers import euler_flow
from .operators import CT, MRI


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    enriched.SPECIALIZE_ZERO_ENRICHMENT = True
    results: list[dict[str, Any]] = []
    for specification in args.cases:
        path = args.root / (specification + ".pt")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        inputs = torch.load(path.parent / "inputs.pt", map_location="cpu", weights_only=True)
        protocol = saved["protocol"]
        job = protocol["job"]
        resolution = job["resolution"]
        model, _ = load_flow_model(resolution, job["circles"], 200, "cpu",
                                  Path(f"outputs/fm_unet/circles-{resolution}-noattn-ch16-b8-e200/n{job['circles']}"))
        operator = (CT(resolution, job["level"], torch.device("cpu")) if job["problem"] == "ct" else
                    MRI(resolution, job["level"], protocol["operator"]["mask_seed"], torch.device("cpu")))
        source = saved["source"].detach().requires_grad_(True)
        image = euler_flow(model, source, 100)
        loss = .5 * (operator(image) - inputs["observed"]).abs().square().sum(dtype=torch.float64)
        exact, = torch.autograd.grad(loss, source)
        _, gx = endpoint_gradient(image.detach(), operator, inputs["observed"])
        source = source.detach()
        result: dict[str, Any] = {"case": specification,
                                  "endpoint_relative_drift_cpu_vs_gpu": float((image.detach()-saved["image"]).norm()/saved["image"].norm()),
                                  "variants": []}
        variants = (("rank8_seed0", 8, 0, None), ("rank8_seed1", 8, 1, None),
                    ("rank8_saved_gradient_seed", 8, 0, saved["optimizer"]["g"].reshape(-1, 1)),
                    ("rank16_seed0", 16, 0, None))
        for name, rank, shift, seed_basis in variants:
            started = time.monotonic()
            endpoint, factors, alpha = sensitivity_factors(
                model, source, 100, rank=rank, trace_probes=0, enrichment_probes=0,
                initialization_oversampling=8, factor_stride=1, zero_terminal_baseline=True,
                seed=protocol["dlr_seed"] + shift, seed_basis=seed_basis)
            assert alpha == 0
            gradient = approx_source_gradient(factors, gx, alpha).flatten()
            U, S, V = factors
            true = exact.flatten()
            projected = V @ (V.T @ true)
            total = (gradient - true).norm()
            coefficient = (gradient - projected).norm()
            subspace = (projected - true).norm()
            decomposition = float((total.square() - coefficient.square() - subspace.square()).abs() / true.square().sum())
            assert decomposition < 1e-4
            entry = {"name": name, "rank": rank, "seed": protocol["dlr_seed"] + shift,
                     "relative_gradient_error": float(total / true.norm()),
                     "subspace_error_over_exact_norm": float(subspace / true.norm()),
                     "coefficient_error_over_exact_norm": float(coefficient / true.norm()),
                     "cosine": float(torch.nn.functional.cosine_similarity(gradient, true, dim=0)),
                     "gradient_norm_ratio": float(gradient.norm() / true.norm()),
                     "negative_dlr_true_slope": -float(true @ gradient),
                     "orthogonal_decomposition_error": decomposition,
                     "dlr_endpoint_relative_error": float((endpoint - image.detach()).norm() / image.detach().norm()),
                     "seconds": time.monotonic() - started}
            result["variants"].append(entry)
            print(json.dumps({"case": specification, **entry}), flush=True)
            (args.output / "factor_diagnostics.json").write_text(json.dumps(results + [result], indent=2) + "\n")
        results.append(result)
        del image, loss, exact


if __name__ == "__main__":
    main()
