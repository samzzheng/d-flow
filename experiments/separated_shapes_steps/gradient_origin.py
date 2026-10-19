"""Diagnostic controls separating DLR coefficients, subspace and basis reuse."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import torch

from experiments.ct_mri_sweep import run
from experiments.ct_mri_sweep.queue import source_hashes
from sampler.ode_samplers import euler_flow


class DiagnosticEvaluator(run.Evaluator):
    def gradient(self, source: torch.Tensor, method: str) -> torch.Tensor:
        if self.source is None or not torch.equal(self.source, source.detach()):
            self.objective(source.detach())
        self.counts["gradients"] += 1
        _, terminal = run.endpoint_gradient(self.image, self.operator, self.observed)
        _, factors, alpha = run.sensitivity_factors(
            self.model, source.detach(), self.settings["flow_steps"],
            **self.settings["dlr"], seed=self.seed, products=self.products,
            seed_basis=self.previous_v)
        assert alpha == 0.0
        estimate = run.approx_source_gradient(factors, terminal, alpha)
        right = factors[2]
        variable = source.detach().requires_grad_(True)
        image = euler_flow(self.model, variable, self.settings["flow_steps"])
        loss = .5 * (self.operator(image) - self.observed).abs().square().sum(dtype=torch.float64)
        exact, = torch.autograd.grad(loss, variable)
        projected = (right @ (right.T @ exact.reshape(-1, 1))).reshape_as(exact)
        norm = float(exact.norm())
        error = float((estimate - exact).norm())
        coefficient = float((estimate - projected).norm())
        subspace = float((projected - exact).norm())
        row = dict(evaluation=self.counts["gradients"], objective=self.value,
                   exact_norm=norm, dlr_norm=float(estimate.norm()),
                   projected_norm=float(projected.norm()),
                   relative_error=error / norm if norm else None,
                   relative_coefficient_error=coefficient / norm if norm else None,
                   relative_subspace_error=subspace / norm if norm else None,
                   cosine=float(torch.nn.functional.cosine_similarity(
                       estimate.flatten(), exact.flatten(), dim=0)) if norm and estimate.norm() else None,
                   decomposition_relative_error=abs(error**2-coefficient**2-subspace**2)/norm**2 if norm else None,
                   projected_descent_dot=float((exact * projected).sum()),
                   source_sha256=hashlib.sha256(source.detach().cpu().numpy().tobytes()).hexdigest())
        with Path(self.settings["gradient_diagnostic_path"]).open("a") as output:
            output.write(json.dumps(row) + "\n")
        if self.settings["dlr_reuse_previous_v"]:
            self.previous_v = right.detach().clone()
        gradient = projected if self.settings["gradient_control"] == "projected_exact" else estimate
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("Nonfinite diagnostic gradient")
        return gradient.detach()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=("baseline", "fresh_basis", "projected_exact"))
    parser.add_argument("--device", required=True, choices=("cuda:1",))
    args = parser.parse_args()
    root = Path("outputs/separated_shapes_steps_20261019")
    out = root / f"gradient_origin_{args.variant}"
    path = out / "production/manifest.json"
    if not path.exists():
        manifest = deepcopy(json.loads((root / "tiny_model_steps_5_rank_32/production/manifest.json").read_text()))
        manifest["created_at"] = datetime.now().astimezone().isoformat()
        settings = manifest["settings"]
        settings["dlr_reuse_previous_v"] = args.variant != "fresh_basis"
        settings["gradient_control"] = args.variant
        settings["gradient_diagnostic_path"] = str((out / "gradients.jsonl").resolve())
        settings["baseline_out"] = str((root / "tiny_model_steps_5_rank_32").resolve())
        manifest["source_sha256"] = source_hashes()
        manifest["source_sha256"][str(Path(__file__).relative_to(run.ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        manifest["protocol"]["methods"] = ["DLR D-Flow"]
        manifest["protocol"]["purpose"] = f"Diagnostic {args.variant}: same small model, five Euler steps, rank 32 and L-BFGS; compute exact gradients for analysis only"
        manifest["protocol"]["selection"] = "64 MRI sample 1, chosen because exact D-Flow succeeds and DLR fails; not held-out validation"
        manifest["protocol"]["warning"] = "projected_exact replaces DLR coefficients with the exact gradient projected onto DLR V; this is a diagnostic surrogate, not a speed candidate"
        run.write_json(path, manifest)
    original_plot = run.plot_reconstruction_comparison

    def diagnostic_plot(*plot_args: object, **plot_kwargs: object) -> object:
        plot_kwargs["title"] = f"Diagnostic {args.variant} (DLR column)\n" + str(plot_kwargs["title"])
        return original_plot(*plot_args, **plot_kwargs)

    run.plot_reconstruction_comparison = diagnostic_plot
    run.Evaluator = DiagnosticEvaluator
    run.METHODS = ("DLR D-Flow",)
    sys.argv = [sys.argv[0], "--manifest", str(path), "--job", "r64_shapes_mri4",
                "--sample-start", "1", "--sample-stop", "2", "--device", args.device, "--memory-fraction", ".10"]
    run.main()


if __name__ == "__main__":
    main()
