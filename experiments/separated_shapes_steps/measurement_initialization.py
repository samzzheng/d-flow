"""Five-step reconstruction with a measurement-informed, paired source."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

from experiments.ct_mri_sweep import run
from experiments.ct_mri_sweep.queue import source_hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mix", type=float, required=True, choices=(.25, .5))
    parser.add_argument("--model", choices=("small", "original"), default="small")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--problem", choices=("ct10", "mri4"), default="mri4")
    parser.add_argument("--sample", type=int, default=1)
    parser.add_argument("--device", required=True, choices=("cuda:1",))
    args = parser.parse_args()
    if args.model == "small" and args.resolution != 64:
        parser.error("The small separated-shapes model is trained only at resolution 64")
    root = Path("outputs/separated_shapes_steps_20261019")
    out = root / f"measurement_init_{args.model}_{args.mix:g}"
    path = out / "production/manifest.json"
    if not path.exists():
        template = "tiny_model_steps_5_rank_32" if args.model == "small" else "steps_10_rank_32"
        manifest = deepcopy(json.loads((root / template / "production/manifest.json").read_text()))
        manifest["created_at"] = datetime.now().astimezone().isoformat()
        settings = manifest["settings"]
        settings["flow_steps"] = 5
        settings.pop("baseline_out", None)
        settings["initialization"] = dict(mix=args.mix, rule="sqrt(1-rho^2)*z + rho*centered_normalized_adjoint_y",
                                          uses_ground_truth=False)
        manifest["source_sha256"] = source_hashes()
        manifest["source_sha256"][str(Path(__file__).relative_to(run.ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        manifest["protocol"]["methods"] = list(run.METHODS)
        manifest["protocol"]["purpose"] = "Change source initialization only: one measurement adjoint, centered and normalized to sqrt(d), mixed with the original random source"
        manifest["protocol"]["selection"] = "Exploratory selected cases; same conditioned source across methods. This changes the initialization distribution, not the flow or objective."
        run.write_json(path, manifest)
    original_solve = run.solve

    def conditioned_solve(evaluator: run.Evaluator, method: str, initial: torch.Tensor,
                          truth: torch.Tensor, clean: torch.Tensor, checkpoint: Path,
                          protocol: run.Settings, interrupted: list[bool]) -> run.Settings:
        variable = torch.zeros_like(initial, requires_grad=True)
        pairing = (evaluator.operator(variable).conj() * evaluator.observed).real.sum()
        backprojection, = torch.autograd.grad(pairing, variable)
        centered = backprojection - backprojection.mean()
        norm = centered.norm()
        if not torch.isfinite(norm) or norm == 0:
            raise FloatingPointError("The centered measurement adjoint is zero or nonfinite")
        direction = centered * (math.sqrt(initial.numel()) / norm)
        conditioned = math.sqrt(1 - args.mix**2) * initial + args.mix * direction
        protocol = {**protocol, "initialization": evaluator.settings["initialization"],
                    "initial_source_sha256": hashlib.sha256(conditioned.detach().cpu().numpy().tobytes()).hexdigest()}
        inputs_path = checkpoint.parent / "inputs.pt"
        inputs = torch.load(inputs_path, weights_only=True, map_location="cpu")
        inputs.update(initial=conditioned.detach().cpu(), unconditioned_initial=initial.detach().cpu(),
                      initialization_direction=direction.detach().cpu(), protocol=protocol)
        run.save_tensor(inputs_path, inputs)
        return original_solve(evaluator, method, conditioned, truth, clean, checkpoint, protocol, interrupted)

    original_plot = run.plot_reconstruction_comparison

    def conditioned_plot(*plot_args: object, **plot_kwargs: object) -> object:
        plot_kwargs["title"] = f"Measurement-informed initialization, rho={args.mix:g}\n" + str(plot_kwargs["title"])
        return original_plot(*plot_args, **plot_kwargs)

    run.solve = conditioned_solve
    run.plot_reconstruction_comparison = conditioned_plot
    sys.argv = [sys.argv[0], "--manifest", str(path), "--job", f"r{args.resolution}_shapes_{args.problem}",
                "--sample-start", str(args.sample), "--sample-stop", str(args.sample+1),
                "--device", args.device, "--memory-fraction", ".10"]
    run.main()


if __name__ == "__main__":
    main()
