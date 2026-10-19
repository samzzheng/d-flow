"""Run a fine-grid exact-gradient control with unchanged reconstruction settings."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.utils.checkpoint import checkpoint

from experiments.ct_mri_sweep import run
from experiments.ct_mri_sweep.queue import source_hashes
from dlr_dflow.api import load_flow_model
from sampler.ode_samplers import euler_flow


def checkpointed_euler(model: torch.nn.Module, source: torch.Tensor, steps: int) -> torch.Tensor:
    image = source
    step_size = 1.0 / steps
    for index in range(steps):
        time = image.new_full((image.shape[0],), index * step_size)
        velocity = checkpoint(model, image, time, use_reentrant=False, preserve_rng_state=False)
        image = image + step_size * velocity
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=250)
    parser.add_argument("--problem", choices=("ct10", "mri4"), default="mri4")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = Path("outputs/separated_shapes_steps_20261019")
    if args.verify_only:
        torch.cuda.set_device(args.device)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model, _ = load_flow_model(128, 0, 200, args.device,
            Path("outputs/fm_unet/separated-shapes-128-noattn-ch16-b8-e200"))
        source = torch.randn(1, 1, 128, 128, generator=torch.Generator().manual_seed(20261019)).to(args.device).requires_grad_(True)
        results = []
        for integrate in (euler_flow, checkpointed_euler):
            image = integrate(model, source, 20)
            gradient, = torch.autograd.grad(image.square().sum(), source)
            results.append((image.detach(), gradient))
        errors = [float((a-b).norm()/a.norm()) for a, b in zip(results[0], results[1], strict=True)]
        assert max(errors) < 1e-5, errors
        run.write_json(root / "exact_reference_checkpoint_validation.json", dict(
            resolution=128, steps=20, endpoint_relative_error=errors[0], gradient_relative_error=errors[1]))
        print("Checkpointed Euler validation:", errors, flush=True)
        return
    path = root / f"exact_reference_checkpointed_steps_{args.steps}/production/manifest.json"
    if not path.exists():
        manifest = deepcopy(json.loads((root / "steps_10_rank_32/production/manifest.json").read_text()))
        manifest["created_at"] = datetime.now().astimezone().isoformat()
        manifest["settings"]["flow_steps"] = args.steps
        manifest["settings"]["execution"]["dflow_checkpointed"] = True
        manifest["source_sha256"] = source_hashes()
        manifest["source_sha256"][str(Path(__file__).relative_to(run.ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        manifest["protocol"]["methods"] = ["D-Flow"]
        manifest["protocol"]["purpose"] = "Fine-grid exact-gradient control with per-step activation checkpointing; unchanged optimizer and flow model"
        run.write_json(path, manifest)
    sys.argv = [sys.argv[0], "--manifest", str(path), "--job", f"r{args.resolution}_shapes_{args.problem}",
                "--sample-start", str(args.sample), "--sample-stop", str(args.sample+1),
                "--device", args.device, "--memory-fraction", "0.40"]
    run.METHODS = ("D-Flow",)
    run.euler_flow = checkpointed_euler
    run.main()


if __name__ == "__main__":
    main()
