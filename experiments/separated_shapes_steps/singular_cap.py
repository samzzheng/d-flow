"""Isolated DLR spectral-preconditioning control with unchanged L-BFGS."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import torch

from dlr_dflow.api import approx_source_gradient
from dlr_dflow.implementation.types import Factors
from experiments.ct_mri_sweep import run
from experiments.ct_mri_sweep.queue import source_hashes


def capped_gradient(factors: Factors, terminal: torch.Tensor, alpha1: float = 1.) -> torch.Tensor:
    left, core, right = factors
    core_left, singular, core_right = torch.linalg.svd(core, full_matrices=False)
    capped = (core_left * singular.clamp(max=singular.median())) @ core_right
    return approx_source_gradient((left, capped, right), terminal, alpha1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--problem", choices=("ct10", "mri4"), default="mri4")
    parser.add_argument("--sample", type=int, default=1)
    args = parser.parse_args()
    root = Path("outputs/separated_shapes_steps_20261019")
    path = root / "singular_cap_steps_5_rank_32/production/manifest.json"
    if not path.exists():
        manifest = deepcopy(json.loads((root / "steps_10_rank_32/production/manifest.json").read_text()))
        manifest["created_at"] = datetime.now().astimezone().isoformat()
        manifest["settings"]["flow_steps"] = 5
        manifest["settings"]["baseline_out"] = str(Path("outputs/separated_shapes_noiseless_20261016").resolve())
        manifest["source_sha256"] = source_hashes()
        manifest["source_sha256"][str(Path(__file__).relative_to(run.ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        manifest["protocol"]["methods"] = ["DLR D-Flow"]
        manifest["protocol"]["purpose"] = "Clip terminal core singular values to their lower median before applying the transpose; unchanged L-BFGS/Armijo"
        manifest["protocol"]["selection"] = "Selected failed case; experimental spectral preconditioning, not the original DLR gradient"
        run.write_json(path, manifest)
    sys.argv = [sys.argv[0], "--manifest", str(path), "--job", f"r{args.resolution}_shapes_{args.problem}",
                "--sample-start", str(args.sample), "--sample-stop", str(args.sample+1),
                "--device", "cuda:0", "--memory-fraction", ".15"]
    run.METHODS = ("DLR D-Flow",)
    run.approx_source_gradient = capped_gradient
    run.main()


if __name__ == "__main__":
    main()
