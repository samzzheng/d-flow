"""Freeze separate noisy manifests without modifying existing experiments."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess

from experiments.ct_mri_sweep.queue import source_hashes
from experiments.ct_mri_sweep.run import ROOT, Settings, jobs, write_json


def settings_for(rank: int | None = None, smoke: bool = False) -> Settings:
    settings = json.loads((ROOT / "experiments/ct_mri_sweep/config_fp32.json").read_text())
    settings["noise_relative"] = .01
    if rank is not None:
        settings.update(resolutions=[64], circles=[1, 2, 3, 4], samples=10)
        settings["dlr"]["rank"] = rank
        settings["scheduling"]["worker_memory_mib"]["64"] = 4096
    if smoke:
        settings.update(resolutions=[64], circles=[3], samples=1)
        settings["stopping"].update(min_updates=2, max_updates=2, window=1)
        settings["execution"]["checkpoint_every"] = 1
    return settings


def initialize(out: Path, smoke: bool = False) -> None:
    hashes = source_hashes()
    hashes.update({str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted((ROOT / "experiments/noisy_ct_mri").glob("*.py"))})
    for name, rank in [("comparison", None), *((f"rank_{rank}", rank) for rank in (2, 4, 8, 12, 16))]:
        settings = settings_for(rank, smoke)
        methods = ["DLR D-Flow"] if rank is not None else ["D-Flow", "P-Flow", "DLR D-Flow"]
        path = out / name / "manifest.json"
        manifest = {"created_at": datetime.now().astimezone().isoformat(), "settings": settings,
                    "source_sha256": deepcopy(hashes),
                    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "jobs": [asdict(job) for job in jobs(settings)], "methods": methods,
                    "count": {"paired_cases": len(jobs(settings)) * settings["samples"],
                              "method_runs": len(jobs(settings)) * settings["samples"] * len(methods)},
                    "protocol": {"smoke_only": smoke, "noise_relative": .01,
                        "noise_definition": "||y-A(x_GT)||/||A(x_GT)|| = 0.01; Gaussian, complex on acquired MRI coefficients",
                        "pairing": "same sample, CPU source draw, mask, noise and DLR seed across ranks and methods; fresh optimizer states",
                        "optimizer": "unchanged LBFGS/geometric Armijo; 100 attempts maximum, two failed searches stop",
                        "dlr": "fixed rank; analytical alpha=1-t; previous V; no trace probes or enrichment",
                        "selection": "final checkpoint, no ground-truth stopping or selection",
                        "figures": "shared plotter with standard method columns, five rows and fixed CT/MRI residual limits"}}
        if path.exists():
            raise FileExistsError(path)
        write_json(path, manifest)
    write_json(out / "plan.json", {"smoke_only": smoke, "rank_study": {
        "resolution": 64, "circles": [1, 2, 3, 4], "ranks": [2, 4, 8, 12, 16],
        "samples_per_acquisition": 10, "method_runs": 400}, "comparison": {
        "resolutions": [64, 128, 250], "circles": [1, 2, 3], "samples_per_acquisition": 30,
        "methods": ["D-Flow", "P-Flow", "DLR D-Flow"], "method_runs": 1620},
        "acquisitions": ["CT10", "Cartesian MRI4"], "noise_relative": .01})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    initialize(args.out.resolve(), args.smoke)
