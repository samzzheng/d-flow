"""Freeze the completed 117k-model comparison using the existing sweep worker."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

from experiments.ct_mri_sweep.queue import source_hashes
from experiments.ct_mri_sweep.run import ROOT, jobs, model_ready, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if (out / "production/manifest.json").exists():
        raise FileExistsError("Use the frozen manifest to resume this experiment")
    original = ROOT / "outputs/separated_shapes_noiseless_20261016/production/manifest.json"
    settings = json.loads(original.read_text())["settings"]
    settings.update(epoch=500, run_template=str(out / "models/r{resolution}"),
                    diagnostic_figures=True)
    settings["dlr"]["rank"] = args.rank
    settings["scheduling"].update(max_workers_per_gpu=4, gpu_headroom_mib=2048,
        worker_memory_mib={str(r): 8192 for r in settings["resolutions"]})
    checkpoints = {}
    for resolution in settings["resolutions"]:
        run, = filter(Path.is_dir, (ROOT / "outputs/fm_unet").glob(
            f"separated-shapes-{resolution}-117k-time32-e500-*"))
        link = out / "models" / f"r{resolution}"
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(run, link.parent), target_is_directory=True)
        checkpoint = run / "checkpoints/epoch_0500.pt"
        checkpoints[str(resolution)] = dict(run=str(run.relative_to(ROOT)),
            sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    assert all(model_ready(job, settings) for job in jobs(settings))
    manifest = dict(created_at=datetime.now().astimezone().isoformat(), settings=settings,
        source_sha256=source_hashes(), checkpoints=checkpoints,
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        jobs=list(map(asdict, jobs(settings))),
        count=dict(acquisitions=12, paired_cases=360, method_runs=1080),
        protocol=dict(smoke_only=False, methods=["D-Flow", "P-Flow", "DLR D-Flow"],
            pairing="Same truth, source, observation and mask for every method; fresh optimizer states",
            initialization="Original seeded standard Gaussian; no measurement-informed initialization",
            checkpoint_selection="Final accepted source; no ground-truth selection",
            noise="Noiseless: observed y equals clean A(x_GT) exactly"))
    write_json(out / "production/manifest.json", manifest)
    smoke = deepcopy(manifest)
    smoke["settings"]["samples"] = 1
    smoke["settings"]["stopping"].update(min_updates=4, max_updates=4, window=1)
    smoke["settings"]["execution"]["checkpoint_every"] = 1
    smoke["protocol"]["smoke_only"] = True
    smoke.pop("count")
    write_json(out / "smoke/manifest.json", smoke)


if __name__ == "__main__":
    main()
