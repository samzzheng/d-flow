"""Compare separate sample processes with one sequential worker on the same inputs."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from .queue import source_hashes
from .run import ROOT, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    manifest = deepcopy(json.loads(args.manifest.read_text()))
    manifest["settings"]["samples"] = args.samples
    manifest["settings"]["stopping"].update(min_updates=args.updates, max_updates=args.updates)
    manifest["protocol"]["smoke_only"] = True
    manifest["protocol"]["benchmark_deterministic"] = args.deterministic
    manifest["source_sha256"] = source_hashes()
    launcher = ("import torch; torch.use_deterministic_algorithms(True); "
                "torch.backends.cudnn.deterministic=True; " if args.deterministic else "")
    worker = "experiments.noisy_ct_mri.rank_worker" if manifest["protocol"]["methods"] == ["DLR D-Flow"] else "experiments.ct_mri_sweep.run"
    launcher += f"import runpy; runpy.run_module({worker!r}, run_name='__main__')"
    timings = {}
    for label in ("separate", "grouped"):
        root = args.out / label
        if root.exists():
            raise FileExistsError(f"Use a fresh benchmark directory: {root}")
        root.mkdir(parents=True)
        path = root / "manifest.json"
        write_json(path, manifest)
        ranges = [(sample, sample + 1) for sample in range(args.samples)] if label == "separate" else [(0, args.samples)]
        elapsed = []
        for start, stop in ranges:
            before = time.perf_counter()
            with (root / f"worker_{start}_{stop}.log").open("w") as log:
                subprocess.run([sys.executable, "-c", launcher, "--manifest", str(path.resolve()),
                    "--job", args.job, "--device", args.device, "--sample-start", str(start), "--sample-stop", str(stop)],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
                    env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
                         **({"CUBLAS_WORKSPACE_CONFIG": ":4096:8"} if args.deterministic else {})})
            elapsed.append(time.perf_counter() - before)
        timings[label] = elapsed
    comparisons = []
    for sample in range(args.samples):
        for method in manifest["protocol"]["methods"]:
            stem = method.lower().replace("-", "_").replace(" ", "_")
            paths = [args.out / label / args.job / f"sample_{sample:03d}" / f"{stem}.pt" for label in timings]
            a, b = [torch.load(path, weights_only=True, mmap=True) for path in paths]
            comparisons.append({"sample": sample, "method": method,
                "relative_differences": {key: float((a[key] - b[key]).norm() / a[key].norm().clamp_min(1e-20))
                                         for key in ("source", "image", "measurement")},
                "same_phi": [row["phi"] for row in a["history"]] == [row["phi"] for row in b["history"]],
                "same_evaluations": a["evaluations"] == b["evaluations"], "same_status": a["status"] == b["status"]})
        for label in timings:
            assert (args.out / label / args.job / f"shard_{sample:03d}_{sample + 1:03d}.json").exists()
    write_json(args.out / "comparison.json", {"deterministic": args.deterministic, "worker_seconds": timings,
        "total_seconds": {label: sum(values) for label, values in timings.items()}, "comparisons": comparisons})
    print(json.dumps(timings), flush=True)


if __name__ == "__main__":
    main()
