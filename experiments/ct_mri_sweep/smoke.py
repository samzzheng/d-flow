"""Short complete-worker CUDA checks, including memory peaks and contention."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .queue import gpu_processes, source_hashes
from .run import ROOT, Job, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--problem", choices=("ct", "mri"), default="ct")
    parser.add_argument("--memory-fraction", type=float, default=.95)
    args = parser.parse_args()
    if gpu_processes()[args.gpu]:
        raise RuntimeError("Smoke checks require an idle GPU")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    settings = json.loads(Path(__file__).with_name("config.json").read_text())
    settings.update(resolutions=[args.resolution], circles=[3], ct_angles=[10], mri_accelerations=[4], samples=1)
    settings["stopping"].update(min_updates=args.updates, max_updates=args.updates, window=1)
    settings["execution"].update(checkpoint_every=min(25, args.updates), memory_fraction=args.memory_fraction, threads=2)
    job = Job(args.resolution, 3, args.problem, 10 if args.problem == "ct" else 4)
    processes = []
    started = time.monotonic()
    peaks: dict[int, float] = {}
    readings: list[dict[str, float]] = []
    try:
        for worker in range(args.workers):
            directory = out / f"worker_{worker}"
            directory.mkdir()
            manifest = {"settings": settings, "source_sha256": source_hashes(), "jobs": [asdict(job)],
                        "purpose": "smoke only; excluded from reconstruction study"}
            write_json(directory / "manifest.json", manifest)
            with (directory / "worker.log").open("w") as log:
                process = subprocess.Popen([sys.executable, "-m", "experiments.ct_mri_sweep.run",
                    "--manifest", str(directory / "manifest.json"), "--job", job.name,
                    "--device", f"cuda:{args.gpu}"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                         "OPENBLAS_NUM_THREADS": "1", "MPLBACKEND": "Agg"})
            processes.append(process)
        while any(process.poll() is None for process in processes):
            rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                            "--format=csv,noheader,nounits"], text=True)
            for row in rows.strip().splitlines():
                pid, memory = row.split(",")
                if int(pid) in {process.pid for process in processes}:
                    peaks[int(pid)] = max(peaks.get(int(pid), 0), float(memory))
            memory, utilization = map(float, subprocess.check_output([
                "nvidia-smi", "-i", str(args.gpu), "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits"], text=True).strip().split(","))
            readings.append({"elapsed_seconds": time.monotonic() - started,
                             "memory_mib": memory, "utilization_percent": utilization})
            time.sleep(.5)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        record = {"resolution": args.resolution, "problem": args.problem, "workers": args.workers,
                  "updates": args.updates, "elapsed_seconds": time.monotonic() - started,
                  "exit_codes": [process.returncode for process in processes],
                  "process_peak_nvml_mib": peaks, "readings": readings}
        write_json(out / "measurements.json", record)
        print(json.dumps({key: value for key, value in record.items() if key != "readings"}), flush=True)


if __name__ == "__main__":
    main()
