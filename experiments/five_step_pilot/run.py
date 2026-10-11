"""Freeze, run and summarize the noiseless single-circle five-step pilot."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

from experiments.ct_mri_sweep.queue import gpu_processes, source_hashes, tasks
from experiments.ct_mri_sweep.run import ROOT, TERMINAL, jobs, model_ready, write_json
from experiments.noisy_ct_mri.queue import Work, choose


def report(out: Path) -> None:
    manifest = json.loads((out / "production/manifest.json").read_text())
    lines = ["# Five-step Euler pilot", "",
             "Noiseless CT10 and Cartesian MRI4; one single-circle test sample at each resolution. "
             "Shared LBFGS/geometric Armijo, FP32, up to 1,000 attempts; two consecutive failed "
             "line searches stop a method. A step limit does not certify convergence. "
             "DLR uses rank8, analytical alpha=1-t, no trace probes/enrichment and previous V.", "",
             "Final saved checkpoints, not retrospective best-image selection. "
             "Five-step Euler changes the generator being optimized; more optimizer steps "
             "do not remove its time-discretization error. One sample cannot establish a population ranking.", "",
             "| Case | Method | Attempts | Accepted updates | Final image error % | Measurement residual % | Status |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    rows = []
    for job in sorted(jobs(manifest["settings"]), key=lambda job: (job.resolution, job.problem)):
        for method, stem in (("D-Flow", "d_flow"), ("P-Flow", "p_flow"), ("DLR D-Flow", "dlr_d_flow")):
            path = out / "production" / job.name / "sample_000" / f"{stem}.json"
            if not path.exists():
                continue
            record = json.loads(path.read_text())
            last = record["history"][-1]
            accepted = sum(row.get("source_step_norm", 0) > 0 for row in record["history"])
            row = {"case": job.name, "method": method, "attempts": last["step"],
                   "accepted_updates": accepted, "image_percent": 100 * last["image_relative_l2"],
                   "residual_percent": 100 * last["measurement_relative_l2_noisy"], "status": record["status"]}
            rows.append(row)
            lines.append(f"| {job.name} | {method} | {row['attempts']} | {accepted} | "
                         f"{row['image_percent']:.4f} | {row['residual_percent']:.4f} | {row['status']} |")
    complete = sum(row["status"] in TERMINAL for row in rows)
    lines += ["", f"Terminal method runs: {complete}/36. Running rows are provisional.", "",
              "Each production acquisition saves `sample_000/comparison.png` and `.pdf`, "
              "including image/residual panels and error versus optimizer attempts."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    write_json(out / "summary.json", {"terminal_runs": complete, "total_runs": 36, "rows": rows})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("init", "run", "report"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shared-controller", type=int, nargs=2, action="append", default=[],
                        metavar=("PID", "MIB"))
    args = parser.parse_args()
    out = args.out.resolve()
    if args.mode == "init":
        settings = json.loads((ROOT / "experiments/ct_mri_sweep/config_fp32.json").read_text())
        settings.update(resolutions=[64, 96, 128, 160, 200, 250], circles=[1], samples=1,
                        flow_steps=5, noise_relative=0.0)
        settings["stopping"].update(min_updates=1000, max_updates=1000)
        settings["scheduling"].update(samples_per_worker=1, max_workers_per_gpu=1,
            worker_memory_mib={str(resolution): 9216 for resolution in settings["resolutions"]})
        hashes = source_hashes()
        hashes.update({str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in [*Path(__file__).parent.glob("*.py"), ROOT / "experiments/noisy_ct_mri/queue.py"]})
        for stage in ("smoke", "production"):
            path = out / stage / "manifest.json"
            if path.exists():
                raise FileExistsError(path)
            current = deepcopy(settings)
            if stage == "smoke":
                current["resolutions"] = [250]
                current["stopping"].update(min_updates=2, max_updates=2, window=1)
                current["execution"]["checkpoint_every"] = 1
            if not all(model_ready(job, current) for job in jobs(current)):
                raise RuntimeError("A requested single-circle model is not trained")
            write_json(path, {"created_at": datetime.now().astimezone().isoformat(), "settings": current,
                "source_sha256": hashes, "jobs": list(map(asdict, jobs(current))),
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "protocol": {"smoke_only": stage == "smoke", "sample": 0,
                    "methods": ["D-Flow", "P-Flow", "DLR D-Flow"], "gpu": 1,
                    "pairing": "Same truth, acquisition mask and source seeds across methods; fresh optimizer states",
                    "stopping": "Up to 1000 attempts; preserve two-failed-search stop; no plateau stop before budget",
                    "noise": "none: y=A(x_GT)", "baseline": "Existing 100-step runs are historical, not a new paired control"}})
        report(out)
        return
    if args.mode == "report":
        report(out)
        return
    lock = (out / "queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid()))
    lock.flush()
    pending = []
    for stage in ("smoke", "production"):
        path = out / stage / "manifest.json"
        manifest = json.loads(path.read_text())
        for name, digest in manifest["source_sha256"].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Source changed: {name}")
        pending.extend(Work(path, task, manifest["settings"], False)
                       for task in sorted(tasks(manifest["settings"]), key=lambda task: (task.job.resolution, task.job.problem))
                       if not task.marker(path.parent).exists())
    shared = {pid: (mib, Path(f"/proc/{pid}/stat").read_text().split()[21])
              for pid, mib in args.shared_controller}
    stopped = [False]
    active: tuple[Work, subprocess.Popen[bytes]] | None = None
    failures = json.loads((out / "failures.json").read_text()) if (out / "failures.json").exists() else {}

    def stop(signum: int, frame: Any) -> None:
        stopped[0] = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    (out / "logs").mkdir(exist_ok=True)
    try:
        while pending or active:
            if stopped[0]:
                break
            if active and active[1].poll() is not None:
                work, process = active
                active = None
                if process.returncode != 0:
                    failures[work.name] = process.returncode
                    write_json(out / "failures.json", failures)
                elif not work.marker.exists():
                    pending.insert(0, work)
                report(out)
            if any("/smoke/" in name for name in failures):
                raise RuntimeError("Preflight failed; production remains blocked")
            pending = [work for work in pending if work.name not in failures]
            shared = {pid: entry for pid, entry in shared.items() if Path(f"/proc/{pid}/stat").exists()
                      and Path(f"/proc/{pid}/stat").read_text().split()[21] == entry[1]
                      and Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"}
            allowed = set(shared)
            for pid in shared:
                children = Path(f"/proc/{pid}/task/{pid}/children")
                if children.exists():
                    allowed.update(map(int, children.read_text().split()))
            if active:
                allowed.add(active[1].pid)
            external = gpu_processes()[1] - allowed
            total, used = map(int, subprocess.check_output(["nvidia-smi", "--id=1",
                "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"], text=True).strip().split(","))
            if external and active:
                active[1].terminate()
            if not active and pending and not external and shutil.disk_usage(out).free > 20 * 2**30:
                # Sequential admission keeps the 250px smoke ahead of every production run.
                work = choose(pending[:1], [], total, used, sum(mib for mib, _ in shared.values()))
                if work is not None:
                    with (out / "logs" / (work.name.replace("/", "_") + ".log")).open("ab") as log:
                        process = subprocess.Popen([sys.executable, "-m", "experiments.ct_mri_sweep.run",
                            "--manifest", str(work.manifest), "--job", work.task.job.name, "--device", "cuda:1",
                            "--sample-start", "0", "--sample-stop", "1", "--memory-fraction", str((9216 - 768) / total)],
                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"})
                    active = (work, process)
                    pending.remove(work)
            write_json(out / "queue_status.json", {"pid": os.getpid(), "gpu": 1,
                "updated_at": datetime.now().astimezone().isoformat(),
                "status": "running" if active else "waiting" if pending else "finished",
                "active": {"job": active[0].name, "pid": active[1].pid} if active else None,
                "pending": [work.name for work in pending], "failures": failures,
                "shared_controllers": shared, "external": sorted(external), "memory_total_used_mib": [total, used]})
            if pending or active:
                time.sleep(5)
    finally:
        if active and active[1].poll() is None:
            active[1].terminate()
            active[1].wait()
        if stopped[0]:
            write_json(out / "queue_status.json", {"status": "paused", "pid": os.getpid(), "gpu": 1})
        report(out)
        lock.close()


if __name__ == "__main__":
    main()
