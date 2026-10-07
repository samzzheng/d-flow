"""Queue the local GPU sweep after training, yielding to other GPU processes.

python -m experiments.ct_mri_sweep.queue init --out outputs/ct_mri_sweep_20261007
python -m experiments.ct_mri_sweep.queue run --out outputs/ct_mri_sweep_20261007
"""
from __future__ import annotations

import argparse
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

from .run import ROOT, Job, Settings, jobs, model_ready, write_json


def source_hashes() -> dict[str, str]:
    paths = [*(ROOT / "experiments/ct_mri_sweep" / name for name in ("__init__.py", "run.py", "queue.py", "operators.py")),
             *(ROOT / "p_flow" / name for name in ("__init__.py", "p_flow.py", "armijo.py")),
             *Path(ROOT / "dlr_dflow").glob("*.py"),
             *Path(ROOT / "dlr_dflow/implementation").glob("*.py"),
             *Path(ROOT / "models/unet").glob("*.py"),
             ROOT / "sampler/ode_samplers.py", ROOT / "sampler/optimizer_state.py",
             ROOT / "scripts/reconstruction_figures.py"]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def runtime_estimate(settings: Settings) -> Settings:
    """Scenario estimate from measured native forward and fastest DLR calls."""
    timings = {}
    for resolution in settings["resolutions"]:
        path = ROOT / f"outputs/iteration_speed_150steps_2026_09_28/res{resolution}.json"
        record = json.loads(path.read_text())
        factor = settings["flow_steps"] / record["protocol"]["steps"]
        timings[resolution] = {key: value * factor for key, value in record["median_seconds"].items()
                               if key in {"dflow", "dlr", "forward_trial"}}
    cases_per_resolution = len(settings["circles"]) * (len(settings["ct_angles"]) + len(settings["mri_accelerations"])) * settings["samples"]
    scenarios = []
    for updates in (100, 250, 500, 1000, settings["stopping"]["max_updates"]):
        for trials in (1, 2):
            seconds = cases_per_resolution * updates * sum(
                value["dflow"] + value["dlr"] + 3 * trials * value["forward_trial"]
                for value in timings.values())
            scenarios.append({"mean_updates_per_method": updates, "mean_trials_per_update": trials,
                              "gpu_hours": seconds / 3600, "one_gpu_days": seconds / 86400,
                              "two_dedicated_gpu_days": seconds / (2 * 86400)})
    return {"kind": "scenario projection, not a completion guarantee", "timings_seconds_100_steps": timings,
            "basis": "RTX 6000 Ada historical 150-step calls scaled linearly to configured step count; rank-8 analytical-alpha fused/graphed DLR; native exact D-Flow; P-Flow reuses accepted trial endpoint",
            "excludes": ["training wait", "other users' GPU occupancy", "new checkpoint effects on convergence",
                         "acquisition overhead changes (especially CT180)", "initial graph capture", "checkpoint and figure IO"],
            "scenarios": scenarios}


def gpu_processes() -> dict[int, set[int]]:
    info = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True)
    ids = {uuid.strip(): int(index) for index, uuid in (row.split(",") for row in info.strip().splitlines())}
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
    result = {index: set() for index in ids.values()}
    for row in processes.strip().splitlines():
        uuid, pid = map(str.strip, row.split(","))
        result[ids[uuid]].add(int(pid))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("init", "run", "status"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--gpus", type=int, nargs="+", default=[1, 0])
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    out = args.out.resolve()
    manifest_path = out / "manifest.json"
    if args.mode == "init":
        if manifest_path.exists():
            raise FileExistsError("Use the existing immutable manifest, or a new output directory")
        settings = json.loads(args.config.read_text())
        if settings["mri_pattern"] not in {"cartesian", "points"}:
            raise ValueError("mri_pattern must be cartesian or points")
        if settings["source_projection"] or settings["execution"]["dflow_checkpointed"]:
            raise ValueError("This fixed protocol uses unconstrained sources and uncheckpointed exact gradients")
        stopping = settings["stopping"]
        if not 0 < 2 * stopping["window"] <= stopping["min_updates"] <= stopping["max_updates"]:
            raise ValueError("Stopping windows must fit within min/max updates")
        manifest = {"created_at": datetime.now().astimezone().isoformat(), "settings": settings,
                    "source_sha256": source_hashes(),
                    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "jobs": [asdict(job) for job in jobs(settings)],
                    "count": {"acquisitions": len(jobs(settings)), "paired_cases": len(jobs(settings)) * settings["samples"],
                              "method_runs": len(jobs(settings)) * settings["samples"] * 3},
                    "protocol": {"objective": "0.5 * ||A G(z) - y||^2; float64 sum, float32 models/sources",
                                 "optimizer": "same p_flow.armijo.ArmijoLBFGS instance type and parameters for all methods",
                                 "initial_step_policy": "first_iteration_only", "armijo_progress": "strict_decrease_nonzero_step",
                                 "dlr_refresh": "every gradient; fixed per-case initialization seed",
                                 "pflow_variant": "identity sensitivity proxy with shared LBFGS/Armijo, no shell projection",
                                 "measurement_noise": "Gaussian direction rescaled to exact relative norm; complex circular direction for MRI",
                                 "queue": "all 24 training summaries complete before any job; one worker per idle GPU; pause if an external CUDA process appears",
                                 "checkpoint_selection": "terminal accepted iterate; no truth-based selection",
                                 "figures": "one PNG and PDF comparison for every paired sample, including all three methods"}}
        write_json(manifest_path, manifest)
        write_json(out / "runtime_estimate.json", runtime_estimate(settings))
        print(json.dumps(manifest["count"], indent=2))
        return
    manifest = json.loads(manifest_path.read_text())
    if args.mode == "status":
        print((out / "queue_status.json").read_text() if (out / "queue_status.json").exists() else "Initialized, not started")
        return
    if len(set(args.gpus)) != len(args.gpus) or args.poll_seconds < 1:
        raise ValueError("Choose unique GPUs and a positive polling interval")
    settings = manifest["settings"]
    lock = (out / "queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid()))
    lock.flush()
    interrupted = [False]

    def stop(signum: int, frame: Any) -> None:
        interrupted[0] = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    active: dict[int, tuple[Job, subprocess.Popen[str]]] = {}
    failures = json.loads((out / "queue_failures.json").read_text()) if (out / "queue_failures.json").exists() else {}
    log_directory = out / "logs"
    log_directory.mkdir(exist_ok=True)
    pending_models: list[str] = []
    previous_status = None
    try:
        while not interrupted[0]:
            for gpu, (job, process) in list(active.items()):
                code = process.poll()
                if code is not None:
                    del active[gpu]
                    if code != 0:
                        failures[job.name] = {"exit_code": code, "log": str(log_directory / f"{job.name}.log")}
                        write_json(out / "queue_failures.json", failures)
            unfinished = [job for job in jobs(settings) if not (out / job.name / "complete.json").exists()]
            if not unfinished:
                status = "complete"
            elif source_hashes() != manifest["source_sha256"]:
                status = "waiting_for_source_review"
            elif shutil.disk_usage(out).free < 20 * 2 ** 30:
                status = "waiting_for_disk_space"
            else:
                pending_models = sorted({f"r{job.resolution}_n{job.circles}" for job in unfinished if not model_ready(job, settings)})
                status = "waiting_for_training" if pending_models else "running"
            external: dict[int, list[int]] = {}
            if status == "running":
                occupancy = gpu_processes()
                for gpu in args.gpus:
                    own_pid = active[gpu][1].pid if gpu in active else None
                    external[gpu] = sorted(occupancy[gpu] - ({own_pid} if own_pid else set()))
                    if external[gpu] and gpu in active:
                        active[gpu][1].send_signal(signal.SIGTERM)
                    if external[gpu] or gpu in active:
                        continue
                    remaining = [job for job in unfinished if job.name not in failures
                                 and job not in [entry[0] for entry in active.values()]]
                    if not remaining:
                        continue
                    job = remaining[0]
                    with (log_directory / f"{job.name}.log").open("a") as log:
                        process = subprocess.Popen([
                            sys.executable, "-m", "experiments.ct_mri_sweep.run", "--manifest", str(manifest_path),
                            "--job", job.name, "--device", f"cuda:{gpu}"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
                    active[gpu] = (job, process)
                if not active:
                    status = "failed_jobs_remaining" if all(job.name in failures for job in unfinished) else "waiting_for_gpus"
            record = {"status": status, "updated_at": datetime.now().astimezone().isoformat(),
                      "training_pending": pending_models, "completed_acquisitions": len(manifest["jobs"]) - len(unfinished),
                      "total_acquisitions": len(manifest["jobs"]), "gpus": args.gpus,
                      "active": {gpu: {"job": job.name, "pid": proc.pid} for gpu, (job, proc) in active.items()},
                      "external_gpu_processes": external, "failed_acquisitions": list(failures)}
            write_json(out / "queue_status.json", record)
            if status != previous_status:
                print(json.dumps(record), flush=True)
                previous_status = status
            if status in {"complete", "failed_jobs_remaining"}:
                break
            time.sleep(args.poll_seconds)
    finally:
        for _, process in active.values():
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
        for _, process in active.values():
            process.wait()
        if interrupted[0]:
            write_json(out / "queue_status.json", {"status": "paused", "updated_at": datetime.now().astimezone().isoformat()})
        lock.close()


if __name__ == "__main__":
    main()
