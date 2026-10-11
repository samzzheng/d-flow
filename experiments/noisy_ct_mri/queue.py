"""One admission controller for noisy studies and unfinished GPU-1 work."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
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

from experiments.ct_mri_sweep.queue import Task, gpu_processes, memory_reservation, tasks
from experiments.ct_mri_sweep.run import ROOT, Settings, model_ready, reviewed_source_hashes, write_json
from .report import report


@dataclass
class Work:
    manifest: Path
    task: Task
    settings: Settings
    rank_only: bool

    @property
    def name(self) -> str:
        return f"{self.manifest.parent.parent.name}/{self.manifest.parent.name}/{self.task.name}"

    @property
    def marker(self) -> Path:
        return self.task.marker(self.manifest.parent)


def choose(pending: list[Work], running: list[Work], total: int, used: int) -> Work | None:
    reserved = sum(memory_reservation(work.task, work.settings, total) for work in running)
    available = total - 1536 - max(used, reserved)
    return next((work for work in pending
                 if memory_reservation(work.task, work.settings, total) <= available), None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--continue-manifest", type=Path)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--gpu0-guard-pids", type=int, nargs="*", default=[])
    parser.add_argument("--gpus", type=int, nargs="+", default=[1, 0])
    parser.add_argument("--max-workers", type=int, default=12)
    args = parser.parse_args()
    out = args.out.resolve()
    lock = (out / "queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid()))
    lock.flush()
    manifests = list(out.glob("*/manifest.json"))
    # Admission alternates between the large comparison and the rank study.
    ordered = [out / "comparison/manifest.json", *(out / f"rank_{rank}/manifest.json" for rank in (2, 4, 8, 12, 16))]
    assert set(ordered) == set(manifests)
    legacy_lock = None
    if args.continue_manifest:
        legacy = args.continue_manifest.resolve()
        legacy_lock = (legacy.parent / "queue.lock").open("a")
        fcntl.flock(legacy_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ordered.insert(0, legacy)
    smoke_manifests = sorted(args.preflight.resolve().glob("*/manifest.json"))
    if len(smoke_manifests) != 6:
        raise ValueError("Preflight must include all five ranks and the three-method comparison")
    ordered.extend(smoke_manifests)
    groups: list[list[Work]] = []
    for path in ordered:
        manifest = json.loads(path.read_text())
        for name, expected in reviewed_source_hashes(path, manifest).items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Source changed: {name}")
        settings = manifest["settings"]
        task_list = tasks(settings)
        if path == out / "comparison/manifest.json":
            task_list.sort(key=lambda task: task.job.resolution)
        groups.append([Work(path, task, settings, manifest.get("methods") == ["DLR D-Flow"])
                       for task in task_list if not task.marker(path.parent).exists()])
    # Put one task from each study in each round, preserving legacy priority.
    pending = [group[index] for index in range(max(map(len, groups), default=0))
               for group in groups if index < len(group)]
    guards = {pid: Path(f"/proc/{pid}/stat").read_text().split()[21]
              for pid in args.gpu0_guard_pids if Path(f"/proc/{pid}/stat").exists()}
    smoke_markers = [task.marker(path.parent) for path in smoke_manifests
                     for task in tasks(json.loads(path.read_text())["settings"])]
    write_json(out / "scheduler_plan.json", {"manifests": list(map(str, ordered)),
               "gpu0_guard_process_starts": guards, "gpus": args.gpus,
               "tasks": len(pending), "memory_margin_mib": 1536,
               "legacy_continues_from_checkpoints": bool(args.continue_manifest)})
    active: dict[str, tuple[int, Work, subprocess.Popen[bytes]]] = {}
    failed = json.loads((out / "failures.json").read_text()) if (out / "failures.json").exists() else {}
    pending = [work for work in pending if work.name not in failed]
    completed: list[str] = []
    stopped = [False]

    def stop(signum: int, frame: Any) -> None:
        stopped[0] = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    (out / "logs").mkdir(exist_ok=True)
    report(out)
    try:
        while pending or active:
            for name, (_, work, process) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                del active[name]
                if code == 0 and work.marker.exists():
                    completed.append(name)
                    report(out)
                elif code == 0:
                    pending.insert(0, work)  # Graceful external-occupancy pause.
                else:
                    failed[name] = {"returncode": code, "manifest": str(work.manifest)}
                    write_json(out / "failures.json", failed)
            if stopped[0]:
                break
            occupancy = gpu_processes()
            rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.total,memory.used",
                                             "--format=csv,noheader,nounits"], text=True)
            memory = {index: (total, used) for index, total, used in
                      (map(int, row.split(",")) for row in rows.strip().splitlines())}
            guards = {pid: start for pid, start in guards.items() if Path(f"/proc/{pid}/stat").exists()
                      and Path(f"/proc/{pid}/stat").read_text().split()[21] == start}
            external = {}
            preflight_passed = all(path.exists() for path in smoke_markers)
            for gpu in args.gpus:
                own = [(work, proc) for device, work, proc in active.values() if device == gpu]
                external[gpu] = sorted(occupancy[gpu] - {proc.pid for _, proc in own})
                if external[gpu]:
                    for _, proc in own:
                        proc.terminate()
                    continue
                if gpu == 0 and guards or shutil.disk_usage(out).free < 20 * 2**30:
                    continue
                while len([1 for device, _, _ in active.values() if device == gpu]) < args.max_workers:
                    eligible = [work for work in pending if model_ready(work.task.job, work.settings)
                                and (preflight_passed or work.manifest.parent.parent != out)]
                    running = [work for device, work, _ in active.values() if device == gpu]
                    total, used = memory[gpu]
                    work = choose(eligible, running, total, used)
                    if work is None:
                        break
                    reservation = memory_reservation(work.task, work.settings, total)
                    fraction = min(work.settings["execution"]["memory_fraction"], (reservation - 768) / total)
                    module = "experiments.noisy_ct_mri.rank_worker" if work.rank_only else "experiments.ct_mri_sweep.run"
                    with (out / "logs" / (work.name.replace("/", "_") + ".log")).open("ab") as log:
                        process = subprocess.Popen([sys.executable, "-m", module, "--manifest", str(work.manifest),
                            "--job", work.task.job.name, "--device", f"cuda:{gpu}",
                            "--sample-start", str(work.task.start), "--sample-stop", str(work.task.stop),
                            "--memory-fraction", str(fraction)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"})
                    active[work.name] = (gpu, work, process)
                    pending.remove(work)
            write_json(out / "queue_status.json", {"status": "running" if active else "waiting" if pending else "finished",
                "updated_at": datetime.now().astimezone().isoformat(), "pid": os.getpid(),
                "active": {name: {"gpu": gpu, "pid": proc.pid,
                           "reserved_mib": memory_reservation(work.task, work.settings, memory[gpu][0])}
                           for name, (gpu, work, proc) in active.items()}, "pending": len(pending),
                "completed_this_session": completed, "failed": failed, "gpu0_guards": list(guards),
                "preflight_passed": preflight_passed,
                "external": external, "memory_total_used_mib": memory})
            if pending or active:
                time.sleep(10)
    finally:
        for _, _, process in active.values():
            if process.poll() is None:
                process.terminate()
        for _, _, process in active.values():
            process.wait()
        if stopped[0]:
            write_json(out / "queue_status.json", {"status": "paused", "pid": os.getpid()})
        lock.close()
        if legacy_lock:
            legacy_lock.close()


if __name__ == "__main__":
    main()
