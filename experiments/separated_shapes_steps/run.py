"""Paired Euler-grid reconstruction pilot using the existing sweep worker."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

from experiments.ct_mri_sweep.queue import source_hashes
from experiments.ct_mri_sweep.run import TERMINAL, write_json

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "outputs/separated_shapes_steps_20261019"
CASES = [(250, "mri4", 0), (64, "mri4", 1), (128, "ct10", 0),
         (96, "mri4", 1), (160, "ct10", 0), (200, "ct10", 1),
         (64, "ct10", 0), (128, "mri4", 2)]
STEMS = ("d_flow", "p_flow", "dlr_d_flow")


def report(out: Path) -> None:
    plan = json.loads((out / "plan.json").read_text())
    records = []
    lines = ["# Separated-shapes Euler-step pilot", "",
             "Selected failure cases plus two successful controls; original epoch-200 larger models. "
             "Noiseless CT10/MRI4x, paired sources and masks, unchanged LBFGS/Armijo, up to 1000 attempts. "
             "Rank 32, analytical alpha=1-t, previous-V reuse. These selected cases cannot establish dataset-wide reliability. "
             "Elapsed times include GPU sharing and are not isolated speed benchmarks.", "",
             "| Steps | Case | Method | Status | Attempts | Image error % | Measurement error % | Seconds |",
             "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for task in plan["tasks"]:
        case = Path(task["manifest"]).parent / task["job"] / f"sample_{task['sample']:03d}"
        for stem in STEMS:
            path = case / f"{stem}.json"
            if not path.exists():
                continue
            result = json.loads(path.read_text())
            last = result["history"][-1]
            record = dict(steps=task["steps"], case=f"{task['job']}/{case.name}", method=result["method"],
                          status=result["status"], attempts=last["step"],
                          image_error_percent=100 * last["image_relative_l2"],
                          measurement_error_percent=100 * last["measurement_relative_l2_clean"],
                          seconds=result["elapsed_seconds"])
            records.append(record)
            lines.append(f"| {record['steps']} | {record['case']} | {record['method']} | {record['status']} | "
                         f"{record['attempts']} | {record['image_error_percent']:.3f} | "
                         f"{record['measurement_error_percent']:.3f} | {record['seconds']:.1f} |")
    write_json(out / "reconstructions.json", records)
    (out / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "run", "report"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-fraction", type=float, default=.30)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.action == "init":
        template = json.loads((ROOT / "outputs/separated_shapes_rank_sweep_20261017/rank_128/production/manifest.json").read_text())
        hashes = source_hashes()
        hashes[str(Path(__file__).resolve().relative_to(ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        tasks = []
        for steps in (10, 20):
            manifest = deepcopy(template)
            manifest["created_at"] = datetime.now().astimezone().isoformat()
            manifest["source_sha256"] = hashes
            manifest["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            settings = manifest["settings"]
            settings.pop("baseline_out", None)
            settings.pop("rank_sweep_root", None)
            settings["flow_steps"] = steps
            settings["dlr"]["rank"] = 32
            manifest["protocol"] = dict(smoke_only=False, methods=["D-Flow", "P-Flow", "DLR D-Flow"],
                pairing="Original sample/source/mask seeds; fresh optimization on each Euler grid",
                selection="Six challenging cases and two successful controls; exploratory, not an unbiased accuracy estimate")
            path = out / f"steps_{steps}_rank_32/production/manifest.json"
            if path.exists():
                raise FileExistsError(f"Refusing to replace frozen manifest: {path}")
            write_json(path, manifest)
            tasks.extend(dict(manifest=str(path), job=f"r{r}_shapes_{problem}", sample=sample, steps=steps)
                         for r, problem, sample in CASES)
        # Start one high-resolution and one low-resolution case to assess memory.
        tasks.sort(key=lambda task: (CASES.index((int(task["job"].split('_')[0][1:]), task["job"].split('_')[-1], task["sample"])) // 2,
                                     task["steps"], task["job"]))
        write_json(out / "plan.json", dict(tasks=tasks, code_sha256=hashes))
        return
    if args.action == "report":
        report(out)
        return
    stopping = [False]

    def stop(signum: int, frame: Any) -> None:
        stopping[0] = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    plan = json.loads((out / "plan.json").read_text())
    with (out / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pending = []
        for task in plan["tasks"]:
            case = Path(task["manifest"]).parent / task["job"] / f"sample_{task['sample']:03d}"
            if not all((case / f"{stem}.json").exists() and
                       json.loads((case / f"{stem}.json").read_text())["status"] in TERMINAL for stem in STEMS):
                pending.append(task)
        active: list[tuple[subprocess.Popen[bytes], dict[str, Any]]] = []
        failures = []
        while pending or active:
            for process, task in list(active):
                code = process.poll()
                if code is not None:
                    active.remove((process, task))
                    if code:
                        failures.append(dict(task=task, exit_code=code))
            if stopping[0]:
                for process, _ in active:
                    process.send_signal(signal.SIGTERM)
                for process, _ in active:
                    process.wait()
                break
            while pending and len(active) < args.workers:
                task = pending.pop(0)
                command = [sys.executable, "-u", "-m", "experiments.ct_mri_sweep.run", "--manifest", task["manifest"],
                           "--job", task["job"], "--sample-start", str(task["sample"]),
                           "--sample-stop", str(task["sample"] + 1), "--device", args.device,
                           "--memory-fraction", str(args.memory_fraction)]
                with (out / f"steps{task['steps']}_{task['job']}_{task['sample']}.log").open("ab") as log:
                    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                active.append((process, task))
            write_json(out / "queue_status.json", dict(status="running" if active else "complete",
                active=[dict(pid=process.pid, **task) for process, task in active], pending=len(pending), failures=failures))
            report(out)
            if active:
                time.sleep(10)
        write_json(out / "queue_status.json", dict(status="paused" if stopping[0] else "finished",
                                                   pending=len(pending), failures=failures))
        report(out)


if __name__ == "__main__":
    main()
