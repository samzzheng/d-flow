"""GPU-0-only bounded two-worker queue; GPU 1 and its sweep are untouched."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

from experiments.ct_mri_sweep.run import write_json
from .report import report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("outputs/ct_mri_sweep_fp32_20261009"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "logs").mkdir(exist_ok=True)
    tasks: list[dict[str, Any]] = []
    unavailable = []
    # Cover the current study first, then all other available models. Three
    # paired Gaussian initial sources per model; full saved spectra, not r=8.
    models = [(r, n) for r in (64, 128, 250, 96, 160, 200) for n in (1, 2, 3, 4)]
    for sample in range(3):
        for resolution, circles in models:
            checkpoint = Path(f"outputs/fm_unet/circles-{resolution}-noattn-ch16-b8-e200/n{circles}/checkpoints/epoch_0200.pt")
            if not checkpoint.exists():
                if sample == 0:
                    unavailable.append({"resolution": resolution, "circles": circles, "checkpoint": str(checkpoint)})
                continue
            name = f"rank_r{resolution}_n{circles}_s{sample}_initial"
            destination = args.output / "ranks" / f"{name}.json"
            tasks.append({"name": name, "kind": "rank", "memory_mib": 32000,
                          "output": str(destination), "command": [sys.executable, "-m", "experiments.multicircle_rank_hybrid.ranks",
                          "--resolution", str(resolution), "--circles", str(circles), "--sample", str(sample),
                          "--output", str(destination), "--device", "cuda:0"]})
    # Saved optimized sources expose variation hidden by initial-only spectra.
    for resolution in (64, 128, 250):
        for circles in (1, 2, 3):
            for problem in ("ct10", "mri4"):
                for method in ("d_flow", "dlr_d_flow"):
                    source = args.root / f"r{resolution}_n{circles}_{problem}/sample_000/{method}.pt"
                    summary = source.with_suffix(".json")
                    if not summary.exists() or json.loads(summary.read_text())["status"] not in {"step_limit", "stalled"}:
                        continue
                    name = f"rank_r{resolution}_n{circles}_{problem}_{method}_final"
                    destination = args.output / "ranks" / f"{name}.json"
                    tasks.append({"name": name, "kind": "rank", "memory_mib": 32000,
                                  "output": str(destination), "command": [sys.executable, "-m", "experiments.multicircle_rank_hybrid.ranks",
                                  "--resolution", str(resolution), "--circles", str(circles),
                                  "--source", str(source), "--output", str(destination), "--device", "cuda:0"]})
    missing_inputs = []
    for sample in range(3):
        for resolution in (64, 128, 250):
            for circles in (2, 3, 1):
                for problem in ("ct10", "mri4"):
                    case = f"r{resolution}_n{circles}_{problem}/sample_{sample:03d}"
                    if not (args.root / case / "inputs.pt").exists():
                        missing_inputs.append(case)
                    name = "hybrid_" + case.replace("/", "_")
                    memory = {64: 5000, 128: 10000, 250: 32000}[resolution]
                    tasks.append({"name": name, "kind": "hybrid", "memory_mib": memory,
                                  "required_input": str(args.root / case / "inputs.pt"),
                                  "output": str(args.output / "hybrid" / case / "complete.json"),
                                  "command": [sys.executable, "-m", "experiments.multicircle_rank_hybrid.hybrid",
                                  "--root", str(args.root), "--output", str(args.output / "hybrid"),
                                  "--case", case, "--device", "cuda:0", "--memory-fraction", str(memory / 49140)]})
    cancelled = []
    if (args.output / "cancel_hybrid.json").exists():
        cancelled = [task["name"] for task in tasks if task["kind"] == "hybrid"
                     and not Path(task["output"]).exists()]
        tasks = [task for task in tasks if task["kind"] != "hybrid"]
    write_json(args.output / "plan.json", {"tasks": tasks, "cancelled_hybrid": cancelled, "unavailable_models": unavailable,
                                           "missing_paired_inputs": missing_inputs, "gpu": 0})
    pending, completed, failed = [], [], []
    for task in tasks:
        path = Path(task["output"])
        if path.exists() and json.loads(path.read_text()).get("status") != "running":
            completed.append(task["name"])
        else:
            pending.append(task)
    active: dict[str, tuple[subprocess.Popen[bytes], dict[str, Any], Any]] = {}
    stopped = [False]

    def stop(signum: int, frame: Any) -> None:
        stopped[0] = True
        for process, _, _ in active.values():
            process.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    environment = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
    while pending or active:
        for name, (process, task, log) in list(active.items()):
            code = process.poll()
            if code is not None:
                log.close()
                del active[name]
                if code == 0 and Path(task["output"]).exists():
                    completed.append(name)
                else:
                    failed.append({"task": name, "returncode": code})
                report(args.output)
        if stopped[0] and not active:
            break
        if not stopped[0]:
            for task in list(pending):
                if "required_input" in task and not Path(task["required_input"]).exists():
                    continue
                if len(active) >= 2 or any(item[1]["kind"] == task["kind"] for item in active.values()):
                    continue
                if sum(item[1]["memory_mib"] for item in active.values()) + task["memory_mib"] > 44000:
                    continue
                log = (args.output / "logs" / f"{task['name']}.log").open("ab")
                process = subprocess.Popen(task["command"], stdout=log, stderr=subprocess.STDOUT, env=environment)
                active[task["name"]] = (process, task, log)
                pending.remove(task)
        write_json(args.output / "queue_status.json", {
            "updated": datetime.now().astimezone().isoformat(), "pid": os.getpid(), "gpu": 0,
            "status": "paused" if stopped[0] else "running" if pending or active else "finished",
            "completed": completed, "failed": failed, "cancelled_hybrid": cancelled,
            "pending": [task["name"] for task in pending],
            "active": {name: {"pid": process.pid, "kind": task["kind"], "reserved_mib": task["memory_mib"]}
                       for name, (process, task, _) in active.items()}})
        if pending or active:
            time.sleep(5)
    report(args.output)
    # Check absent checkpoints once more without silently replacing epoch 200.
    write_json(args.output / "missing_models_final_check.json", [
        {**entry, "now_available": Path(entry["checkpoint"]).exists()} for entry in unavailable])


if __name__ == "__main__":
    main()
