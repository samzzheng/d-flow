"""Generate shape datasets and train one model at a time on the selected GPU.

CPU generation overlaps GPU training. Completed runs are skipped, failed stages
are recorded, and existing unfinished outputs are never overwritten.
"""

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


def save_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now().astimezone().isoformat()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    models = parser.add_mutually_exclusive_group()
    models.add_argument("--deeper-model", action="store_true")
    models.add_argument("--model-117k", action="store_true")
    parser.add_argument("--resolutions", type=int, nargs="+", choices=(96, 128, 160, 200, 250),
                        default=[96, 128, 160, 200, 250])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--after-run", type=Path, help="Wait for this trainer and its final comparison to exit")
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/separated_shapes_multires_20261015"))
    args = parser.parse_args()
    epochs = args.epochs or (500 if args.deeper_model or args.model_117k else 200)
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    environment = {
        **os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "1", "TORCHINDUCTOR_COMPILE_THREADS": "2",
        "TQDM_MININTERVAL": "30", "MPLBACKEND": "Agg",
    }
    state: dict[str, Any] = {"pid": os.getpid(), "device": args.device, "status": "running", "jobs": []}
    for resolution in dict.fromkeys(args.resolutions):
        dataset = Path(f"data/separated-shapes-{resolution}.pt")
        architecture = "117k-time32" if args.model_117k else "53k-time32" if args.deeper_model else "noattn-ch16-b8"
        tag = f"-{args.run_tag}" if args.run_tag else ""
        run = Path(f"outputs/fm_unet/separated-shapes-{resolution}-{architecture}-e{epochs}{tag}")
        ready = dataset.with_suffix(".json")
        summary = run / "run_summary.json"
        dataset_complete = False
        if dataset.exists() and ready.exists():
            metadata = json.loads(ready.read_text())
            dataset_complete = (
                metadata["resolution"] == resolution and metadata["supersample"] == 8
                and metadata["split_sizes"] == {"train": 80_000, "val": 10_000, "test": 10_000}
            )
        training_complete = False
        if summary.exists():
            result = json.loads(summary.read_text())
            training_complete = result["status"] == "complete" and result["epochs_completed"] == epochs
        # Existing live training belongs to another controller: do not duplicate it.
        launch = run / "launch.json"
        if launch.exists() and not training_complete:
            pid = json.loads(launch.read_text())["pid"]
            command_line = Path(f"/proc/{pid}/cmdline")
            if command_line.exists() and b"train_separated_shapes.py" in command_line.read_bytes():
                raise RuntimeError(f"Existing live training for {resolution}: PID {pid}")
        state["jobs"].append({
            "resolution": resolution, "dataset": str(dataset), "run": str(run),
            "generate": "complete" if dataset_complete else "pending",
            "train": "complete" if training_complete else "pending",
            "generate_command": [sys.executable, "-u", "-m", "dataset.generate_separated_shapes",
                                 "--resolution", str(resolution), "--workers", "4"],
            "train_command": [sys.executable, "-u", "scripts/train_separated_shapes.py",
                              "--resolution", str(resolution), "--device", args.device,
                              "--epochs", str(epochs), "--out-dir", str(run)]
                              + (["--model-117k"] if args.model_117k else ["--deeper-model"] if args.deeper_model else []),
            "generate_log": str(args.output / f"generate_{resolution}.log"),
            "train_log": str(args.output / f"train_{resolution}.log"),
        })
    state_path = args.output / "status.json"
    active: dict[str, tuple[subprocess.Popen[bytes], dict[str, Any]]] = {}
    save_state(state_path, state)
    if args.after_run is not None:
        state.update(status="waiting_for_previous_run", after_run=str(args.after_run))
        save_state(state_path, state)
        pid = json.loads((args.after_run / "launch.json").read_text())["pid"]
        while True:
            command_line = Path(f"/proc/{pid}/cmdline")
            if not command_line.exists() or b"train_separated_shapes.py" not in command_line.read_bytes():
                break
            time.sleep(10)
        summary = args.after_run / "run_summary.json"
        if not summary.exists() or json.loads(summary.read_text())["status"] != "complete":
            state["status"] = "blocked_previous_training_incomplete"
            save_state(state_path, state)
            raise RuntimeError("Previous trainer exited without completing; queued training not started")
        state["status"] = "running"
        save_state(state_path, state)
    while True:
        changed = False
        for stage, (process, job) in list(active.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            complete = returncode == 0
            if complete and stage == "train":
                summary = Path(job["run"]) / "run_summary.json"
                complete = summary.exists()
                if complete:
                    result = json.loads(summary.read_text())
                    complete = result["status"] == "complete" and result["epochs_completed"] == epochs
            if complete and stage == "generate":
                complete = Path(job["dataset"]).with_suffix(".json").exists()
            job[stage] = "complete" if complete else "failed"
            job[f"{stage}_returncode"] = returncode
            job[f"{stage}_finished_at"] = datetime.now().astimezone().isoformat()
            if stage == "generate" and not complete:
                job["train"] = "blocked"
            print(f"{stage} {job['resolution']}: {job[stage]}", flush=True)
            del active[stage]
            changed = True
        for stage in ("generate", "train"):
            if stage in active:
                continue
            job = next((item for item in state["jobs"] if item[stage] == "pending"
                        and (stage == "generate" or item["generate"] == "complete")), None)
            if job is None:
                continue
            with Path(job[f"{stage}_log"]).open("ab") as log:
                process = subprocess.Popen(job[f"{stage}_command"], cwd=root, env=environment,
                                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            job[stage] = "running"
            job[f"{stage}_pid"] = process.pid
            job[f"{stage}_started_at"] = datetime.now().astimezone().isoformat()
            active[stage] = (process, job)
            print(f"{stage} {job['resolution']}: PID {process.pid}", flush=True)
            changed = True
        if not active and all(item[stage] not in ("pending", "running")
                              for item in state["jobs"] for stage in ("generate", "train")):
            state["status"] = "complete" if all(item["train"] == "complete" for item in state["jobs"]) else "finished_with_failures"
            save_state(state_path, state)
            break
        if changed:
            save_state(state_path, state)
        time.sleep(10)


if __name__ == "__main__":
    main()
