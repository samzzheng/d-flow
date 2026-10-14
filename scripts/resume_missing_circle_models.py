"""Finish incomplete epoch-200 circle models sequentially on GPU 0."""
from __future__ import annotations

from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "outputs/missing_circle_training_20261014"
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    runs = []
    for resolution in (200, 250, 160, 128, 96, 64):
        for circles in (4, 3, 2, 1):
            run = root / f"outputs/fm_unet/circles-{resolution}-noattn-ch16-b8-e200/n{circles}"
            summary = run / "run_summary.json"
            if summary.exists() and json.loads(summary.read_text())["status"] == "complete":
                continue
            config = json.loads((run / "config.json").read_text())
            checkpoint = max((run / "checkpoints").glob("epoch_*.pt"))
            epoch = int(checkpoint.stem.split("_")[1])
            arch = config["arch"]
            command = [sys.executable, "-u", "train_fm_unet.py", "--data", config["data"],
                "--out-dir", str(run), "--epochs", str(config["epochs"]),
                "--batch", str(config["batch"]), "--micro-batch", str(config["micro_batch"]),
                "--lr", str(config["lr"]), "--wd", str(config["wd"]),
                "--ema-decay", str(config["ema_decay"]), "--t-alpha", str(config["t_alpha"]),
                "--sample-steps", str(config["checkpoint_sampling"]["steps"]),
                "--ch", str(arch["ch"]), "--ch-mul", ",".join(map(str, arch["ch_mul"])),
                "--att-channels", ",".join(map(str, arch["att_channels"])),
                "--groups", str(arch["groups"]), "--mid-attn" if arch["mid_attn"] else "--no-mid-attn",
                "--resume-checkpoint", str(checkpoint), "--start-epoch", str(epoch),
                "--device", "cuda:0", "--threads", "4"]
            runs.append({"resolution": resolution, "circles": circles, "command": command,
                         "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                         "original_config": config, "status": "pending"})
    state = {"pid": os.getpid(), "gpu": 0, "runs": runs,
             "optimizer_resume": "Legacy EMA checkpoints have no AdamW state; existing resume procedure reinitializes it.",
             "training_source_sha256": hashlib.sha256((root / "train_fm_unet.py").read_bytes()).hexdigest()}
    status = out / "status.json"
    status.write_text(json.dumps(state, indent=2) + "\n")
    for item in runs:
        item.update(status="running", started_at=datetime.now().astimezone().isoformat())
        status.write_text(json.dumps(state, indent=2) + "\n")
        run = root / f"outputs/fm_unet/circles-{item['resolution']}-noattn-ch16-b8-e200/n{item['circles']}"
        with (run / "train.log").open("ab") as log:
            process = subprocess.run(item["command"], cwd=root, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                     "OPENBLAS_NUM_THREADS": "1", "TQDM_MININTERVAL": "60"}, check=False)
        item.update(status="complete" if process.returncode == 0 else "failed",
                    returncode=process.returncode, finished_at=datetime.now().astimezone().isoformat())
        status.write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()
