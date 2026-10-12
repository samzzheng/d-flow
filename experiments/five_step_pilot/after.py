"""Start a frozen sweep only after every shard of its prerequisite completes."""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any

from experiments.ct_mri_sweep.queue import tasks
from experiments.ct_mri_sweep.run import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    args = parser.parse_args()
    out, prerequisite = args.out.resolve(), args.after.resolve()
    manifest_path = prerequisite / "production/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    markers = [task.marker(manifest_path.parent) for task in tasks(manifest["settings"])]
    lock = (out / "dependency.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid()))
    lock.flush()
    stopped = [False]

    def stop(signum: int, frame: Any) -> None:
        stopped[0] = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while not stopped[0]:
        completed = sum(path.exists() for path in markers)
        previous = json.loads((prerequisite / "queue_status.json").read_text())
        ready = completed == len(markers) and previous["status"] == "finished" and not previous.get("failures")
        write_json(out / "queue_status.json", {"status": "starting" if ready else "queued_after_prerequisite",
            "pid": os.getpid(), "gpu": 1, "active": {}, "updated_at": datetime.now().astimezone().isoformat(),
            "prerequisite": str(prerequisite), "prerequisite_completed_shards": completed,
            "prerequisite_total_shards": len(markers), "prerequisite_status": previous["status"]})
        if ready:
            write_json(out / "dependency_satisfied.json", {"at": datetime.now().astimezone().isoformat(),
                "prerequisite": str(prerequisite), "completed_shards": completed,
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()})
            lock.close()
            os.execv(sys.executable, [sys.executable, "-m", "experiments.five_step_pilot.run", "run", "--out", str(out)])
        time.sleep(30)
    write_json(out / "queue_status.json", {"status": "paused", "pid": os.getpid(), "gpu": 1,
                                          "prerequisite": str(prerequisite)})
    lock.close()


if __name__ == "__main__":
    main()
