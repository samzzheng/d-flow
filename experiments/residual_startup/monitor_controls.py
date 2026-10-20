"""Refresh saved initialization-control reports until both worker queues exit."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

from experiments.ct_mri_sweep.run import write_json
from experiments.residual_startup.measurement_controls import report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--queue-pids", type=int, nargs="+", required=True)
    args = parser.parse_args()
    while True:
        report(args.out)
        summary = json.loads((args.out / "summary.json").read_text())
        alive = [pid for pid in args.queue_pids if Path(f"/proc/{pid}").exists()]
        status = "running"
        if summary["completed"] == summary["expected"]:
            status = "complete"
        elif not alive:
            status = "incomplete"
        write_json(args.out / "queue_status.json", dict(
            status=status, completed=summary["completed"], expected=summary["expected"],
            queue_pids=alive, updated_at=datetime.now().astimezone().isoformat(),
            gpu=0, samples=[20, 22, 9, 14]))
        if status != "running":
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
