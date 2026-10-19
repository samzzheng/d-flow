"""Isolate the source-displacement cap in the five-step L-BFGS reconstruction."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

from experiments.ct_mri_sweep import run
from experiments.ct_mri_sweep.queue import source_hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=250)
    parser.add_argument("--problem", choices=("ct10", "mri4"), default="mri4")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = Path("outputs/separated_shapes_steps_20261019")
    path = root / "cap_0001_steps_5_rank_32/production/manifest.json"
    if not path.exists():
        manifest = deepcopy(json.loads((root / "steps_10_rank_32/production/manifest.json").read_text()))
        manifest["created_at"] = datetime.now().astimezone().isoformat()
        manifest["settings"]["flow_steps"] = 5
        manifest["settings"]["optimizer"]["step_cap"] = .001
        manifest["source_sha256"] = source_hashes()
        manifest["source_sha256"][str(Path(__file__).relative_to(run.ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        manifest["protocol"]["purpose"] = "Paired source-displacement cap control: 0.001*sqrt(d) instead of 0.01*sqrt(d); original history-100 LBFGS"
        manifest["protocol"]["selection"] = "Selected failed case; exploratory control, not held-out validation"
        run.write_json(path, manifest)
    sys.argv = [sys.argv[0], "--manifest", str(path), "--job", f"r{args.resolution}_shapes_{args.problem}",
                "--sample-start", str(args.sample), "--sample-stop", str(args.sample+1),
                "--device", args.device, "--memory-fraction", "0.20"]
    run.main()


if __name__ == "__main__":
    main()
