"""Sequential same-source terminal weighting controls on GPU1."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.ct_mri_sweep.run import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", default="r250_shapes_mri4/sample_002")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    scripts = [Path(__file__), Path(__file__).with_name("high_rank_reconstruction.py"),
               Path(__file__).with_name("spectral_weighting.py")]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in scripts}
    plan = dict(status="running", case=args.case, coordinator_pid=os.getpid(),
                script_hashes=hashes, runs=[])
    for variant in ("unchanged", "median_cap", "norm_matched_cap"):
        if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest for path, digest in hashes.items()):
            raise RuntimeError("Study source changed between paired variants")
        command = [sys.executable, "-u", "-m", "experiments.separated_shapes_117k.high_rank_reconstruction",
                   "--case", args.case, "--out", str(args.out/variant), "--ranks", "32",
                   "--memory-fraction", ".04", "--cudnn-deterministic", "--repeat-initial-gradient", "4",
                   "--terminal-weighting", variant]
        record = dict(variant=variant, command=command, started_at=time.time())
        plan["runs"].append(record)
        process = subprocess.Popen(command)
        record["pid"] = process.pid
        write_json(args.out/"queue.json", plan)
        record["returncode"] = process.wait()
        record["finished_at"] = time.time()
        if record["returncode"]:
            plan["status"] = "failed"
            write_json(args.out/"queue.json", plan)
            return
        result = json.loads((args.out/variant/args.case/"rank32/dlr_d_flow.json").read_text())
        record["result_status"] = result["status"]
        if result["status"] not in ("stalled", "step_limit", "objective_plateau"):
            plan["status"] = "paused"
            write_json(args.out/"queue.json", plan)
            return
        write_json(args.out/"queue.json", plan)
    plan["status"] = "complete"
    write_json(args.out/"queue.json", plan)


if __name__ == "__main__":
    main()
