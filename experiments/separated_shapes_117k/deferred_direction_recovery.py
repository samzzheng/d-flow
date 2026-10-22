"""Run a recorded recovery after its same-sample process and controls finish."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue", type=Path)
    args = parser.parse_args()
    configuration = json.loads(args.queue.read_text())
    process = Path(f"/proc/{configuration['dependency_pid']}/stat")
    while process.exists():
        try:
            state = process.read_text().rsplit(")", 1)[1].split()
        except FileNotFoundError:
            break
        if state[19] != configuration["dependency_start_time"] or state[0] == "Z":
            break
        time.sleep(30)
    for path in configuration["completed_summaries"]:
        summary = json.loads(Path(path).read_text())["float64"]
        if summary["status"] not in {"stalled", "step_limit"}:
            raise RuntimeError(f"Dependency is incomplete: {path}")
        if summary["status"] == "step_limit" and summary["total_attempts"] != 1000:
            raise RuntimeError(f"Dependency did not reach its recorded budget: {path}")
    for path, expected in configuration["file_hashes"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Queued input or implementation changed: {path}")
    configuration.update(status="running", started=time.time(), queue_pid=os.getpid())
    args.queue.write_text(json.dumps(configuration, indent=2)+"\n")
    with Path(configuration["log"]).open("a") as log:
        result = subprocess.run(configuration["command"], stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "MPLCONFIGDIR": "/tmp/dflow-mpl", "UV_CACHE_DIR": "/tmp/dflow-uv-cache"})
    configuration.update(status="complete" if result.returncode == 0 else "failed",
                         returncode=result.returncode, finished=time.time())
    args.queue.write_text(json.dumps(configuration, indent=2)+"\n")
    result.check_returncode()


if __name__ == "__main__":
    main()
