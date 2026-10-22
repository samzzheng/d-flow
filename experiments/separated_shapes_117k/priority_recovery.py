"""Temporarily suspend one baseline while sequential paired controls run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    process = Path(f"/proc/{plan['baseline_pid']}/stat")
    assert process.read_text().rsplit(")", 1)[1].split()[19] == plan["baseline_start_time"]
    queues = [(Path(path), json.loads(Path(path).read_text())) for path in plan["queues"]]
    for _, configuration in queues:
        assert configuration["status"] == "queued"
        for path, expected in configuration["file_hashes"].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    child = None
    try:
        os.kill(plan["baseline_pid"], signal.SIGSTOP)
        for _ in range(100):
            if process.read_text().rsplit(")", 1)[1].split()[0] == "T":
                break
            time.sleep(.1)
        else:
            raise RuntimeError("Baseline did not enter the stopped state")
        plan.update(status="running", coordinator_pid=os.getpid(), suspended_at=time.time())
        args.plan.write_text(json.dumps(plan, indent=2)+"\n")
        for path, configuration in queues:
            configuration.update(status="running", scheduling="priority sequence with baseline suspended")
            path.write_text(json.dumps(configuration, indent=2)+"\n")
            with Path(configuration["log"]).open("a") as log:
                child = subprocess.Popen(configuration["command"], stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, "MPLCONFIGDIR": "/tmp/dflow-mpl", "UV_CACHE_DIR": "/tmp/dflow-uv-cache"})
                result = child.wait()
            configuration.update(status="complete" if result == 0 else "failed", returncode=result)
            path.write_text(json.dumps(configuration, indent=2)+"\n")
            child = None
            if result:
                raise RuntimeError(f"Recovery failed: {path}")
        plan["status"] = "complete"
    except BaseException:
        plan["status"] = "interrupted_or_failed"
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait()
        raise
    finally:
        if process.exists() and process.read_text().rsplit(")", 1)[1].split()[19] == plan["baseline_start_time"]:
            os.kill(plan["baseline_pid"], signal.SIGCONT)
            plan["baseline_resumed_at"] = time.time()
        args.plan.write_text(json.dumps(plan, indent=2)+"\n")


if __name__ == "__main__":
    main()
