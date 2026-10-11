"""One bounded GPU-0 adaptive worker alongside the existing studies."""
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
from .adaptive import VARIANTS


def report(output: Path) -> None:
    lines = ["# Rank-adaptive DLR pilot", "",
             "64px, two/three circles, CT10/MRI4, samples0–1. 100 Euler steps, "
             "FP32 without TF32, noiseless observations, up to100 optimizer attempts. "
             "Pure fixed8/fixed32 controls use the same batched-graph backend as adaptation.", "",
             "Rank is fixed during each factor evolution and optimizer step. Every five attempts "
             "and after a failed search, compare same-source gradients along4/8/16/32/64. "
             "The two tolerances are5% and1%; shrinking requires twice as close agreement. "
             "Clear LBFGS pairs and cached gradient only when the rank changes. "
             "Agreement is a heuristic; shared approximation bias can fool it. "
             "Rank64 is marked unvalidated if no comparison satisfies tolerance.", "",
             "| Case | Variant | Attempts | Final image error % | Measurement residual % | Rank min–max | Checks at unresolved cap | Status |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    paired = 0
    for case in sorted(output.glob("r*/sample_*")):
        items = []
        for variant in VARIANTS:
            path = case / f"{variant}.json"
            if not path.exists():
                continue
            item = json.loads(path.read_text())
            items.append(item)
            last = item["history"][-1]
            ranks = [row["rank"] for row in item["history"][1:]] or [last["rank"]]
            lines.append(f"| {case.parent.name}/{case.name} | {variant} | {last['step']} | "
                         f"{100 * last['image_relative_l2']:.3f} | {100 * last['measurement_relative_l2_clean']:.3f} | "
                         f"{min(ranks)}–{max(ranks)} | {sum(r['cap_unvalidated'] for r in item['rank_checks'])} | {item['status']} |")
        if len(items) == len(VARIANTS) and all(item["status"] in {"stalled", "step_limit"} for item in items):
            assert all(item["protocol"] == items[0]["protocol"] for item in items)
            paired += 1
    lines += ["", f"Completed four-way paired cases: {paired}/8. Running rows are not final comparisons.", "",
              "Errors use final saved checkpoints, not best-image selection. Histories record rank changes, "
              "curvature use, caps, failed searches, and factor-evaluation counts. "
              "No speed advantage or exact-gradient certification is implied by this pilot."]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        report(args.output)
        return
    (args.output / "logs").mkdir(exist_ok=True)
    cases = [f"r64_n{circles}_{problem}/sample_{sample:03d}"
             for sample in (0, 1) for circles in (3, 2) for problem in ("ct10", "mri4")]
    write_json(args.output / "plan.json", {"cases": cases, "variants": VARIANTS, "gpu": 0,
               "memory_cap_mib": 4000, "threads": 1, "flow_steps": 100, "optimizer_attempts": 100})
    active: subprocess.Popen[bytes] | None = None
    stopped = [False]

    def stop(signum: int, frame: Any) -> None:
        stopped[0] = True
        if active is not None and active.poll() is None:
            active.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    completed, failed = [], []
    for case in cases:
        if stopped[0]:
            break
        done = args.output / case / "complete.json"
        if done.exists():
            completed.append(case)
            continue
        with (args.output / "logs" / (case.replace("/", "_") + ".log")).open("ab") as log:
            active = subprocess.Popen([sys.executable, "-m", "experiments.multicircle_rank_hybrid.adaptive",
                                      "--output", str(args.output), "--case", case], stdout=log,
                                      stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1",
                                      "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
            while active.poll() is None:
                write_json(args.output / "queue_status.json", {"status": "running", "pid": os.getpid(),
                           "updated": datetime.now().astimezone().isoformat(), "gpu": 0,
                           "active": {"case": case, "pid": active.pid}, "completed": completed, "failed": failed})
                report(args.output)
                time.sleep(10)
            if active.returncode == 0 and done.exists():
                completed.append(case)
            elif not stopped[0]:
                failed.append({"case": case, "returncode": active.returncode})
    write_json(args.output / "queue_status.json", {"status": "paused" if stopped[0] else "finished",
               "updated": datetime.now().astimezone().isoformat(), "pid": os.getpid(), "gpu": 0,
               "completed": completed, "failed": failed, "active": None})
    report(args.output)


if __name__ == "__main__":
    main()
