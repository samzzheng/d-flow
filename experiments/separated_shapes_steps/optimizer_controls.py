"""Paired five-step control for shorter L-BFGS history."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

from experiments.ct_mri_sweep.queue import source_hashes
from experiments.ct_mri_sweep.run import write_json
from experiments.separated_shapes_steps import run as queue


def report(out: Path) -> None:
    plan = json.loads((out / "plan.json").read_text())
    rows = []
    lines = ["# Five-step optimizer controls", "",
             "Two selected failed cases, unchanged original models, source/acquisition seeds, five Euler steps and rank 32. "
             "History five gives short-history LBFGS, compared with the existing history-100 baseline. "
             "All retain geometric halving, first-attempt scaling, the 0.01*sqrt(d) cap and a 1000-attempt budget. "
             "Settings are exploratory, not independently validated. Timings include GPU sharing.", "",
             "| Optimizer | Case | Method | Status | Attempts | Image error % | Measurement error % | Seconds |",
             "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for task in plan["tasks"]:
        case = Path(task["manifest"]).parent / task["job"] / f"sample_{task['sample']:03d}"
        for stem in queue.STEMS:
            path = case / f"{stem}.json"
            if not path.exists():
                continue
            result = json.loads(path.read_text())
            last = result["history"][-1]
            row = dict(variant=task["variant"], case=f"{task['job']}/{case.name}", method=result["method"],
                       status=result["status"], attempts=last["step"], image_error_percent=100*last["image_relative_l2"],
                       measurement_error_percent=100*last["measurement_relative_l2_clean"], seconds=result["elapsed_seconds"])
            rows.append(row)
            lines.append(f"| {row['variant']} | {row['case']} | {row['method']} | {row['status']} | {row['attempts']} | "
                         f"{row['image_error_percent']:.3f} | {row['measurement_error_percent']:.3f} | {row['seconds']:.1f} |")
    write_json(out / "results.json", rows)
    (out / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "run", "report"))
    parser.add_argument("--out", type=Path, default=queue.DEFAULT / "optimizer_controls")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.action == "init":
        template = json.loads((queue.DEFAULT / "steps_10_rank_32/production/manifest.json").read_text())
        hashes = source_hashes()
        for path in (Path(__file__), Path(queue.__file__)):
            hashes[str(path.relative_to(queue.ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        tasks = []
        for variant, history, lr in (("lbfgs_h5", 5, 1.),):
            manifest = deepcopy(template)
            manifest["created_at"] = datetime.now().astimezone().isoformat()
            manifest["source_sha256"] = hashes
            manifest["settings"]["flow_steps"] = 5
            manifest["settings"]["optimizer"].update(history_size=history, lr=lr)
            manifest["protocol"]["optimizer_control"] = variant
            manifest["protocol"]["selection"] = "Two failed cases; tuning diagnostics, not a held-out benchmark"
            path = out / variant / "production/manifest.json"
            if path.exists():
                raise FileExistsError(path)
            write_json(path, manifest)
            tasks.extend(dict(manifest=str(path), job=job, sample=sample, steps=5, variant=variant)
                         for job, sample in (("r64_shapes_mri4", 1), ("r128_shapes_ct10", 0)))
        write_json(out / "plan.json", dict(tasks=tasks, code_sha256=hashes))
    elif args.action == "report":
        report(out)
    else:
        queue.report = report
        sys.argv = [sys.argv[0], "run", "--out", str(out), "--workers", "1", "--device", "cuda:0", "--memory-fraction", ".15"]
        queue.main()


if __name__ == "__main__":
    main()
