"""Refresh completed figures on CPU and follow specified in-flight cases."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import time

from experiments.ct_mri_sweep.replot import replot
from experiments.ct_mri_sweep.run import TERMINAL, write_json


def complete(case: Path) -> bool:
    paths = [case / f"{stem}.json" for stem in ("d_flow", "p_flow", "dlr_d_flow")]
    return all(path.exists() and json.loads(path.read_text())["status"] in TERMINAL for path in paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--follow-cases", type=Path, nargs="*", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    cases = sorted({path.parent for root in args.roots for path in root.glob("r*/sample_*/inputs.pt")
                    if complete(path.parent)})
    following = set(args.follow_cases) - set(cases)
    done = []
    write_json(args.out / "plan.json", {"pid": os.getpid(), "cases": list(map(str, cases)),
               "following": sorted(map(str, following)), "workers": args.workers, "device": "cpu"})
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while cases or following:
            write_json(args.out / "status.json", {"status": "running", "pid": os.getpid(),
                "updated_at": datetime.now().astimezone().isoformat(), "completed": len(done),
                "pending": len(cases), "following": sorted(map(str, following))})
            for case, _ in zip(cases, pool.map(replot, cases)):
                done.append(str(case))
                if len(done) % 10 == 0:
                    write_json(args.out / "status.json", {"status": "running", "pid": os.getpid(),
                        "updated_at": datetime.now().astimezone().isoformat(), "completed": len(done),
                        "following": sorted(map(str, following))})
            cases = sorted(case for case in following if complete(case))
            following.difference_update(cases)
            if following and not cases:
                time.sleep(30)
    write_json(args.out / "status.json", {"status": "finished", "pid": os.getpid(),
        "updated_at": datetime.now().astimezone().isoformat(), "completed": len(done), "cases": done})


if __name__ == "__main__":
    main()
