"""Compare recorded solver wall times on paired five/100-step samples."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import torch

from experiments.ct_mri_sweep.run import TERMINAL, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    production = args.out / "production"
    new_settings = json.loads((production / "manifest.json").read_text())["settings"]
    old_settings = json.loads((args.baseline / "manifest.json").read_text())["settings"]
    for key in ("noise_relative", "optimizer", "dlr", "dlr_reuse_previous_v", "source_projection", "execution"):
        assert new_settings[key] == old_settings[key], key
    assert new_settings["flow_steps"] == 5 and old_settings["flow_steps"] == 100
    rows = []
    for case in sorted(production.glob("r*/sample_000")):
        baseline = args.baseline / case.relative_to(production)
        if not (baseline / "inputs.pt").exists():
            continue
        new_inputs = torch.load(case / "inputs.pt", weights_only=True, map_location="cpu")
        old_inputs = torch.load(baseline / "inputs.pt", weights_only=True, map_location="cpu")
        # The subset hashes differ because manifests select one vs thirty images.
        # Verify the actual paired arrays instead.
        for key in ("truth", "initial", "clean", "observed", "mask"):
            left, right = new_inputs[key], old_inputs[key]
            assert (left is None and right is None) or torch.equal(left, right), (case, key)
        for path in sorted(case.glob("*.json")):
            if not (baseline / path.name).exists():
                continue
            new = json.loads(path.read_text())
            old = json.loads((baseline / path.name).read_text())
            assert new["status"] in TERMINAL and old["status"] in TERMINAL
            for key in ("checkpoint_sha256", "job", "sample", "operator", "source_seed", "noise_seed", "dlr_seed"):
                assert new["protocol"][key] == old["protocol"][key], (path, key)
            row = {"case": case.parent.name, "method": new["method"]}
            for label, record in (("five", new), ("hundred", old)):
                attempts = record["history"][-1]["step"]
                row[label] = {"attempts": attempts, "seconds": record["elapsed_seconds"],
                    "seconds_per_attempt": record["elapsed_seconds"] / attempts,
                    "line_search_trials": sum(item.get("line_search_trials", 0) for item in record["history"]),
                    "status": record["status"], "final_image_percent": 100 * record["history"][-1]["image_relative_l2"]}
            rows.append(row)
    lines = ["# Recorded five-step versus 100-step wall times", "",
             "Paired sample0 at64/128/250, CT10 and MRI4, noiseless. Identical saved truths, "
             "sources, masks and observations; checkpoint hashes and numerical settings verified. "
             "One-vs-thirty-sample dataset hashes differ, so pairing was checked from saved tensors.", "",
             "These are solver wall times including Armijo, metrics and intermediate checkpointing. "
             "They exclude model loading, backend validation, final figure writing and queue waits. "
             "GPU contention differed; optimizer trajectories and line-search counts also differ. "
             "This is not an isolated ODE benchmark or time-to-equal-accuracy comparison.", "",
             "| Method | 5 steps: s/attempt | 100 steps: s/attempt | Observed ratio | 5 steps: total min | 100 steps: total min |",
             "| --- | --- | --- | --- | --- | --- |"]
    aggregate = {}
    for method in ("D-Flow", "P-Flow", "DLR D-Flow"):
        paired = [row for row in rows if row["method"] == method]
        result = {label: {"seconds": sum(row[label]["seconds"] for row in paired),
                          "attempts": sum(row[label]["attempts"] for row in paired)}
                  for label in ("five", "hundred")}
        for value in result.values():
            value["seconds_per_attempt"] = value["seconds"] / value["attempts"]
        ratio = result["hundred"]["seconds_per_attempt"] / result["five"]["seconds_per_attempt"]
        aggregate[method] = {**result, "observed_per_attempt_ratio": ratio}
        lines.append(f"| {method} | {result['five']['seconds_per_attempt']:.3f} | "
                     f"{result['hundred']['seconds_per_attempt']:.3f} | {ratio:.1f} | "
                     f"{result['five']['seconds']/60:.2f} | {result['hundred']['seconds']/60:.2f} |")
    launch = json.loads((args.out / "launch.json").read_text())
    status = json.loads((args.out / "queue_status.json").read_text())
    elapsed = None
    if status["status"] == "finished":
        elapsed = (datetime.fromisoformat(status["updated_at"]) - datetime.fromisoformat(launch["at"])).total_seconds()
        lines += ["", f"Entire six-resolution five-step pilot, including smoke/overhead: {elapsed/60:.2f} elapsed minutes."]
    write_json(args.out / "timing_comparison.json", {"paired_runs": rows, "aggregates": aggregate,
               "pilot_elapsed_seconds": elapsed})
    (args.out / "timing_comparison.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
