"""Audit target shape counts in the existing five-sample rank pilot."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from scipy.ndimage import label

from experiments.ct_mri_sweep.run import write_json


def main() -> None:
    base = Path("outputs/separated_shapes_noiseless_20261016/production")
    dlr_base = Path("outputs/separated_shapes_rank_sweep_20261017/rank_128/production")
    records = []
    for resolution in (64, 96, 128, 160, 200, 250):
        for sample in range(5):
            for problem in ("ct10", "mri4"):
                case = base / f"r{resolution}_shapes_{problem}" / f"sample_{sample:03d}"
                inputs = torch.load(case / "inputs.pt", weights_only=True, map_location="cpu")
                truth = inputs["truth"]
                count = label(truth.squeeze().numpy() > .5)[1]
                row = dict(resolution=resolution, problem=problem, sample=sample, count=count,
                           area=float((truth > .5).float().mean()))
                paths = {method: case / f"{method}.json" for method in ("d_flow", "p_flow")}
                paths["dlr128"] = dlr_base / case.parent.name / case.name / "dlr_d_flow.json"
                for method, path in paths.items():
                    row[method] = json.loads(path.read_text())["history"][-1]["image_relative_l2"]
                records.append(row)
    root = Path("outputs/separated_shapes_steps_20261019")
    write_json(root / "truth_morphology.json", records)
    lines = ["# Target-count audit of the existing rank pilot", "",
             "Connected components of the GT mask at intensity >0.5; first five test indices, six resolutions, CT/MRI. "
             "Only one- and three-object targets occur here: two- and four-object targets are not covered by this pilot. "
             "CT/MRI share truths, and resolution comparisons are paired. Counts below are acquisition cases, not independent geometries. "
             "The >20% threshold is a descriptive severe-error threshold, not a convergence criterion.", "",
             "| GT objects | Acquisition cases | D-Flow >20% error | P-Flow >20% error | DLR128 >20% error |",
             "| ---: | ---: | ---: | ---: | ---: |"]
    for count in range(1, 5):
        selected = [row for row in records if row["count"] == count]
        failures = [str(sum(row[method] > .2 for row in selected)) if selected else "n/a"
                    for method in ("d_flow", "p_flow", "dlr128")]
        lines.append(f"| {count} | {len(selected)} | {' | '.join(failures)} |")
    (root / "truth_morphology.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
