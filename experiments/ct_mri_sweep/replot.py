"""Regenerate comparison panels from saved arrays and histories, without GPU work."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from scripts.reconstruction_figures import METHODS, RESIDUAL_VMAX, plot_reconstruction_comparison
from .run import Settings


def replot(case: Path) -> None:
    torch.set_num_threads(1)
    inputs = torch.load(case / "inputs.pt", map_location="cpu", weights_only=True)
    settings = json.loads((case.parents[1] / "manifest.json").read_text())["settings"]
    results: dict[str, Settings] = {}
    for method in METHODS:
        path = case / (method.lower().replace("-", "_").replace(" ", "_") + ".pt")
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            results[method] = {key: saved[key] for key in ("image", "measurement", "history", "status")}
    fig = plot_reconstruction_comparison(inputs["truth"], inputs["clean"], {
        name: (value["image"], value["measurement"]) for name, value in results.items()},
        observation=inputs["observed"], histories={name: value["history"] for name, value in results.items()},
        measurement_log=inputs["protocol"]["job"]["problem"] == "mri",
        residual_vmax=RESIDUAL_VMAX[inputs["protocol"]["job"]["problem"]],
        title=f"{case.parent.name} | test {inputs['protocol']['sample']} | {100*settings['noise_relative']:g}% noise | {settings['flow_steps']}-step Euler\n"
              + "; ".join(f"{name}: {value['status']} ({len(value['history'])-1} attempts)" for name, value in results.items()))
    for extension in ("png", "pdf"):
        fig.savefig(case / f"comparison.{extension}", dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    cases = sorted({case for path in args.paths for case in
                    ([path] if (path / "inputs.pt").exists() else map(lambda p: p.parent, path.glob("r*/sample_*/inputs.pt")))})
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for case, _ in zip(cases, pool.map(replot, cases)):
            print(case, flush=True)


if __name__ == "__main__":
    main()
