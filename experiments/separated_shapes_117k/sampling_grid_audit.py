"""Compare Euler time grids on cached unconditional sources before inverse tests."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label
import torch

from dlr_dflow.api import load_flow_model
from experiments.ct_mri_sweep.run import save_tensor, write_json
from experiments.separated_shapes_117k.flow_grid_audit import euler_grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda:1")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.08, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    metadata = json.loads((args.model / "baseline_comparison.json").read_text())
    checkpoint = args.model / "checkpoints/epoch_0500.pt"
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == metadata["models"]["small"]["checkpoint_sha256"]
    sources = torch.load(args.model / "eval/capacity_comparison_sources.pt", map_location="cpu", weights_only=True)
    assert hashlib.sha256(sources.numpy().tobytes()).hexdigest() == metadata["source_sha256"]
    if not 1 <= args.samples <= sources.shape[0]:
        parser.error("Sample count must fit the cached source pool")
    saved = torch.load(args.model / "eval/capacity_comparison_samples.pt", map_location="cpu", weights_only=True)
    model, _ = load_flow_model(args.resolution, 0, 500, str(device), args.model)
    grids = dict(uniform5=(0., .2, .4, .6, .8, 1.),
                 moderate5=(0., .3, .55, .75, .9, 1.),
                 end_clustered5=(0., .4, .7, .85, .95, 1.),
                 uniform10=tuple(index/10 for index in range(11)),
                 uniform20=tuple(index/20 for index in range(21)))
    images = {"saved_uniform100": saved["small_100"][:args.samples]}
    summaries = {}
    args.out.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for name, grid in grids.items():
            images[name] = torch.cat([euler_grid(model, batch.to(device), grid).cpu()
                                     for batch in sources[:args.samples].split(4)])
        for name, values in images.items():
            counts = []
            for value in values[:, 0].numpy():
                components, _ = label(value >= .5, structure=np.ones((3, 3)))
                sizes = np.bincount(components.reshape(-1))[1:]
                counts.append(int((sizes >= 4).sum()))
            summaries[name] = dict(samples=len(counts), component_counts=counts,
                histogram={str(count): counts.count(count) for count in sorted(set(counts))},
                empty_fraction=counts.count(0)/len(counts),
                mean_foreground_fraction=float((values >= .5).float().mean()),
                outside_unit_interval_rms=float((values-values.clamp(0, 1)).square().mean().sqrt()))
    summaries["uniform5"]["relative_difference_from_saved"] = float(
        (images["uniform5"]-saved["small_5"][:args.samples]).norm()/saved["small_5"][:args.samples].norm())
    report = dict(checkpoint_sha256=metadata["models"]["small"]["checkpoint_sha256"],
        source_sha256=metadata["source_sha256"], grids=grids, summaries=summaries,
        note="Paired unconditional sources. Component counts do not certify geometry or inverse accuracy.")
    write_json(args.out / "summary.json", report)
    save_tensor(args.out / "samples.pt", images)
    figure, axes = plt.subplots(min(4, args.samples), len(images), figsize=(15, 10), squeeze=False,
                                constrained_layout=True)
    for column, (name, values) in enumerate(images.items()):
        for row in range(axes.shape[0]):
            axes[row, column].imshow(values[row, 0], cmap="gray", vmin=0, vmax=1)
            axes[row, column].set_title(f"{name}\n{summaries[name]['component_counts'][row]} components", fontsize=9)
            axes[row, column].set_axis_off()
    figure.savefig(args.out / "samples.png", dpi=130)
    plt.close(figure)
    print(json.dumps({name: value["histogram"] for name, value in summaries.items()}), flush=True)


if __name__ == "__main__":
    main()
