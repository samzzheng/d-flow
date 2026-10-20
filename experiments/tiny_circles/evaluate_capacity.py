"""Paired unconditional sampling audit after the 49k two-circle pilot finishes."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import cmocean
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import label
import torch

from dlr_dflow.api import load_flow_model
from sampler.ode_samplers import ode_integrate


def morphology(images: NDArray[np.float32]) -> dict[str, Any]:
    """Count substantial components; quantify disk shape without clipping."""
    records = []
    yy, xx = np.indices(images.shape[-2:])
    for image in images:
        counts, disk_ious = {}, []
        for threshold in (.25, .5, .75):
            labels, count = label(image > threshold, structure=np.ones((3, 3)))
            sizes = np.bincount(labels.ravel())
            components = [index for index in range(1, count + 1) if sizes[index] >= 8]
            counts[str(threshold)] = len(components)
            if threshold == .5:
                for index in components:
                    region = labels == index
                    cy, cx = yy[region].mean(), xx[region].mean()
                    radius = math.sqrt(float(region.sum()) / math.pi)
                    disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
                    disk_ious.append(float((disk & region).sum() / (disk | region).sum()))
        foreground = image > .75
        background = image < .25
        records.append(dict(counts=counts,
                            mean_disk_iou=float(np.mean(disk_ious)) if disk_ious else None,
                            minimum_disk_iou=min(disk_ious) if disk_ious else None,
                            foreground_rms_from_one=float(np.sqrt(np.mean((image[foreground] - 1) ** 2))) if foreground.any() else None,
                            background_rms=float(np.sqrt(np.mean(image[background] ** 2))) if background.any() else None,
                            minimum=float(image.min()), maximum=float(image.max())))
    minimum_ious = [r["minimum_disk_iou"] for r in records if r["minimum_disk_iou"] is not None]
    return dict(samples=len(records),
                two_component_fraction={str(t): sum(r["counts"][str(t)] == 2 for r in records) / len(records)
                                        for t in (.25, .5, .75)},
                count_histogram=dict(Counter(r["counts"]["0.5"] for r in records)),
                median_minimum_disk_iou=float(np.median(minimum_ious)) if minimum_ious else None,
                records=records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=Path(
        "outputs/fm_unet/circles-64-noattn-ch4-e500-mbtuned-20261019/n2"))
    parser.add_argument("--trainer-pid", type=int)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    out = args.run / "capacity_evaluation"
    out.mkdir(exist_ok=True)
    while not (args.run / "run_summary.json").exists():
        if args.trainer_pid is not None and not Path(f"/proc/{args.trainer_pid}").exists():
            (out / "status.json").write_text(json.dumps(dict(status="blocked", reason="Training exited before completion")))
            raise RuntimeError("Training exited before completion")
        (out / "status.json").write_text(json.dumps(dict(status="waiting_for_training", samples=args.samples,
                                                        seed=args.seed, trainer_pid=args.trainer_pid)))
        time.sleep(30)
    training = json.loads((args.run / "run_summary.json").read_text())
    assert training["status"] == "complete" and training["epochs_completed"] == 500
    torch.set_num_threads(4)
    torch.cuda.set_device(torch.device(args.device))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    initial = torch.randn(args.samples, 1, 64, 64, generator=torch.Generator().manual_seed(args.seed))
    dataset = torch.load("data/circles-64-n2.pt", weights_only=True, mmap=True)
    reference = dataset["test"].float().numpy()[:, 0] / 255
    result = dict(seed=args.seed, samples=args.samples, euler_steps=100, rk4_steps=100,
                  precision="FP32, TF32 disabled",
                  initial_noise_sha256=hashlib.sha256(initial.numpy().tobytes()).hexdigest(),
                  metrics_note="8-connected components with area>=8 pixels; thresholds0.25/0.5/0.75. Disk IoU uses centroid and equal-area disk at threshold0.5. Morphology proxies, not a complete generation-quality score.",
                  reference=morphology(reference), models={})
    samples = {}
    for name, run in (("22k", args.baseline), ("49k", args.run)):
        model, _ = load_flow_model(run_dir=run, epoch=500, device=args.device)
        config = json.loads((run / "config.json").read_text())
        result["models"][name] = dict(parameters=config["params"], run=str(run),
            checkpoint_sha256=hashlib.sha256((run / "checkpoints/epoch_0500.pt").read_bytes()).hexdigest())
        for method in ("euler", "rk4"):
            (out / "status.json").write_text(json.dumps(dict(status="running", model=name, method=method)))
            generated = []
            with torch.no_grad():
                for batch in initial.split(16):
                    image = ode_integrate(
                        lambda t, x: model(x, t.expand(x.shape[0])), batch.to(args.device),
                        ode_opts={"options": {"step_size": .01}}, sampler=method)
                    if not torch.isfinite(image).all():
                        raise FloatingPointError("Nonfinite unconditional sample")
                    generated.append(image.cpu())
            images = torch.cat(generated)
            samples[f"{name}_{method}"] = images
            result["models"][name][method] = morphology(images.numpy()[:, 0])
            (out / "results.json").write_text(json.dumps(result, indent=2))
        del model
        torch.cuda.empty_cache()
    torch.save(dict(initial=initial, generated=samples), out / "paired_samples.pt")
    order = ("22k_euler", "49k_euler", "22k_rk4", "49k_rk4")
    for start in range(0, min(args.samples, 16), 8):
        figure, axes = plt.subplots(4, min(8, args.samples - start), figsize=(15, 8), squeeze=False,
                                    layout="constrained")
        for row, name in enumerate(order):
            for column, ax in enumerate(axes[row]):
                ax.imshow(samples[name][start + column, 0], cmap=cmocean.cm.dense_r, vmin=0, vmax=1)
                ax.set_title(f"Source {start + column}", fontsize=9)
                if column == 0:
                    ax.set_ylabel(name.replace("_", " / "))
                ax.set_xticks([])
                ax.set_yticks([])
        figure.suptitle("64² two-circle models · 100 Euler / RK4 steps · paired initial noise\n"
                       "Fixed display range [0,1]; metrics use original values")
        figure.savefig(out / f"paired_samples_{start:03d}.png", dpi=140)
        plt.close(figure)
    (out / "status.json").write_text(json.dumps(dict(status="complete", samples=args.samples,
                                                    models=list(result["models"])), indent=2))


if __name__ == "__main__":
    main()
