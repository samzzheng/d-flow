"""Summarize completed 64-pixel shape capacity audits without rerunning models."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "outputs/fm_unet"
    folders = {
        "53k": root / "separated-shapes-64-53k-time32-e500-20261020",
        "117k": root / "separated-shapes-64-117k-time32-e500-20261020",
        "200k": root / "separated-shapes-64-200k-time32-e500-20261020",
    }
    models: dict[str, dict] = {}
    flags: dict[str, dict[str, np.ndarray]] = {}
    source_hashes: set[str] = set()
    for name, folder in folders.items():
        audit = folder / "capacity_epoch_0500"
        comparison = json.loads((audit / "baseline_comparison.json").read_text())
        templates = json.loads((audit / "template_quality.json").read_text())
        counts = json.loads((audit / "shape_quality.json").read_text())
        source_hashes.add(comparison["source_sha256"])
        if comparison["sample_count"] != 1024:
            raise ValueError(f"Expected 1024-source final audit: {audit}")
        for label, key in ((name, "small"), ("original", "baseline")):
            models[label] = dict(comparison["models"][key], generation={})
            flags[label] = {}
            for steps in ("5", "100"):
                generated = f"{key}_{steps}"
                count = counts["generated"][generated]["thresholds"]["0.5"]
                outliers = templates["generated"][generated]["outlier_images"]
                models[label]["generation"][steps] = {
                    "samples": 1024,
                    "count_outside_one_to_four": count["component_count_histogram"]["0"]
                    + count["more_than_four_components"],
                    "template_outlier_images": len(outliers),
                    "four_components": count["component_count_histogram"]["4"],
                }
                flags[label][steps] = np.zeros(1024)
                flags[label][steps][outliers] = 1
    if len(source_hashes) != 1:
        raise ValueError("Final audits must use identical source hashes")
    paired: dict[str, dict] = {}
    for steps in ("5", "100"):
        difference = flags["117k"][steps] - flags["200k"][steps]
        rng = np.random.default_rng(20261020)
        bootstrap = np.concatenate([
            difference[rng.integers(0, 1024, (1000, 1024))].mean(axis=1)
            for _ in range(10)
        ])
        paired[steps] = {
            "difference_percentage_points": 100 * float(difference.mean()),
            "paired_bootstrap_95_percent_interval": (
                100 * np.quantile(bootstrap, [0.025, 0.975])
            ).tolist(),
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 20261020,
        }
    destination = folders["200k"]
    report = {
        "models": models,
        "source_sha256": next(iter(source_hashes)),
        "source_seed": 4271,
        "template_iou_cutoff": templates["cutoff"],
        "real_reference_outlier_images": templates["reference_images_with_outlier"],
        "real_reference_images": templates["reference_images"],
        "paired_117k_minus_200k_template_outliers": paired,
        "recommendation": "200k is the smallest tested model retaining approximately original quality; 117k is a smaller, lower-quality alternative.",
        "limitation": "No global minimum parameter count or formal quality equivalence is established. Template flags are not certified invalid shapes. One training seed per architecture; original trained 200 epochs, others 500. No inverse-reconstruction comparison.",
    }
    (destination / "capacity_summary.json").write_text(json.dumps(report, indent=2) + "\n")

    order = ["53k", "117k", "200k", "original"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
    labels = [f'{models[name]["parameters"]:,}' for name in order]
    axes[0].bar(labels, [models[name]["test_fm_mse"] for name in order], color="#657b8b")
    axes[0].set_ylabel("Paired test FM MSE")
    axes[0].set_xlabel("Parameters")
    for steps, marker in (("100", "o"), ("5", "s")):
        axes[1].plot(labels, [100 * models[name]["generation"][steps]["template_outlier_images"] / 1024
                             for name in order], marker=marker, label=f"{steps}-step Euler")
    axes[1].set_ylabel("Images with template outlier (%)")
    axes[1].set_xlabel("Parameters")
    axes[1].legend()
    fig.savefig(destination / "capacity_comparison.png", dpi=160)
    plt.close(fig)
    rows = []
    for name in order:
        model = models[name]
        gen = model["generation"]["100"]
        rows.append(f'<tr><td>{model["parameters"]:,}</td><td>{model["epochs"]}</td>'
                    f'<td>{model["test_fm_mse"]:.6f}</td>'
                    f'<td>{gen["template_outlier_images"]}/1024 ({100 * gen["template_outlier_images"] / 1024:.1f}%)</td>'
                    f'<td>{gen["count_outside_one_to_four"]}/1024</td></tr>')
    interval = paired["100"]["paired_bootstrap_95_percent_interval"]
    html = f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><title>64² separated shapes model capacity — 2026-10-20</title>
<style>body{{font:16px/1.5 system-ui;margin:2rem auto;max-width:1000px;padding:0 1rem;color:#222}}
table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;padding:.6rem;border-bottom:1px solid #ddd}}img{{max-width:100%}}code{{font-size:.9em}}</style>
<h1>64² separated shapes model capacity</h1><p>2026-10-20</p>
<p><strong>Recommend the 205,793-parameter model:</strong> it uses 56.4% fewer parameters than the original 472,017-parameter model, with essentially matched held-out FM loss and similar shape diagnostics. This is the smallest tested model retaining approximately original quality, not a proven minimum over architectures.</p>
<p>The 116,617-parameter model converged and uses 75.3% fewer parameters than the original, but its test loss is 3.4% higher and its 100-step shape-template outlier rate is higher. Keep it as a size/quality tradeoff, rather than calling it equivalent.</p>
<ul><li>64² separated disks and triangles; 1–4 objects; mandatory 8×8 antialiasing; 80k/10k/10k train/validation/test.</li>
<li>No attention. Compact models use four levels with channel multipliers 1/2/3/3 and 32 sinusoidal time features; hidden time-MLP widths scale with base width.</li>
<li>Compact models: 500 epochs, batch/microbatch 512, AdamW LR 0.0002, weight decay 0.0001, cosine schedule, EMA 0.999; GPU1. Original reference: 200 epochs.</li>
<li>Full held-out FM evaluation; 1,024 identical Gaussian sources across models, seed 4271; FP32 Euler generation with TF32 disabled.</li></ul>
<h2>Final quality at 100 Euler steps</h2>
<table><thead><tr><th>Parameters</th><th>Epochs</th><th>Test FM MSE</th><th>Template-outlier images</th><th>Count outside 1–4</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<img src="capacity_comparison.png" alt="Model size versus test FM loss and template-outlier rate">
<p>The paired 117k-minus-200k outlier difference is {paired['100']['difference_percentage_points']:.2f} percentage points; the 95% paired bootstrap interval is [{interval[0]:.2f}, {interval[1]:.2f}]. This interval measures source-sampling variability for these trained checkpoints, not training-seed variability.</p>
<h2>Interpretation and limits</h2>
<p>Template flags compare thresholded components against real training shapes. The cutoff is calibrated on held-out real components and flags 26/1024 real reference images. Flags are not certified invalid shapes and do not test spacing, intensity, or inverse reconstruction. Detected counts alone cannot certify valid geometry. Both 117k and 200k reached nearly flat validation loss late in training. The 53k model retained a much larger shape-fidelity deficit; the rejected 100k run was removed as requested.</p>
<p>All retained compact runs have 500-epoch histories, loss plots, ten checkpoints and ten previews. Final 117k and 200k checkpoints load through the public model loader and produce finite outputs.</p>
<h2>Artifacts</h2><ul>
<li><a href="capacity_summary.json">Machine-readable results and checkpoint hashes</a></li>
<li><a href="eval/epoch_0500.png">200k final training preview</a> · <a href="loss_curves.png">200k loss curves</a></li>
<li><a href="../{folders['117k'].name}/eval/epoch_0500.png">117k final training preview</a> · <a href="../{folders['117k'].name}/loss_curves.png">117k loss curves</a></li>
<li><a href="capacity_epoch_0500/eval/capacity_comparison_100steps.png">Original versus 200k, first eight paired sources</a></li>
<li><a href="../{folders['117k'].name}/capacity_epoch_0500/eval/capacity_comparison_100steps.png">Original versus 117k, first eight paired sources</a></li></ul>
<p>Rebuild with <code>uv run --no-sync --python .venv/bin/python python scripts/report_shapes_capacity.py</code>.</p></html>'''
    (destination / "capacity_report.html").write_text(html)
    print(destination / "capacity_report.html")


if __name__ == "__main__":
    main()
