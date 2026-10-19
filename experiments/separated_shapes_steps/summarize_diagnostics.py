"""Summarize fixed-source grid refinement separately from reconstruction accuracy."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    root = Path("outputs/separated_shapes_steps_20261019/diagnostics")
    data = json.loads((root / "summary.json").read_text())
    lines = ["# Fixed-source Euler-grid diagnostics", "",
             "Eight selected paired cases, original larger epoch-200 models, noiseless measurements. "
             "Sources are held fixed while the Euler grid changes: these are not reoptimized reconstructions. "
             "A recovered image norm does not establish recovered shape locations or accuracy.", "",
             "The five-step native replay exactly reproduces every saved final image in this audit. "
             "Several failed sources give nearly blank five-step images, but much larger images on finer grids. "
             "This implicates the discrete generator map; it does not establish that increasing steps resolves optimization failures. "
             "Gradient attenuation is also not universal: some failed sources have amplified exact gradients.", "",
             "| Case | Source | Steps | Image error % | Image norm / GT norm | Exact gradient / endpoint gradient norm | Exact-proxy cosine |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    figure, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    for case, axis in zip(data["cases"], axes.flat, strict=True):
        for label in ("initial", "D-Flow", "P-Flow", "DLR128"):
            rows = [row for row in case["sources"] if row["source"] == label]
            axis.plot([row["steps"] for row in rows], [row["image_norm_over_gt"] for row in rows],
                      marker="o", markersize=3, label=label)
            for row in rows:
                if row["steps"] in (5, 10, 20):
                    lines.append(f"| {case['case']} | {label} | {row['steps']} | "
                                 f"{100*row['image_relative_l2']:.3f} | {row['image_norm_over_gt']:.4f} | "
                                 f"{row['exact_over_endpoint_gradient_norm']:.4f} | {row['exact_proxy_cosine']:.4f} |")
        axis.axhline(1, color="gray", linestyle=":", linewidth=1)
        axis.set(title=case["case"].replace("_shapes", ""), xlabel="Euler steps", ylabel="Image norm / GT norm", xscale="log")
        axis.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    figure.suptitle("Fixed sources: image norm changes with Euler grid (not reoptimized accuracy)")
    figure.savefig(root / "fixed_source_norms.png", dpi=160)
    plt.close(figure)
    (root / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
