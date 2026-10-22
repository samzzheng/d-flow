"""Summarize selected-source objective geometry without treating it as a trial."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    assert data["status"] == "complete"
    lines = ["# Objective geometry at failed sources", "", data["selection"], "",
        "All cases use the same 250² MRI4× acquisition and five-step generator as their successful paired P-Flow endpoint. The diagnostic promotes the saved model/source to FP64 while preserving the original FP32 time features. No reconstruction updates or new optimizer are tested.", "",
        "| Failed method | Sample | Starting image error | Exact gradient norm | Largest tested negative-gradient displacement that decreases loss | Maximum sampled bridge rise | Target image error |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    derivative_errors = []
    for row, ax in zip(data["rows"], axes.flat, strict=True):
        method = "D-Flow" if Path(row["failed_source"]).name == "d_flow.pt" else "DLR D-Flow"
        assert row["saved_endpoint_relative_drift"] < 1e-5
        assert row["path"][-1]["image_error_percent"] < 10
        derivative_errors += [x["slope_error_over_gradient_norm"] for x in row["finite_differences"] if x["length"] == 1e-8]
        probes = row["probes"]+row["finite_differences"]
        decreasing = [p["length"] for p in probes if p["direction"] == "negative_exact" and p["actual_change"] < 0]
        largest = max(decreasing) if decreasing else None
        rise = 100*(max(p["phi"] for p in row["path"])/row["initial_phi"]-1)
        lines.append(f"| {method} | {row['sample']} | {row['path'][0]['image_error_percent']:.3f}% | {row['exact_gradient_norm']:.4g} | {largest:.0e} | {rise:.4f}% | {row['path'][-1]['image_error_percent']:.3f}% |")
        ax.plot([p["fraction"] for p in row["path"]], [p["phi"]/row["initial_phi"] for p in row["path"]], "o-", markersize=3)
        ax.axhline(1, color="gray", linestyle="--", linewidth=.7)
        ax.set(xlabel="Fraction toward successful P-Flow source", ylabel="Objective / starting objective",
               title=f"{method}, sample {row['sample']}", ylim=(0, 1.03))
        inset = ax.inset_axes([.55, .48, .4, .32])
        early = [p for p in row["path"] if 0 < p["fraction"] <= .2]
        inset.semilogx([p["fraction"] for p in early], [100*(p["phi"]/row["initial_phi"]-1) for p in early], "o-", markersize=2)
        inset.axhline(0, color="gray", linestyle="--", linewidth=.7)
        inset.set_title("Early objective change (%)", fontsize=7)
        inset.tick_params(labelsize=6)
    assert max(derivative_errors) < 1e-4
    fig.suptitle("Selected failed endpoints: direct paths toward known successful sources")
    fig.savefig(args.results.with_name("source_bridge.png"), dpi=150)
    plt.close(fig)
    lines += ["", "The displacement column is the largest decreasing step among the explicitly sampled lengths, not a certified line-search limit. The bridge rise is relative to each starting objective. A rise along this straight path does not exclude another descending path.", "",
        f"Native gradient slopes agree with central differences at displacement 1e-8: maximum absolute slope error divided by exact gradient norm is {max(derivative_errors):.3g}. Saved FP32 endpoint recomputation agrees; promoting to FP64 changes endpoint values by up to {100*max(r['fp64_endpoint_relative_change'] for r in data['rows']):.3f}% relative to the original endpoint. All promoted successful targets remain below 10% image error.", "",
        "## Interpretation", "",
        "The failed endpoints have nonzero exact gradients. A first-order descending direction need not decrease the objective at a larger finite displacement: higher-order curvature can dominate. The observed sign changes at very small displacements support a narrow local descent region. They do not establish a global stationary point or a specific layer as the cause.", "",
        "All sampled bridges rise above their starting objective before reaching a successful endpoint. The same discrete generator can represent useful images, but optimization from a failed endpoint encounters unfavorable local geometry. P-Flow's earlier successful trajectory does not imply that its proxy direction is a descent direction at the failed source; several measured exact/proxy cosines are negative.", "",
        "For DLR this separates two problems: missing useful directions and navigating the actual objective even when a direction is accurate. More rank can address the first without fixing the second. A blanket terminal-rank increase is therefore not a sufficient correction. Existing source-initialization and preserved-state optimizer controls remain the relevant reconstruction tests; this diagnostic does not justify a new default or relaxed acceptance criterion.", "",
        "![Source paths and early objective changes](source_bridge.png)", "",
        f"[Full measurements]({args.results.name})"]
    args.results.with_name("source_bridge_report.md").write_text("\n".join(lines)+"\n")
    print(f"Verified {len(data['rows'])} selected endpoints; maximum normalized slope discrepancy {max(derivative_errors):.3g}.")


if __name__ == "__main__":
    main()
