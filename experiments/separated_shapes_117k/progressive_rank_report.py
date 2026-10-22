"""Report live or completed high-start rank controls with explicit partial labels."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from scripts.reconstruction_figures import RESIDUAL_VMAX, plot_reconstruction_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--baseline", type=Path, default=Path(
        "outputs/separated_shapes_117k_noiseless_20261021/production"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    cases = sorted(args.root.glob("r*/sample_*"))
    fig, axes = plt.subplots(2, len(cases), figsize=(6*len(cases), 8), squeeze=False,
                             constrained_layout=True)
    rows = []
    completed_panels = []
    lines = ["# High-start rank controls: latest saved evidence", "",
        "Paired five-Euler-step, noiseless, FP32 controls with maximum rank 256, floor 32, analytical alpha, zero enrichment and unchanged LBFGS/geometric Armijo. Each sample runs its variants sequentially; pending variants are not results. The launch report documents the early graph-cache OOM and state-preserving resume.", "",
        "| Case | Policy | Status | Attempts | Image error | Measurement residual | First terminal rank | Last recorded terminal rank |",
        "|---|---|---|---:|---:|---:|---:|---:|"]
    colors = {"unrestricted256": "tab:blue", "retain_initial256": "tab:orange", "gradual256": "tab:green"}
    for column, case in enumerate(cases):
        name = str(case.relative_to(args.root))
        for variant, color in colors.items():
            path = case/variant/"dlr_d_flow.json"
            if not path.exists():
                lines.append(f"| {name} | {variant} | Pending | — | — | — | — | — |")
                continue
            saved = json.loads(path.read_text())
            records = list(map(json.loads, (path.parent/"rank_history.jsonl").read_text().splitlines()))
            # A resumed worker may repeat evaluations since its last checkpoint;
            # use the latest record for each evaluation number, not both branches.
            latest = {record["evaluation"]: record for record in records}
            records = [latest[index] for index in sorted(latest)]
            history = saved["history"]
            row = dict(case=name, variant=variant, status=saved["status"],
                attempt=history[-1]["step"], image_error=100*history[-1]["image_relative_l2"],
                residual=100*history[-1]["measurement_relative_l2_noisy"],
                first_terminal_rank=records[0]["ranks"][-1]["rank"],
                last_terminal_rank=records[-1]["ranks"][-1]["rank"],
                last_rank_path=[x["rank"] for x in records[-1]["ranks"][1:]],
                last_rank_objective=records[-1]["objective"],
                initial_rank_below_cap_evaluations=sum(record["ranks"][1]["rank"] < 256 for record in records),
                rank_evaluations=len(records),
                terminal_at_cap_fraction=sum(record["ranks"][-1]["rank"] == 256 for record in records)/len(records))
            rows.append(row)
            if saved["status"] in {"stalled", "step_limit", "objective_plateau"}:
                inputs = torch.load(args.baseline/name/"inputs.pt", map_location="cpu", weights_only=True, mmap=True)
                config = saved["protocol"]["experiment"]
                assert hashlib.sha256(inputs["initial"].numpy().tobytes()).hexdigest() == config["initial_source_sha256"]
                assert torch.equal(inputs["observed"], inputs["clean"])
                reconstructions = {}
                for method, method_path in (("D-Flow", args.baseline/name/"d_flow.pt"),
                                            ("P-Flow", args.baseline/name/"p_flow.pt"),
                                            ("DLR D-Flow", path.with_suffix(".pt"))):
                    if not method_path.exists():
                        continue
                    result = torch.load(method_path, map_location="cpu", weights_only=True, mmap=True)
                    if result["status"] not in {"stalled", "step_limit", "objective_plateau"}:
                        continue
                    for key, value in inputs["protocol"].items():
                        assert result["protocol"][key] == value
                    actual = float((result["image"].double()-inputs["truth"].double()).norm()/inputs["truth"].double().norm())
                    assert abs(actual-result["history"][-1]["image_relative_l2"]) < 1e-6
                    reconstructions[method] = (result["image"], result["measurement"])
                comparison = plot_reconstruction_comparison(inputs["truth"], inputs["clean"], reconstructions,
                    observation=inputs["observed"], include_errors=True,
                    measurement_log=inputs["protocol"]["job"]["problem"] == "mri",
                    residual_vmax=RESIDUAL_VMAX[inputs["protocol"]["job"]["problem"]],
                    title=f"{name}: {variant}, noiseless; completed paired methods")
                comparison.savefig(path.parent/"paired_methods.png", dpi=100)
                plt.close(comparison)
                completed_panels.append(f"[{name}, {variant}]({name}/{variant}/paired_methods.png)")
            lines.append(f"| {name} | {variant} | {row['status']} | {row['attempt']} | {row['image_error']:.3f}% | {row['residual']:.3f}% | {row['first_terminal_rank']} | {row['last_terminal_rank']} |")
            axes[0, column].semilogy([x["step"] for x in history],
                [100*x["image_relative_l2"] for x in history], color=color, label=variant)
            axes[1, column].plot([x["evaluation"] for x in records],
                [x["ranks"][-1]["rank"] for x in records], color=color, label=variant)
        axes[0, column].set(title=name, xlabel="Optimization attempts", ylabel="Relative image error (%)", ylim=(.1, 150))
        axes[1, column].set(xlabel="Gradient evaluation number", ylabel="Terminal retained rank", ylim=(0, 265))
        axes[1, column].axhline(256, color="gray", linestyle="--", linewidth=.8)
        for axis in axes[:, column]:
            axis.legend(fontsize=8)
            axis.grid(alpha=.2)
    for row in rows:
        if row["variant"] == "unrestricted256":
            lines += ["", f"{row['case']}: unrestricted initialization retained fewer than 256 directions in {row['initial_rank_below_cap_evaluations']}/{row['rank_evaluations']} recorded evaluations. Where it already retains 256, the initialization-only floor changes no retained rank. Differences from that control cannot automatically be attributed to a rank-policy improvement; floating-point execution variability and trajectory divergence remain possible."]
    if (args.root/"ct7_historical_comparison.json").exists():
        lines += ["", "The earlier CT rank-128 result is a paired-problem historical reference, not a strict rank-cap-only control: derivative-kernel and graph execution settings also differ. [Verified scope and configuration differences](ct7_historical_comparison.json)."]
    lines += ["", "Completed paired reconstruction panels: "+(" · ".join(completed_panels) or "None yet."), "",
        "Rank is measured on the approximated residual core with relative cutoff 0.001, not on a certified exact Jacobian. A small initial terminal rank does not imply that the same rank remains sufficient at later optimized sources. Retaining nearly the full cap indicates limited compression under this criterion, not proof that the true Jacobian requires that many directions.", "",
        "Rank and image curves have different horizontal axes: gradient evaluations versus optimizer attempts. The last rank record may precede the latest accepted source; they should not be interpreted as a fixed-point pair. Duplicated evaluation numbers after checkpoint recovery use the most recent record. All curves are partial until their saved status is terminal.", "",
        "The gradual policy limits shrinking inside each flow pass. Rank can rise across optimizer iterations because each new gradient initializes a new maximum-rank sketch. Accuracy comparisons between policies require completed paired runs or common completed prefixes; no such comparison is claimed while other policies are pending.", "",
        "![Reconstruction errors and terminal ranks](latest_progress.png)", "",
        "[Launch and recovery protocol](report.md) · [Independent saved-array verification](live_verification.json)"]
    fig.savefig(args.root/"latest_progress.png", dpi=140)
    plt.close(fig)
    (args.root/"latest_progress.json").write_text(json.dumps(rows, indent=2)+"\n")
    (args.root/"latest_progress.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
