"""Summarize the paired adaptive-rank pilot without treating partial runs as final."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--fixed-control-roots", nargs="*", type=Path, default=[])
    args = parser.parse_args()
    config = json.loads((args.root / "reconstruct_config.json").read_text())
    adaptive = config["adaptive"]
    controls = []
    for root in args.fixed_control_roots:
        control = json.loads((root / "reconstruct_config.json").read_text())
        bounds = control["adaptive"]
        if bounds["rank_min"] != bounds["rank_max"]:
            raise ValueError(f"Expected a fixed-rank control: {root}")
        controls.append((root, control, f"fixed{bounds['rank_min']}_canonical"))
    lines = [
        "# Separated-shapes adaptive-rank pilot",
        f"Updated {datetime.now().astimezone().isoformat(timespec='minutes')}.",
        "",
        "Paired noiseless CT10/MRI4× cases; 117k models, five Euler steps, "
        "analytical alpha=1-t, native forward normalization, zero enrichment, "
        "and the same LBFGS/Armijo settings and 1000-attempt limit.",
        f"Adaptive bounds: {adaptive['rank_min']}–{adaptive['rank_max']}; "
        f"retain core singular values at least {adaptive['rank_tol']:g} times the largest, "
        "subject to these bounds. This cutoff is not a bound on gradient error.",
        "The maximum-rank initialization sketch is rebuilt at each gradient evaluation. "
        "With zero enrichment, rank can only decrease within one flow solve; "
        "it can increase again at the next gradient evaluation.",
        "Fixed-rank canonical controls use equal rank bounds in the adaptive code path. "
        "They help distinguish rank selection from changes in factor arithmetic and execution.",
        "",
        "Errors below use the last saved source. Partial results are explicitly marked; "
        "these selected cases do not estimate population performance.",
        "",
        "| Case | Variant | Status | Attempts | Relative image error | Clean measurement error |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    rank_sections: list[str] = []
    completed = 0
    total = 0
    for case in config["cases"]:
        paired_keys: set[tuple[str, str]] = set()
        histories = {}
        folders = [(variant, args.root / variant / case) for variant in config["variants"]]
        folders.extend((label, root / "adaptive" / case) for root, control, label in controls
                       if case in control["cases"])
        total += len(folders)
        for variant, folder in folders:
            path = folder / "dlr_d_flow.json"
            if not path.exists():
                lines.append(f"| {case} | {variant} | Pending | — | — | — |")
                continue
            record = json.loads(path.read_text())
            histories[variant] = record
            protocol = record["protocol"]
            paired_keys.add((protocol["initial_source_sha256"], protocol["checkpoint_sha256"]))
            final = record["history"][-1]
            terminal = record["status"] in {"stalled", "step_limit", "objective_plateau"}
            completed += int(terminal)
            status = f"Complete ({record['status']})" if terminal else f"Partial ({record['status']})"
            lines.append(
                f"| {case} | {variant} | {status} | {final['step']} | "
                f"{100*final['image_relative_l2']:.3f}% | "
                f"{100*final['measurement_relative_l2_clean']:.3f}% |")
        if len(paired_keys) > 1:
            raise ValueError(f"Unpaired sources or checkpoints in {case}")
        rank_path = args.root / "adaptive" / case / "rank_history.jsonl"
        if not rank_path.exists():
            continue
        # Read only newline-terminated records while the worker may be appending.
        contents = rank_path.read_text()
        records = list(map(json.loads, contents[:contents.rfind('\n') + 1].splitlines()))
        if not records:
            continue
        figure, axes = plt.subplots(1, 3, figsize=(14, 3.5), constrained_layout=True)
        for index in sorted({0, len(records) // 2, len(records) - 1}):
            record = records[index]
            axes[0].plot([item["flow_step"] for item in record["ranks"]],
                         [item["rank"] for item in record["ranks"]], marker="o",
                         label=f"Gradient evaluation {record['evaluation']}")
        axes[0].set(xlabel="Euler step", ylabel="Residual rank", ylim=(0, adaptive["rank_max"] * 1.05))
        axes[0].legend(fontsize=8)
        axes[1].plot([record["evaluation"] for record in records],
                     [record["ranks"][-1]["rank"] for record in records])
        axes[1].set(xlabel="Gradient evaluation", ylabel="Terminal rank",
                    ylim=(0, adaptive["rank_max"] * 1.05))
        for variant, record in histories.items():
            label = variant + (" (partial)" if record["status"] not in {"stalled", "step_limit", "objective_plateau"} else "")
            axes[2].plot([item["step"] for item in record["history"]],
                         [100 * item["image_relative_l2"] for item in record["history"]], label=label)
        axes[2].set(xlabel="Optimizer attempt", ylabel="Relative image error (%)")
        axes[2].legend(fontsize=8)
        figure.suptitle(case)
        filename = f"ranks_{case.replace('/', '_')}.png"
        figure.savefig(args.root / filename, dpi=140)
        plt.close(figure)
        rank_sections.extend(["", f"## {case}", "",
            "Last recorded flow ranks: " + " → ".join(str(item["rank"]) for item in records[-1]["ranks"]) + ".",
            f"![Rank history]({filename})"])
    lines.extend(["", f"Completed {completed}/{total} method runs.",
                  "All available variants within each case have matching initial-source and checkpoint hashes.",
                  *rank_sections])
    (args.root / "report.md").write_text("\n".join(lines) + "\n")
    print(args.root / "report.md")


if __name__ == "__main__":
    main()
