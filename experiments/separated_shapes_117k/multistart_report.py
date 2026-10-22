"""Validate shared restart budgets and report selected-source reconstruction results."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from experiments.ct_mri_sweep.run import METHODS, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    lines = ["# Measurement-selected restart controls", "",
        f"Updated {datetime.now().astimezone().isoformat(timespec='minutes')}.", "",
        "Four cached starts per method: original Gaussian, half/full measurement-adjoint mixtures, "
        "and an independent Gaussian. Each receives up to 50 probe attempts; the lowest measurement "
        "loss selects a continuation with the remaining shared 1000-attempt budget. "
        "The selected optimizer state is preserved. Image error never selects a source.", "",
        "These are selected failure cases, not a population estimate. Methods share the candidate "
        "pool but may choose different sources. Partial results are labeled.", "",
        "| Case | Run | Method | DLR rank / factor update / goal steps | Selected source | Status | Total attempts | Image error | Measurement residual |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |"]
    figures = []
    paired_pools = {}
    for root in args.roots:
        config = json.loads((root / "config.json").read_text())
        for case in config["cases"]:
            folder = root / case
            pool = torch.load(folder / "pool.pt", map_location="cpu", weights_only=True)
            source_hashes = {name: hashlib.sha256(value.numpy().tobytes()).hexdigest()
                             for name, value in pool["sources"].items()}
            assert source_hashes == pool["source_hashes"]
            signature = (pool["protocol"]["checkpoint_sha256"], source_hashes,
                hashlib.sha256(pool["truth"].numpy().tobytes()).hexdigest(),
                hashlib.sha256(pool["observed"].numpy().tobytes()).hexdigest(), pool["protocol"]["operator"])
            if case in paired_pools and signature != paired_pools[case]:
                raise ValueError(f"Unpaired candidate pools for {case}")
            paired_pools[case] = signature
            figure, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
            plotted = False
            for method in METHODS:
                if method not in config["methods"]:
                    continue
                stem = method.lower().replace("-", "_").replace(" ", "_")
                dlr = config["settings"]["dlr"]
                rank_label = (f"{dlr['rank_min']}–{dlr['rank_max']} adaptive" if dlr.get("adaptive_rank")
                              else str(dlr["rank"]))
                factor_label = (f"{rank_label} / {config.get('factor_integrator', 'fast')} / {config.get('goal_steps', 0)}"
                                if method == "DLR D-Flow" else "—")
                path = folder / f"{stem}.json"
                if not path.exists():
                    lines.append(f"| {case} | {root.name} | {method} | {factor_label} | — | Probing / pending | — | — | — |")
                    continue
                record = json.loads(path.read_text())
                protocol = record["protocol"]
                probes = {name: json.loads((folder / name / f"{stem}.json").read_text())
                          for name in pool["sources"]}
                for name, probe in probes.items():
                    assert probe["protocol"]["initial_source_sha256"] == source_hashes[name]
                    assert probe["protocol"]["checkpoint_sha256"] == pool["protocol"]["checkpoint_sha256"]
                    assert probe["history"][-1]["step"] <= config["probe_updates"]
                winner = min(probes, key=lambda name: probes[name]["history"][-1]["phi"])
                assert protocol["selected_candidate"] == winner
                assert protocol["initial_source_sha256"] == source_hashes[winner]
                probe_history = probes[winner]["history"]
                assert record["history"][:len(probe_history)] == probe_history
                spent = sum(probe["history"][-1]["step"] for probe in probes.values())
                assert protocol["probe_attempts_total"] == spent
                total = spent + record["history"][-1]["step"] - probe_history[-1]["step"]
                assert total <= config["budget"]
                terminal = record["status"] in {"stalled", "step_limit", "objective_plateau"}
                final = record["history"][-1]
                if terminal:
                    saved = torch.load(path.with_suffix(".pt"), map_location="cpu", weights_only=True)
                    image_error = (saved["image"].double()-pool["truth"].double()).norm()/pool["truth"].double().norm()
                    residual = ((saved["measurement"].to(torch.complex128)-pool["observed"].to(torch.complex128)).norm()
                                / pool["observed"].to(torch.complex128).norm())
                    torch.testing.assert_close(image_error, image_error.new_tensor(final["image_relative_l2"]),
                                               atol=1e-6, rtol=1e-5)
                    torch.testing.assert_close(residual, residual.new_tensor(final["measurement_relative_l2_noisy"]),
                                               atol=1e-6, rtol=1e-5)
                status = "Complete" if terminal else "Partial"
                lines.append(f"| {case} | {root.name} | {method} | {factor_label} | {winner} | {status} | {total} | "
                    f"{100*final['image_relative_l2']:.3f}% | {100*final['measurement_relative_l2_noisy']:.3f}% |")
                rows.append(dict(case=case, run=root.name, method=method, selected=winner, status=record["status"],
                    rank=config["settings"]["dlr"]["rank"] if method == "DLR D-Flow" else None,
                    adaptive_rank_min=dlr.get("rank_min") if method == "DLR D-Flow" and dlr.get("adaptive_rank") else None,
                    adaptive_rank_max=dlr.get("rank_max") if method == "DLR D-Flow" and dlr.get("adaptive_rank") else None,
                    factor_integrator=config.get("factor_integrator", "fast") if method == "DLR D-Flow" else None,
                    goal_steps=config.get("goal_steps", 0) if method == "DLR D-Flow" else None,
                    complete=terminal, total_attempts=total, image_error=final["image_relative_l2"],
                    residual=final["measurement_relative_l2_noisy"], source_hash=source_hashes[winner]))
                attempts, image_errors, residuals = [], [], []
                best = None
                offset = 0
                for probe in probes.values():
                    for item in probe["history"]:
                        if best is None or item["phi"] < best["phi"]:
                            best = item
                        attempts.append(offset + item["step"])
                        image_errors.append(100*best["image_relative_l2"])
                        residuals.append(100*best["measurement_relative_l2_noisy"])
                    offset += probe["history"][-1]["step"]
                for item in record["history"][len(probe_history):]:
                    attempts.append(spent + item["step"] - probe_history[-1]["step"])
                    image_errors.append(100*item["image_relative_l2"])
                    residuals.append(100*item["measurement_relative_l2_noisy"])
                label = method + (" (partial)" if not terminal else "")
                axes[0].plot(attempts, image_errors, label=label)
                axes[1].plot(attempts, residuals, label=label)
                plotted = True
            if plotted:
                for axis, ylabel in zip(axes, ("Relative image error (%)", "Measurement residual (%)")):
                    axis.set(xlabel="Total optimizer attempts across starts", ylabel=ylabel)
                    axis.legend(fontsize=8)
                figure.suptitle(f"{case}: {root.name}")
                filename = f"{root.name}_{case.replace('/', '_')}.png"
                figure.savefig(args.out / filename, dpi=140)
                figures.extend(["", f"![{case}]({filename})"])
            plt.close(figure)
    lines.extend(["", "Checks: cached source hashes, measurement-loss selection, preserved continuation history, "
        "and total attempt budgets. Completed metrics are recomputed from saved image/measurement arrays. "
        "Curves show the best measurement-loss candidate seen during probing, then the selected continuation. "
        "They do not select by image error. The factor-update column distinguishes the fast recompression, "
        "frozen-increment and two-pass refinement controls. Positive goal steps additionally change the "
        "projection criterion. These are distinguished from the notes' one-Euler-per-stage KSL core.", *figures])
    (args.out / "report.md").write_text("\n".join(lines)+"\n")
    write_json(args.out / "summary.json", dict(rows=rows, checks_passed=True))


if __name__ == "__main__":
    main()
