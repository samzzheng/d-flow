#!/usr/bin/env python3
"""Organize the completed 750-step TV/LBFGS sweep by TV weight."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


ROOT = Path("outputs/dflow_tv_lr_comparison_750_steps")
BASELINE_ROOT = Path("outputs/dflow_tv_coarse_sweep_no_line_search_750_steps")
DEST_ROOT = ROOT / "by_tv_weight"

WEIGHTS = {
    "tv_0": ("tv_0", "0"),
    "tv_1e-6": ("tv_1e-6", "1e-6"),
    "tv_3e-6": ("tv_3e-6", "3e-6"),
    "tv_1e-5": ("tv_1e-5", "1e-5"),
    "tv_3e-5": ("tv_3e-5", "3e-5"),
    "tv_1e-4": ("tv_0.0001", "1e-4"),
    "tv_3e-4": ("tv_0.0003", "3e-4"),
    "tv_1e-3": ("tv_0.001", "1e-3"),
}
LEARNING_RATES = ("0.1", "0.3", "1.0")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def update_result_paths(result_dir: Path) -> None:
    samples_path = result_dir / "dflow_blur_scalability_samples.csv"
    fields, rows = read_csv(samples_path)
    for row in rows:
        row["figure_path"] = str(
            result_dir
            / "figures"
            / f"res_{int(float(row['resolution']))}"
            / f"n1_sample_{int(float(row['sample_idx'])):03d}.png"
        )
    write_csv(samples_path, fields, rows)

    stats_path = result_dir / "dflow_blur_scalability_stats.json"
    stats = json.loads(stats_path.read_text())
    stats["config"]["out_dir"] = str(result_dir)
    stats["config"].pop("shards", None)
    if result_dir.name == "lr_1.0":
        stats["config"]["device"] = "cuda:0,cuda:1"
    for sample in stats["samples"]:
        sample["figure_path"] = str(
            result_dir
            / "figures"
            / f"res_{int(sample['resolution'])}"
            / f"n1_sample_{int(sample['sample_idx']):03d}.png"
        )
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")


def validate_result(result_dir: Path) -> None:
    _, rows = read_csv(result_dir / "dflow_blur_scalability_samples.csv")
    keys = {(int(float(row["resolution"])), int(float(row["sample_idx"]))) for row in rows}
    expected = {(resolution, sample) for resolution in (64, 96, 128, 160) for sample in range(5)}
    figures = list((result_dir / "figures").glob("res_*/n1_sample_*.png"))
    if len(rows) != 20 or keys != expected or len(figures) != 20:
        raise RuntimeError(
            f"Incomplete result {result_dir}: rows={len(rows)}, "
            f"keys={len(keys)}, figures={len(figures)}"
        )


def move_log(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))


def main() -> None:
    if DEST_ROOT.exists():
        raise RuntimeError(f"Destination already exists: {DEST_ROOT}")
    DEST_ROOT.mkdir(parents=True)

    global_samples: list[dict[str, object]] = []
    global_summaries: list[dict[str, object]] = []
    sample_fields: list[str] | None = None
    summary_fields: list[str] | None = None

    for weight_label, (baseline_label, weight_value) in WEIGHTS.items():
        weight_dir = DEST_ROOT / weight_label
        weight_dir.mkdir()

        for learning_rate in ("0.1", "0.3"):
            source = ROOT / f"lr_{learning_rate}" / weight_label
            destination = weight_dir / f"lr_{learning_rate}"
            if not source.exists():
                raise RuntimeError(f"Missing source result: {source}")
            shutil.move(str(source), str(destination))
            move_log(
                ROOT / f"lr_{learning_rate}" / "logs" / f"{weight_label}.log",
                destination / "run.log",
            )
            update_result_paths(destination)
            validate_result(destination)

        baseline_source = BASELINE_ROOT / baseline_label
        baseline_destination = weight_dir / "lr_1.0"
        if not baseline_source.exists():
            raise RuntimeError(f"Missing baseline result: {baseline_source}")
        shutil.move(str(baseline_source), str(baseline_destination))

        figure_destination = baseline_destination / "figures"
        for shard_id in (0, 1):
            shard_figures = (
                BASELINE_ROOT
                / "shards"
                / f"{weight_label}_shard_{shard_id}"
                / "figures"
            )
            if not shard_figures.exists():
                raise RuntimeError(f"Missing baseline shard figures: {shard_figures}")
            for figure in shard_figures.glob("res_*/n1_sample_*.png"):
                destination = figure_destination / figure.parent.name / figure.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    raise RuntimeError(f"Duplicate figure destination: {destination}")
                shutil.move(str(figure), str(destination))

        update_result_paths(baseline_destination)
        validate_result(baseline_destination)

        weight_samples: list[dict[str, object]] = []
        weight_summaries: list[dict[str, object]] = []
        for learning_rate in LEARNING_RATES:
            result_dir = weight_dir / f"lr_{learning_rate}"
            current_sample_fields, samples = read_csv(
                result_dir / "dflow_blur_scalability_samples.csv"
            )
            current_summary_fields, summaries = read_csv(
                result_dir / "dflow_blur_scalability_summary.csv"
            )
            sample_fields = sample_fields or current_sample_fields
            summary_fields = summary_fields or current_summary_fields
            for row in samples:
                compiled = {
                    "lbfgs_lr": learning_rate,
                    "tv_weight": weight_value,
                    **row,
                }
                weight_samples.append(compiled)
                global_samples.append(compiled)
            for row in summaries:
                compiled = {
                    "lbfgs_lr": learning_rate,
                    "tv_weight": weight_value,
                    **row,
                }
                weight_summaries.append(compiled)
                global_summaries.append(compiled)

        write_csv(
            weight_dir / "all_learning_rates_samples.csv",
            ["lbfgs_lr", "tv_weight", *(sample_fields or [])],
            weight_samples,
        )
        write_csv(
            weight_dir / "all_learning_rates_summary.csv",
            ["lbfgs_lr", "tv_weight", *(summary_fields or [])],
            weight_summaries,
        )

    write_csv(
        DEST_ROOT / "all_tv_weights_and_learning_rates_samples.csv",
        ["lbfgs_lr", "tv_weight", *(sample_fields or [])],
        global_samples,
    )
    write_csv(
        DEST_ROOT / "all_tv_weights_and_learning_rates_summary.csv",
        ["lbfgs_lr", "tv_weight", *(summary_fields or [])],
        global_summaries,
    )

    logs_dir = DEST_ROOT / "_launcher_logs"
    move_log(ROOT / "lr_0.1_launcher.log", logs_dir / "lr_0.1_launcher.log")
    move_log(ROOT / "lr_0.3_launcher.log", logs_dir / "lr_0.3_launcher.log")
    move_log(ROOT / "launcher.log", logs_dir / "comparison_launcher.log")
    move_log(BASELINE_ROOT / "launcher.log", logs_dir / "lr_1.0_launcher.log")

    # These are redundant only after every aggregate and figure set has validated.
    shutil.rmtree(BASELINE_ROOT / "shards")
    shutil.rmtree(BASELINE_ROOT / "logs")
    for shard_launcher in (
        BASELINE_ROOT / "gpu0_shard1_launcher.log",
        BASELINE_ROOT / "gpu1_shard0_launcher.log",
    ):
        shard_launcher.unlink()

    for obsolete_dir in (
        ROOT / "lr_0.1" / "logs",
        ROOT / "lr_0.3" / "logs",
        ROOT / "lr_0.1",
        ROOT / "lr_0.3",
        BASELINE_ROOT,
    ):
        obsolete_dir.rmdir()

    if len(global_samples) != 480 or len(global_summaries) != 96:
        raise RuntimeError(
            f"Unexpected compiled counts: samples={len(global_samples)}, "
            f"summaries={len(global_summaries)}"
        )

    print(f"Organized 480 samples into {DEST_ROOT}")


if __name__ == "__main__":
    main()
