#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


WEIGHT_LABELS = {
    0.0: "0",
    1e-6: "1e-6",
    3e-6: "3e-6",
    1e-5: "1e-5",
    3e-5: "3e-5",
    1e-4: "1e-4",
    3e-4: "3e-4",
    1e-3: "1e-3",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_result_set(path: Path, payload: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "dflow_blur_scalability_stats.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    write_csv(path / "dflow_blur_scalability_samples.csv", payload["samples"])
    write_csv(path / "dflow_blur_scalability_summary.csv", payload["summary"])


def canonical_weight_label(weight: float) -> str:
    for candidate, label in WEIGHT_LABELS.items():
        if abs(weight - candidate) <= max(1e-15, abs(candidate) * 1e-12):
            return label
    raise ValueError(f"Unexpected TV weight: {weight}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    aggregate_root = root / "aggregate_by_tv_weight"
    provenance_root = root / "provenance"

    if aggregate_root.exists() or provenance_root.exists():
        raise RuntimeError("Results already appear to be organized")

    merged: dict[float, tuple[Path, dict]] = {}
    for stats_path in sorted(root.glob("tv_*/dflow_blur_scalability_stats.json")):
        payload = json.loads(stats_path.read_text())
        weight = float(payload["config"]["tv_weight"])
        if weight in merged:
            raise RuntimeError(f"Duplicate merged result for TV weight {weight}")
        merged[weight] = (stats_path.parent, payload)

    if set(merged) != set(WEIGHT_LABELS):
        raise RuntimeError(f"Expected weights {sorted(WEIGHT_LABELS)}, found {sorted(merged)}")

    figure_paths: dict[tuple[float, int, int], str] = {}
    for weight, (_, payload) in sorted(merged.items()):
        label = canonical_weight_label(weight)
        for row in payload["samples"]:
            resolution = int(row["resolution"])
            sample_idx = int(row["sample_idx"])
            source = Path(row["figure_path"])
            destination_dir = root / f"res_{resolution}" / f"tv_{label}" / "figures"
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / source.name
            if not source.exists():
                raise FileNotFoundError(source)
            if destination.exists():
                raise FileExistsError(destination)
            shutil.move(str(source), str(destination))
            figure_paths[(weight, resolution, sample_idx)] = str(destination)

    aggregate_root.mkdir()
    for weight, (old_dir, payload) in sorted(merged.items()):
        label = canonical_weight_label(weight)
        for row in payload["samples"]:
            key = (weight, int(row["resolution"]), int(row["sample_idx"]))
            row["figure_path"] = figure_paths[key]

        aggregate_dir = aggregate_root / f"tv_{label}"
        payload["config"]["out_dir"] = str(aggregate_dir)
        write_result_set(aggregate_dir, payload)

        for resolution in sorted({int(row["resolution"]) for row in payload["samples"]}):
            result_dir = root / f"res_{resolution}" / f"tv_{label}"
            samples = [
                dict(row) for row in payload["samples"] if int(row["resolution"]) == resolution
            ]
            summary = [
                dict(row) for row in payload["summary"] if int(row["resolution"]) == resolution
            ]
            config = dict(payload["config"])
            config.update(
                {
                    "out_dir": str(result_dir),
                    "resolutions": [resolution],
                    "num_samples": len(samples),
                    "sample_indices": sorted(int(row["sample_idx"]) for row in samples),
                }
            )
            write_result_set(
                result_dir,
                {"config": config, "samples": samples, "summary": summary},
            )

        shutil.rmtree(old_dir)

    shards = root / "shards"
    for stats_path in sorted(shards.glob("*/dflow_blur_scalability_stats.json")):
        payload = json.loads(stats_path.read_text())
        weight = float(payload["config"]["tv_weight"])
        for row in payload["samples"]:
            key = (weight, int(row["resolution"]), int(row["sample_idx"]))
            row["figure_path"] = figure_paths[key]
        write_result_set(stats_path.parent, payload)

    provenance_root.mkdir()
    shutil.move(str(shards), str(provenance_root / "shards"))
    logs = root / "logs"
    if logs.exists():
        shutil.move(str(logs), str(provenance_root / "logs"))
    for name in (
        "gpu0_shard1_launcher.log",
        "gpu1_shard0_launcher.log",
        "supervisor.log",
    ):
        path = root / name
        if path.exists():
            shutil.move(str(path), str(provenance_root / name))

    manifest = {
        "layout": "resolution/tv_weight",
        "resolutions": [64, 96, 128, 160],
        "tv_weights": [WEIGHT_LABELS[weight] for weight in sorted(WEIGHT_LABELS)],
        "samples_per_resolution_weight": 5,
        "figures": len(figure_paths),
        "aggregate_results": str(aggregate_root),
        "raw_provenance": str(provenance_root),
    }
    (root / "layout.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Organized {len(figure_paths)} figures into resolution/TV-weight folders")


if __name__ == "__main__":
    main()
