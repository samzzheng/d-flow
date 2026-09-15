#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

from benchmark_dflow_blur_scalability import SampleResult, summarize, write_csv


def weight_label(weight: float) -> str:
    return f"{weight:g}".replace("e-0", "e-").replace("e+0", "e+")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-samples-per-weight", type=int, default=20)
    args = parser.parse_args()

    grouped: dict[float, list[tuple[Path, dict]]] = {}
    for path in sorted((args.root / "shards").glob("*/dflow_blur_scalability_stats.json")):
        payload = json.loads(path.read_text())
        weight = float(payload["config"]["tv_weight"])
        grouped.setdefault(weight, []).append((path, payload))

    if not grouped:
        raise RuntimeError(f"No completed shard statistics found under {args.root / 'shards'}")

    result_fields = {field.name for field in fields(SampleResult)}
    for weight, shards in sorted(grouped.items()):
        samples: dict[tuple[int, int], dict] = {}
        for path, payload in shards:
            for row in payload["samples"]:
                key = (int(row["resolution"]), int(row["sample_idx"]))
                if key in samples:
                    raise RuntimeError(f"Duplicate sample {key} for TV weight {weight}: {path}")
                samples[key] = row

        if len(samples) != args.expected_samples_per_weight:
            raise RuntimeError(
                f"TV weight {weight} has {len(samples)} samples; "
                f"expected {args.expected_samples_per_weight}"
            )

        rows = [SampleResult(**{key: row[key] for key in result_fields}) for _, row in sorted(samples.items())]
        summary = summarize(rows)
        config = dict(shards[0][1]["config"])
        output_dir = args.root / f"tv_{weight_label(weight)}"
        config.update(
            {
                "out_dir": str(output_dir),
                "num_samples": len({row.sample_idx for row in rows}),
                "sample_indices": sorted({row.sample_idx for row in rows}),
                "shards": [str(path.parent) for path, _ in shards],
                "total_wall_s": max(float(payload["config"]["total_wall_s"]) for _, payload in shards),
            }
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output = {
            "config": config,
            "samples": [row.__dict__ for row in rows],
            "summary": summary,
        }
        (output_dir / "dflow_blur_scalability_stats.json").write_text(
            json.dumps(output, indent=2) + "\n"
        )
        write_csv(
            output_dir / "dflow_blur_scalability_samples.csv",
            [row.__dict__ for row in rows],
        )
        write_csv(output_dir / "dflow_blur_scalability_summary.csv", summary)
        print(f"Merged TV weight {weight:g}: {len(rows)} samples")


if __name__ == "__main__":
    main()
