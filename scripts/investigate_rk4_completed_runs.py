"""Read-only audit of completed 250px Euler/RK4 inverse solves.

Outputs are a snapshot; running jobs and their files are never changed.
Local and remotely run results stay separate to expose backend effects.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "armijo_local": ROOT / "outputs/euler_rk4_250_noise001_armijo",
    "armijo_modal": ROOT / "outputs/modal_runs/euler_rk4_250_noise001_armijo",
    "old_no_line_search": ROOT / "outputs/euler_rk4_250_noise001_cpu",
}


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def summarize_pairs(pairs):
    if not pairs:
        return {}
    out = {"n": len(pairs), "samples": [p["sample"] for p in pairs]}
    for metric in ("image_relative_l2", "measurement_relative_l2_noisy", "measurement_relative_l2_clean"):
        e = np.array([float(p["euler"][metric]) for p in pairs])
        r = np.array([float(p["rk4"][metric]) for p in pairs])
        delta = r-e
        rng = np.random.default_rng(20260927)
        bootstrap_samples = delta[rng.integers(0, len(delta), size=(20000, len(delta)))]
        boot = bootstrap_samples.mean(axis=1)
        out[metric] = {"euler_mean": float(e.mean()), "rk4_mean": float(r.mean()),
            "euler_median": float(np.median(e)), "rk4_median": float(np.median(r)),
            "euler_std": float(e.std(ddof=1)) if len(e)>1 else None,
            "rk4_std": float(r.std(ddof=1)) if len(r)>1 else None,
            "rk4_wins": int((r<e).sum()), "euler_wins": int((e<r).sum()),
            "paired_rk4_minus_euler_mean": float(delta.mean()),
            "paired_mean_bootstrap_95pct": np.quantile(boot, [.025,.975]).tolist(),
            "paired_rk4_minus_euler_median": float(np.median(delta)),
            "paired_median_bootstrap_95pct": np.quantile(np.median(bootstrap_samples, axis=1), [.025,.975]).tolist()}
    out["rk4_fits_noisy_data_better_but_image_worse"] = [p["sample"] for p in pairs
        if float(p["rk4"]["measurement_relative_l2_noisy"]) < float(p["euler"]["measurement_relative_l2_noisy"])
        and float(p["rk4"]["image_relative_l2"]) > float(p["euler"]["image_relative_l2"])]
    out["both_300_steps"] = sum(int(p["euler"]["steps_run"])==300 and int(p["rk4"]["steps_run"])==300 for p in pairs)
    out["both_images_below_5pct"] = sum(max(float(p[m]["image_relative_l2"]) for m in ("euler","rk4")) < .05 for p in pairs)
    out["median_rk4_wall_ratio"] = statistics.median(float(p["rk4"]["elapsed_s"])/float(p["euler"]["elapsed_s"]) for p in pairs)
    for method in ("euler", "rk4"):
        out[method+"_stops"] = {str(tau): {"reached": sum(p[method]["discrepancy_first_crossing"][str(tau)] is not None for p in pairs),
            "median_first_step": statistics.median([p[method]["discrepancy_first_crossing"][str(tau)] for p in pairs
                if p[method]["discrepancy_first_crossing"][str(tau)] is not None])
            if any(p[method]["discrepancy_first_crossing"][str(tau)] is not None for p in pairs) else None}
            for tau in (1.0, 1.05, 1.1, 1.2)}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/rk4_accuracy_2026_09_27/completed_runs.json")
    args = parser.parse_args()
    report = {"snapshot_utc": datetime.now(timezone.utc).isoformat(), "groups": {}}
    for group, root in GROUPS.items():
        rows, duplicates, settings = {}, [], {}
        for sample_file in sorted(root.glob("*/samples.csv")):
            directory = sample_file.parent
            config = json.loads((directory / "settings.json").read_text())
            settings[str(directory)] = config
            history_path = directory / "iterations.csv"
            histories = read_csv(history_path) if history_path.exists() else []
            for row in read_csv(sample_file):
                if int(row["resolution"]) != 250 or row["method"] not in ("forward_euler", "rk4"):
                    continue
                sample, method = int(row["sample_index"]), row["method"]
                key = (sample, method)
                record = row | {"source_csv": str(sample_file)}
                history = [r for r in histories if int(r["sample_index"])==sample and r["method"]==method]
                history.sort(key=lambda r:int(r["outer_step"]))
                record["history"] = history
                record["discrepancy_first_crossing"] = {str(tau): next((int(r["outer_step"]) for r in history
                    if float(r["measurement_relative_l2"]) <= tau * config["noise_level"]), None)
                    for tau in (1.0, 1.05, 1.1, 1.2)}
                npz = directory / f"figures/res_250/sample_{sample:03d}/{method}.npz"
                if npz.exists():
                    with np.load(npz) as fields:
                        clean, noisy, predicted = [fields[k].astype(np.float64).ravel() for k in
                            ("clean_measurement", "noisy_measurement", "predicted_measurement")]
                        error, noise = predicted-clean, noisy-clean
                        record["consistent_clamped_metrics"] = {
                            "noisy_residual": float(np.linalg.norm(predicted-noisy)/np.linalg.norm(noisy)),
                            "clean_residual": float(np.linalg.norm(error)/np.linalg.norm(clean)),
                            "error_noise_cosine": float(error@noise/(np.linalg.norm(error)*np.linalg.norm(noise))),
                            "noise_direction_fit": float(error@noise/(noise@noise)),
                            "noise_norm_relative_clean": float(np.linalg.norm(noise)/np.linalg.norm(clean)),
                            "noise_norm_relative_noisy": float(np.linalg.norm(noise)/np.linalg.norm(noisy)),
                        }
                        record["discrepancy_first_crossing"] = {str(tau): next((int(r["outer_step"]) for r in history
                            if float(r["measurement_relative_l2"]) <= tau * np.linalg.norm(noise)/np.linalg.norm(noisy)), None)
                            for tau in (1.0, 1.05, 1.1, 1.2)}
                        record["discrepancy_normalization"] = "||noise|| / ||noisy measurement|| from saved tensors"
                if key in rows:
                    duplicates.append({"sample":sample, "method":method, "paths":[rows[key]["source_csv"],str(sample_file)]})
                    # Prefer the longer solve; tie-break paths are stable.
                    if int(rows[key]["steps_run"])>int(record["steps_run"]):
                        continue
                rows[key] = record
        samples = sorted({s for s,m in rows if (s,"forward_euler") in rows and (s,"rk4") in rows})
        pairs = [{"sample":s,"euler":rows[s,"forward_euler"],"rk4":rows[s,"rk4"]} for s in samples]
        report["groups"][group] = {"settings":settings, "pairs":pairs,
            "unpaired": [{"sample":s,"method":m,"source_csv":r["source_csv"]} for (s,m),r in rows.items() if s not in samples],
            "duplicates":duplicates, "summary":summarize_pairs(pairs)}
        print(group, json.dumps(report["groups"][group]["summary"],indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print("saved",args.out)


if __name__ == "__main__":
    main()
