"""Small CPU-only terminal training diagnostic; no inverse solves or training.

The matched corruption samples assess velocity error and its implied denoiser
error, not the error of an integrated generator. Exact t=1 is included as an
untrained boundary diagnostic, where the FM target retains independent x0.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import torch
import benchmark_dflow_dlr_sparse_ct_circles_gpu as bench


def main() -> None:
    torch.set_num_threads(4)
    output = Path("outputs/rk4_training_evidence_audit_2026_09_27")
    output.mkdir(parents=True, exist_ok=True)
    template = "outputs/fm_unet/circles-{res}-noattn-ch16-b8/n1"
    specifications = (
        (64, 100, template, "res64_epoch100"),
        (64, 200, template, "res64_epoch200_uniform"),
        (64, 200, template + "_tbias", "res64_epoch200_tbias"),
        (250, 100, template, "res250_epoch100"),
    )
    report = {
        "device": "cpu", "threads": 4, "seed": 170927,
        "samples": 8, "times": [0.5, 0.9, 0.99, 0.995, 0.999, 0.9999, 1.0],
        "note": "On training-path corruptions; not along the generator trajectory.",
        "models": {},
    }
    start = time.perf_counter()
    for resolution, epoch, model_template, name in specifications:
        truth = bench.normalize_images(torch.load(
            f"data/circles-{resolution}-n1.pt", map_location="cpu",
            weights_only=True, mmap=True)["test"][:report["samples"]])
        x0 = torch.randn(truth.shape, generator=torch.Generator().manual_seed(report["seed"]))
        model = bench.load_model(resolution, epoch, torch.device("cpu"), model_template)
        rows = []
        with torch.no_grad():
            for scalar in report["times"]:
                t = torch.full((truth.shape[0],), scalar)
                xt = (1 - scalar) * x0 + scalar * truth
                velocity = model(xt, t)
                target = truth - x0
                velocity_mse = (velocity - target).square().mean().item()
                denoised = xt + (1 - scalar) * velocity
                row = {
                    "t": scalar,
                    "velocity_mse": velocity_mse,
                    "denoiser_mse": (denoised - truth).square().mean().item(),
                    "identity_denoiser_mse": (1 - scalar) ** 2 * velocity_mse,
                }
                if scalar == 1.0:
                    row["expected_fm_mse_over_independent_x0"] = (
                        1 + (velocity - truth).square().mean().item())
                    row["model_minus_truth_mse"] = (velocity - truth).square().mean().item()
                rows.append(row)
        report["models"][name] = {
            "resolution": resolution, "epoch": epoch, "template": model_template,
            "attention": bool(model.mid_attn or any(model.att_channels)), "rows": rows,
        }
        report["elapsed_seconds"] = time.perf_counter() - start
        (output / "terminal_training_probe.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"model": name, "elapsed_seconds": report["elapsed_seconds"],
                          "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
