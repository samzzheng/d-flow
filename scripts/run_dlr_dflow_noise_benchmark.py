"""DLR-D-Flow sparse-view CT noise study at high resolution.

Runs the enriched dynamic-low-rank D-Flow solver at 128 and 160 with relative
Gaussian measurement noise at 1%, 5% and 10%, using a 100-step flow integrator.

Usage:  python scripts/run_dlr_dflow_noise_benchmark.py <device> <jobs>
        jobs is a comma-separated list of "<resolution>:<noise_pct>:<test_index>".
"""
from __future__ import annotations

import csv
import fcntl
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dlr_dflow.dlr_dflow as dd

OUT = ROOT / "outputs/dlr_dflow_noise_benchmark"
CSV = OUT / "results.csv"
FIELDS = ["resolution", "noise_level", "test_index", "rank", "flow_steps",
          "outer_steps", "lr", "angles", "psnr_db", "image_mse",
          "image_rel_l2", "measurement_mse", "noise_floor_mse", "best_step",
          "seconds", "sec_per_step", "peak_gib"]

RANK = 8
FLOW_STEPS = 100
OUTER_STEPS = int(__import__("os").environ.get("DLR_OUTER_STEPS", "120"))
LR = 0.05
N_ANGLES = 15
EPOCH = 50


def sparse_ct(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    k = angles.numel()
    rep = x[:, None].expand(b, k, c, h, w).reshape(b * k, c, h, w)
    ab = angles.repeat(b)
    cos, sin = torch.cos(ab), torch.sin(ab)
    theta = torch.zeros(b * k, 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    grid = F.affine_grid(theta, rep.shape, align_corners=False)
    rot = F.grid_sample(rep, grid, mode="bilinear",
                        padding_mode="zeros", align_corners=False)
    return rot.mean(-2).reshape(b, k, c, w).permute(0, 2, 1, 3)


def add_relative_noise(clean: torch.Tensor, level: float, seed: int) -> torch.Tensor:
    """Gaussian noise scaled so that ||eta|| / ||clean|| == level (repo convention)."""
    if level <= 0:
        return clean.clone()
    generator = torch.Generator(device=clean.device).manual_seed(seed)
    noise = torch.randn(clean.shape, generator=generator,
                        device=clean.device, dtype=clean.dtype)
    noise = noise / (noise.norm() + 1e-30) * (level * clean.norm())
    return clean + noise


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    return 10 * math.log10(1.0 / max((a - b).pow(2).mean().item(), 1e-12))


def append_row(row: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(CSV, "a", newline="") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        fcntl.flock(handle, fcntl.LOCK_UN)


def main() -> None:
    device = sys.argv[1]
    jobs = []
    for job in sys.argv[2].split(","):
        res, pct, idx = job.split(":")
        jobs.append((int(res), float(pct) / 100.0, int(idx)))
    OUT.mkdir(parents=True, exist_ok=True)

    cache: dict[int, tuple] = {}
    for res, level, idx in jobs:
        if res not in cache:
            model, _ = dd.load_flow_model(res, 1, EPOCH, device)
            test = torch.load(ROOT / f"data/circles-{res}-n1.pt",
                              map_location="cpu", weights_only=False)["test"]
            images = (test[:5].float() / 255.0).clone()
            del test
            angles = torch.linspace(0, math.pi, N_ANGLES + 1, device=device)[:-1]
            cache[res] = (model, images, angles)
        model, images, angles = cache[res]

        x_star = images[idx].unsqueeze(0).to(device)
        operator = lambda t: sparse_ct(t, angles)
        y_clean = operator(x_star).detach()
        y_obs = add_relative_noise(y_clean, level,
                                   seed=int(1e6 * level) + res * 10 + idx)
        noise_floor = (y_obs - y_clean).square().mean().item()

        z = torch.randn(1, 1, res, res, device=device,
                        generator=torch.Generator(device=device).manual_seed(res * 1000 + idx))
        m = torch.zeros_like(z)
        v = torch.zeros_like(z)
        best = {"misfit": float("inf"), "x": None, "step": 0}
        history = []

        torch.cuda.reset_peak_memory_stats(device)
        start = time.time()
        for k in range(1, OUTER_STEPS + 1):
            x1, factors, alpha1 = dd.sensitivity_factors(
                model, z, FLOW_STEPS, rank=RANK, method="enriched")
            g_x = dd.terminal_gradient(x1, operator, y_obs)
            grad = dd.approx_source_gradient(factors, g_x, alpha1).reshape_as(z)

            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad * grad
            z = z - LR * (m / (1 - 0.9 ** k)) / ((v / (1 - 0.999 ** k)).sqrt() + 1e-8)

            with torch.no_grad():
                xk = dd.euler_flow(model, z, FLOW_STEPS).clamp(0, 1)
                misfit = (operator(xk) - y_obs).square().mean().item()
            history.append({"step": k, "misfit": misfit, "psnr": psnr(xk, x_star)})
            if misfit < best["misfit"]:
                best = {"misfit": misfit, "x": xk.detach().clone(), "step": k}
        elapsed = time.time() - start

        x_hat = best["x"]
        rel_l2 = ((x_hat - x_star).norm() / x_star.norm()).item()
        row = {
            "resolution": res, "noise_level": level, "test_index": idx,
            "rank": RANK, "flow_steps": FLOW_STEPS, "outer_steps": OUTER_STEPS,
            "lr": LR, "angles": N_ANGLES,
            "psnr_db": round(psnr(x_hat, x_star), 3),
            "image_mse": (x_hat - x_star).square().mean().item(),
            "image_rel_l2": rel_l2,
            "measurement_mse": best["misfit"],
            "noise_floor_mse": noise_floor,
            "best_step": best["step"], "seconds": round(elapsed, 1),
            "sec_per_step": round(elapsed / OUTER_STEPS, 3),
            "peak_gib": round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 2),
        }
        append_row(row)
        torch.save({"x_star": x_star.cpu(), "x_hat": x_hat.cpu(),
                    "y_obs": y_obs.cpu(), "y_clean": y_clean.cpu(),
                    "history": history, "row": row},
                   OUT / f"sample_{res}_{int(level*100):02d}_{idx}.pt")
        print(f"[{device}] res {res} noise {level:.0%} idx {idx}: "
              f"PSNR {row['psnr_db']:6.2f} dB  misfit {best['misfit']:.3e} "
              f"(floor {noise_floor:.3e})  @{best['step']}  {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
