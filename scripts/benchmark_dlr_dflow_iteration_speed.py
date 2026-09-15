"""Per-iteration speed: standard D-Flow vs DLR-D-Flow, one inverse solve each.

Standard D-Flow differentiates through the Euler flow (backpropagation through the
ODE solver). DLR-D-Flow evolves the rank-r sensitivity factors alongside the flow.
One outer iteration = source gradient + Adam update + objective evaluation.
"""
from __future__ import annotations

import csv
import json
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
import os
RANK, FLOW_STEPS, N_ANGLES, LR = 8, 100, 15, 0.05
TRACE_PROBES = int(os.environ.get("DLR_TRACE_PROBES", "32"))
RESOLUTIONS = [int(v) for v in os.environ.get("DLR_RESOLUTIONS", "64,96,128,160").split(",")]
WARMUP, TIMED = 1, 3


def sparse_ct(x, angles):
    b, c, h, w = x.shape
    k = angles.numel()
    rep = x[:, None].expand(b, k, c, h, w).reshape(b * k, c, h, w)
    ab = angles.repeat(b)
    cos, sin = torch.cos(ab), torch.sin(ab)
    th = torch.zeros(b * k, 2, 3, device=x.device, dtype=x.dtype)
    th[:, 0, 0] = cos; th[:, 0, 1] = -sin
    th[:, 1, 0] = sin; th[:, 1, 1] = cos
    grid = F.affine_grid(th, rep.shape, align_corners=False)
    rot = F.grid_sample(rep, grid, mode="bilinear",
                        padding_mode="zeros", align_corners=False)
    return rot.mean(-2).reshape(b, k, c, w).permute(0, 2, 1, 3)


def time_solver(model, z0, operator, y_obs, device, kind):
    """Average wall-clock per outer iteration, after a warm-up iteration."""
    z = z0.clone()
    m, v = torch.zeros_like(z), torch.zeros_like(z)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    start = None
    for k in range(1, WARMUP + TIMED + 1):
        if k == WARMUP + 1:
            torch.cuda.synchronize(); start = time.time()
        if kind == "dflow":                       # backprop through the ODE solver
            zl = z.detach().requires_grad_(True)
            x1 = dd.euler_flow(model, zl, FLOW_STEPS)
            loss = 0.5 * (operator(x1) - y_obs).pow(2).sum()
            (grad,) = torch.autograd.grad(loss, zl)
        else:                                     # DLR-D-Flow low-rank sensitivity
            x1, factors, alpha1 = dd.sensitivity_factors(
                model, z, FLOW_STEPS, rank=RANK, method="enriched",
                trace_probes=TRACE_PROBES)
            g_x = dd.terminal_gradient(x1, operator, y_obs)
            grad = dd.approx_source_gradient(factors, g_x, alpha1).reshape_as(z)
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad * grad
        z = z - LR * (m / (1 - 0.9 ** k)) / ((v / (1 - 0.999 ** k)).sqrt() + 1e-8)
        with torch.no_grad():
            xk = dd.euler_flow(model, z, FLOW_STEPS).clamp(0, 1)
            _ = (operator(xk) - y_obs).square().mean().item()
    torch.cuda.synchronize()
    return (time.time() - start) / TIMED, torch.cuda.max_memory_allocated(device) / 2 ** 30


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for res in RESOLUTIONS:
        model, _ = dd.load_flow_model(res, 1, 50, device)
        test = torch.load(ROOT / f"data/circles-{res}-n1.pt",
                          map_location="cpu", weights_only=False)["test"]
        x_star = (test[0].unsqueeze(0).float() / 255.0).to(device)
        del test
        angles = torch.linspace(0, math.pi, N_ANGLES + 1, device=device)[:-1]
        operator = lambda t: sparse_ct(t, angles)
        y_obs = operator(x_star).detach()
        z0 = torch.randn(1, 1, res, res, device=device,
                         generator=torch.Generator(device=device).manual_seed(res))

        row = {"resolution": res, "d": res * res, "flow_steps": FLOW_STEPS, "rank": RANK}
        for kind, label in [("dflow", "D-Flow"), ("dlr", "DLR-D-Flow")]:
            try:
                sec, peak = time_solver(model, z0, operator, y_obs, device, kind)
                row[f"{kind}_sec_per_iter"] = round(sec, 3)
                row[f"{kind}_peak_gib"] = round(peak, 2)
                print(f"res {res:3d}  {label:11s} {sec:8.2f} s/iter   peak {peak:6.2f} GiB", flush=True)
            except torch.OutOfMemoryError:
                row[f"{kind}_sec_per_iter"] = None
                row[f"{kind}_peak_gib"] = None
                print(f"res {res:3d}  {label:11s}      OOM  (exceeds device memory)", flush=True)
                torch.cuda.empty_cache()
        if row.get("dflow_sec_per_iter") and row.get("dlr_sec_per_iter"):
            row["ratio_dlr_over_dflow"] = round(
                row["dlr_sec_per_iter"] / row["dflow_sec_per_iter"], 2)
        rows.append(row)
        del model
        torch.cuda.empty_cache()

    keys = ["resolution", "d", "flow_steps", "rank", "dflow_sec_per_iter",
            "dflow_peak_gib", "dlr_sec_per_iter", "dlr_peak_gib", "ratio_dlr_over_dflow"]
    suffix = "" if TRACE_PROBES == 32 else f"_probes{TRACE_PROBES}"
    with open(OUT / f"iteration_speed{suffix}.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})
    json.dump(rows, open(OUT / f"iteration_speed{suffix}.json", "w"), indent=1)
    print("\nwrote", OUT / f"iteration_speed{suffix}.csv")


if __name__ == "__main__":
    main()
