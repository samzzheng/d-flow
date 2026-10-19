"""Why does L-BFGS diverge on DLR-D-Flow gradients? Trace the first steps.

For each sample, reproduces the matched-protocol setup (250px, 1% noise, same
seeds, relative_scaled objective) and runs L-BFGS (lr=1, no line search) on the
DLR gradient. At every DLR iterate it also computes the EXACT D-Flow gradient,
and logs: true objective, cos(g_dlr, g_exact), |g_dlr|/|g_exact|, and the
directional derivatives of the L-BFGS direction under both gradients. A
direction that looks like descent to DLR (d.g_dlr < 0) but is ascent for the
true objective (d.g_exact > 0) is the smoking gun.

Usage: python scripts/diagnose_dlr_lbfgs.py --device cuda:1 --samples 3 6 10 0
"""
from __future__ import annotations

import argparse, json, math, sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_dflow_dlr_sparse_ct_circles_gpu as bench
import dlr_dflow.dlr_dflow as dd
from dlr_dflow.graphed_products import GraphedAProducts

p = argparse.ArgumentParser()
p.add_argument("--device", default="cuda:1")
p.add_argument("--samples", type=int, nargs="+", default=[3, 6, 10, 0])
p.add_argument("--iters", type=int, default=8)
p.add_argument("--resolution", type=int, default=250)
p.add_argument("--out", type=Path, default=ROOT / "outputs/optimizer_diagnosis/lbfgs_trace.json")
a = p.parse_args()

R, base_seed, noise, steps = a.resolution, 20260807, 0.01, 100
dev = torch.device(a.device); torch.cuda.set_device(dev)
model = bench.load_model(R, 100, dev, "outputs/fm_unet/circles-{res}-noattn-ch16-b8/n1")
bench.configure_dlr_kernels(R)
products = GraphedAProducts(model, torch.zeros(1, 1, R, R, device=dev))
data = torch.load(ROOT / f"data/circles-{R}-n1.pt", map_location="cpu", weights_only=True, mmap=True)["test"]
angles = torch.linspace(0, math.pi, 16, device=dev)[:-1]


def exact(z, y):
    z = z.detach().clone().requires_grad_(True)
    x = bench.solve_flow(model, z, steps, "forward_euler", True)
    obj, rel = bench.data_objective(bench.sparse_radon(x, angles), y, R, "relative_scaled")
    g, = torch.autograd.grad(obj, z)
    return obj.item(), math.sqrt(rel.item()), g.flatten()


def dlr(z, y):
    x, f, al = dd.sensitivity_factors(model, z.detach(), steps, rank=8, method="enriched",
                                      trace_probes=32, initialization_oversampling=8,
                                      enrichment_probes=4, factor_stride=1, products=products)
    xt = x.detach().requires_grad_(True)
    obj, rel = bench.data_objective(bench.sparse_radon(xt, angles), y, R, "relative_scaled")
    gx, = torch.autograd.grad(obj, xt)
    return obj.item(), math.sqrt(rel.item()), dd.approx_source_gradient(f, gx, al).flatten()


results = {}
for s in a.samples:
    gt = bench.normalize_images(data[s:s + 1]).to(dev)
    with torch.no_grad():
        clean = bench.sparse_radon(gt, angles)
    gen = torch.Generator(device="cpu").manual_seed(base_seed + R * 1000 + s)
    noisy, _, _ = bench.add_relative_noise(clean.cpu(), noise, gen)
    y = noisy.to(dev)
    z = torch.randn(gt.shape, generator=torch.Generator(device="cpu").manual_seed(
        base_seed + 10_000_000 + R * 1000 + s)).to(dev).flatten()
    # hand-rolled L-BFGS identical to torch's no-line-search update
    S, Y, rows = [], [], []
    g_prev = z_prev = None
    for k in range(a.iters + 1):
        fd, rd, gd = dlr(z.view_as(gt), y)
        fe, re_, ge = exact(z.view_as(gt), y)
        if k > 0:
            s_, y_ = z - z_prev, gd - g_prev
            if y_.dot(s_) > 1e-10:
                S.append(s_); Y.append(y_)
                S, Y = S[-100:], Y[-100:]
        # two-loop
        q = -gd.clone(); al_ = []
        for s_, y_ in reversed(list(zip(S, Y))):
            rho = 1 / y_.dot(s_); a_ = rho * s_.dot(q); q -= a_ * y_; al_.append((rho, a_))
        H0 = (S[-1].dot(Y[-1]) / Y[-1].dot(Y[-1])) if S else 1.0
        q *= H0
        for (s_, y_), (rho, a_) in zip(zip(S, Y), reversed(al_)):
            b = rho * y_.dot(q); q += s_ * (a_ - b)
        d = q
        t = min(1.0, 1.0 / gd.abs().sum().item()) if k == 0 else 1.0
        cos = (gd.dot(ge) / (gd.norm() * ge.norm())).item()
        row = dict(k=k, rel=rd, obj=fd, cos_dlr_exact=cos,
                   norm_ratio=(gd.norm() / ge.norm()).item(),
                   dir_deriv_dlr=(t * d.dot(gd)).item(), dir_deriv_exact=(t * d.dot(ge)).item(),
                   step_norm=(t * d).norm().item(), z_norm=z.norm().item(),
                   pairs=len(S))
        rows.append(row)
        print(f"s={s} k={k} rel={rd:.4g} cos={cos:+.3f} |gd|/|ge|={row['norm_ratio']:.3f} "
              f"d.g_dlr={row['dir_deriv_dlr']:+.3e} d.g_exact={row['dir_deriv_exact']:+.3e} "
              f"|step|={row['step_norm']:.3g} |z|={row['z_norm']:.3g}", flush=True)
        if not math.isfinite(rd) or rd > 1e6:
            break
        z_prev, g_prev = z.clone(), gd.clone()
        z = z + t * d
    results[s] = rows
    a.out.write_text(json.dumps(results, indent=1))
