"""Can alpha_t DURING the DLR evolution be improved?  Exact test on circles-64.

Each test point z (starting source, and exact-gradient Armijo L-BFGS iterates 10
and 20; 1% noise sparse CT) gets the exact dense J_t path (d=4096, 100 Euler
steps, J_t kept on CPU). The DLR evolution (rank 8, 4 enrichment probes) is then
rerun with different alpha_t paths via enriched.evolve_trace_centered_enriched's
alpha_rate hook -- alpha is integrated from its rate, alpha_{k+1} = alpha_k +
h alpha_dot_k, so every variant is a consistent split J_t = alpha_t I + Y_t:

  probes{m}      default Hutchinson rate tr(P^T A J P)/(m d), m = 4, 32, 128
                 (128 = what averaging 4 independent 32-probe sketches gives)
  analytic       alpha_dot = -1, i.e. alpha_t = 1 - t: the plain-OT scalar part,
                 prod (1 - h/(1-t_k)); probe-free (run with 1 probe)
  deflated{m}    Hutchinson on the complement of the current U_t:
                 sum((I - U U^T) P * A J P) / (m (d - r)), m = 4, 32
  oracle_trace   exact tr(A_t J_t)/d from the dense path
  oracle_comp    exact complement-gain rate from the dense path, current U_t

Scored on the terminal transpose action against the exact J_1^T g:
error with alpha_1 = 0 (the benchmark default) and with the variant's own
alpha_1. The terminal-alpha_1 comparison of test_alpha_baseline.py is repeated
on the default (probes32) factors, reusing the same dense J_1.

Usage: python scripts/test_alpha_evolution.py [cuda:1]
"""
from __future__ import annotations

import json, math, sys, time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_dflow_dlr_sparse_ct_circles_gpu as bench
import dlr_dflow.dlr_dflow as dd
from dlr_dflow import enriched as E
from dlr_dflow.graphed_products import GraphedAProducts
from dlr_dflow.tangent_unet import unet_jvp

DEV = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:1")
R, STEPS, RANK, NOISE, SEED = 64, 100, 8, 0.01, 20260807
SAMPLES, ITERATES = [0, 1, 2], [0, 10, 20]
OUT = ROOT / "outputs/alpha_evolution_test"; OUT.mkdir(parents=True, exist_ok=True)
torch.cuda.set_device(DEV)
d, h = R * R, 1.0 / STEPS

model, _ = dd.load_flow_model(R, 1, 50, str(DEV))
bench.configure_dlr_kernels(R)
products = GraphedAProducts(model, torch.zeros(1, 1, R, R, device=DEV))
angles = torch.linspace(0, math.pi, 16, device=DEV)[:-1]
data = torch.load(ROOT / f"data/circles-{R}-n1.pt", map_location="cpu", weights_only=True)["test"]


def dense_path(z):
    """J_0..J_N (CPU, float32) through the same Euler steps, plus x_1."""
    x, J = z.reshape(1, 1, R, R).clone(), torch.eye(d, device=DEV)
    path = [J.cpu()]
    for i in range(STEPS):
        cols, primal = [], None
        for s in range(0, d, 512):
            with torch.no_grad():
                p, out = unet_jvp(model, i * h, x, J[:, s:s + 512].T.reshape(-1, 1, R, R))
            primal = p if primal is None else primal
            cols.append(out.reshape(-1, d).T)
        J = J + h * torch.cat(cols, 1)
        x = x + h * primal.reshape(1, 1, R, R)
        path.append(J.cpu())
    return x, path


def loss_grad_x(x, y):
    xt = x.detach().requires_grad_(True)
    obj, rel = bench.data_objective(bench.sparse_radon(xt, angles), y, R, "relative_scaled")
    g, = torch.autograd.grad(obj, xt)
    return g.flatten(), math.sqrt(rel.item())


def exact_iterates(z0, y, keep):
    src = z0.clone().requires_grad_(True)

    def objective(c):
        with torch.no_grad():
            v, _ = bench.data_objective(
                bench.sparse_radon(dd.euler_flow(model, c, STEPS), angles), y, R,
                "relative_scaled")
        return float(v)
    opt = bench.ArmijoLBFGS(src, objective, lr=1.0, history=100)

    def closure():
        opt.zero_grad()
        v, _ = bench.data_objective(
            bench.sparse_radon(dd.euler_flow(model, src, STEPS), angles), y, R,
            "relative_scaled")
        v.backward()
        return v.detach()
    out = {0: z0.clone()}
    for k in range(1, max(keep) + 1):
        opt.step(closure)
        if k in keep:
            out[k] = src.detach().clone()
    return out


def comp_trace(J, U):
    """tr((I - U U^T) J (I - U U^T)) = tr J - tr(U^T J U), on CPU."""
    if U.shape[1] == 0:
        return torch.trace(J).item()
    Uc = U.detach().cpu()
    return torch.trace(J).item() - (Uc * (J @ Uc)).sum().item()


def variants(path):
    trJ = [torch.trace(J).item() for J in path]

    def probes_of(m):
        return E._trace_probes(d, m, 0 + 1, DEV, torch.float32)[0]

    def deflated(m):
        P = probes_of(m)

        def rate(k, pd, U):
            r = U.shape[1]
            Q = P - U @ (U.T @ P) if r else P
            return (Q * pd).sum() / (m * (d - r))
        return rate

    def oracle_comp(k, pd, U):
        r = U.shape[1]
        return (comp_trace(path[k + 1], U) - comp_trace(path[k], U)) / (h * (d - r))

    return {
        "probes4": dict(trace_probes=4),
        "probes32": dict(trace_probes=32),
        "probes128": dict(trace_probes=128),
        "analytic": dict(trace_probes=1, alpha_rate=lambda k, pd, U: -1.0),
        "deflated4": dict(trace_probes=4, alpha_rate=deflated(4)),
        "deflated32": dict(trace_probes=32, alpha_rate=deflated(32)),
        "oracle_trace": dict(trace_probes=4, alpha_rate=lambda k, pd, U: (trJ[k + 1] - trJ[k]) / (h * d)),
        "oracle_comp": dict(trace_probes=4, alpha_rate=oracle_comp),
    }


def score(factors, alpha, g, target):
    U, S, V = factors
    low = V @ (S.T @ (U.T @ g))
    out = {}
    for tag, a in (("a0", 0.0), ("own", float(alpha))):
        est = a * g + low
        out[f"err_{tag}"] = ((target - est).norm() / target.norm()).item()
        out[f"cos_{tag}"] = (est @ target / (est.norm() * target.norm())).item()
    return out


rows = []
for s in SAMPLES:
    gt = bench.normalize_images(data[s:s + 1]).to(DEV)
    with torch.no_grad():
        clean = bench.sparse_radon(gt, angles)
    noisy, _, _ = bench.add_relative_noise(
        clean.cpu(), NOISE, torch.Generator(device="cpu").manual_seed(SEED + R * 1000 + s))
    y = noisy.to(DEV)
    z0 = torch.randn(gt.shape, generator=torch.Generator(device="cpu").manual_seed(
        SEED + 10_000_000 + R * 1000 + s)).to(DEV)
    points = exact_iterates(z0, y, ITERATES)
    for k in ITERATES:
        z = points[k]; t0 = time.time()
        x1, path = dense_path(z)
        J1 = path[-1].to(DEV)
        g, rel = loss_grad_x(x1, y)
        target = J1.T @ g
        dense_s = time.time() - t0
        row = dict(sample=s, iterate=k, residual=rel,
                   armijo_initial_step_policy=bench.ArmijoLBFGS.initial_step_policy,
                   armijo_progress_policy=bench.ArmijoLBFGS.progress_policy,
                   z_over_sqrt_d=z.norm().item() / math.sqrt(d), dense_seconds=dense_s,
                   exact_alpha1_trace=torch.trace(J1).item() / d, variants={})
        for name, kw in variants(path).items():
            torch.cuda.synchronize(); tv = time.time()
            x_dlr, factors, a1 = E.evolve_trace_centered_enriched(
                model, z, STEPS, rank=RANK, initialization_oversampling=8,
                enrichment_probes=4, factor_stride=1, products=products,
                zero_terminal_baseline=False, **kw)
            torch.cuda.synchronize()
            res = score(factors, a1, g, target)
            res.update(alpha1=float(a1), seconds=time.time() - tv,
                       exact_comp_alpha1=comp_trace(path[-1], factors[0]) / (d - RANK))
            row["variants"][name] = res
            if name == "probes32":
                U = factors[0]
                row["probes32_terminal_alpha"] = dict(
                    exact_trace=torch.trace(J1).item() / d,
                    comp_dlrU=comp_trace(path[-1], U) / (d - RANK))
        rows.append(row)
        print(f"sample {s} iterate {k}: residual {rel:.4f}  |z|/sqrt(d) {row['z_over_sqrt_d']:.3f}"
              f"  dense {dense_s:.0f}s", flush=True)
        for name, v in row["variants"].items():
            print(f"   {name:13s} err(a1=0) {v['err_a0']:.4f}  err(own a1={v['alpha1']:+.4f}) "
                  f"{v['err_own']:.4f}   {v['seconds']:.1f}s", flush=True)
        (OUT / "results.json").write_text(json.dumps(rows, indent=1))
        del path, J1
print("ALPHA EVOLUTION TEST DONE")
