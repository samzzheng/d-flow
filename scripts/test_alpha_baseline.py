"""Which terminal baseline alpha_1 should DLR use? Exact test on circles-64.

At each test point z (starting source and exact-gradient L-BFGS iterates), forms
the exact dense J_1 (d=4096, 100 Euler steps) and the DLR factors (rank 8,
same settings as the benchmarks), then scores the transpose action

    J_1^T g  ~  alpha g + V S^T U^T g,   g = d(loss)/dx_1 on 1%-noise sparse CT

for these alphas:
  zero        alpha = 0: current benchmark default (zero_terminal_baseline=True);
              also what the analytic OT prefactor gives here (prod (1-1/j) = 0)
  dlr_trace   the evolution's own Hutchinson tr(J_1)/d, 32 probes
  exact_trace exact tr(J_1)/d  (dlr_trace minus its probe noise: bias vs variance)
  comp_dlrU   exact complement gain tr((I-UU^T) J_1 (I-UU^T))/(d-r), U from DLR
  comp_svdU   same with U = exact top-r left singular vectors of J_1
  comp_hpp    deflated Hutch++ estimate of comp_dlrU, 32 probes (one draw; the
              spread over 200 draws is reported separately)
  oracle      the alpha that minimises the error for this g (lower bound)
"""
from __future__ import annotations

import json, math, sys, time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_dflow_dlr_sparse_ct_circles_gpu as bench
import dlr_dflow.dlr_dflow as dd
from dlr_dflow.graphed_products import GraphedAProducts
from dlr_dflow.tangent_unet import unet_jvp

DEV = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:1")
R, STEPS, RANK, PROBES, NOISE, SEED = 64, 100, 8, 32, 0.01, 20260807
SAMPLES, ITERATES = [0, 1, 2], [0, 10, 20]
OUT = ROOT / "outputs/alpha_baseline_test"; OUT.mkdir(parents=True, exist_ok=True)
torch.cuda.set_device(DEV)
d = R * R

model, _ = dd.load_flow_model(R, 1, 50, str(DEV))
bench.configure_dlr_kernels(R)
products = GraphedAProducts(model, torch.zeros(1, 1, R, R, device=DEV))
angles = torch.linspace(0, math.pi, 16, device=DEV)[:-1]
data = torch.load(ROOT / f"data/circles-{R}-n1.pt", map_location="cpu", weights_only=True)["test"]


def exact_J1(z):
    """Dense forward sensitivity through the same Euler steps (chunked JVPs)."""
    x, J, h = z.reshape(1, 1, R, R).clone(), torch.eye(d, device=DEV), 1.0 / STEPS
    for i in range(STEPS):
        cols, primal = [], None
        for s in range(0, d, 512):
            with torch.no_grad():
                p, out = unet_jvp(model, i * h, x, J[:, s:s + 512].T.reshape(-1, 1, R, R))
            primal = p if primal is None else primal
            cols.append(out.reshape(-1, d).T)
        J = J + h * torch.cat(cols, 1)
        x = x + h * primal.reshape(1, 1, R, R)
    return x, J


def loss_grad_x(x, y):
    xt = x.detach().requires_grad_(True)
    obj, rel = bench.data_objective(bench.sparse_radon(xt, angles), y, R, "relative_scaled")
    g, = torch.autograd.grad(obj, xt)
    return g.flatten(), math.sqrt(rel.item())


def rademacher(m, gen):
    return (torch.randint(0, 2, (d, m), generator=gen, device=DEV) * 2 - 1).float()


def exact_iterates(z0, y, keep):
    """Exact-gradient Armijo L-BFGS (method-neutral hard iterates)."""
    src = z0.clone().requires_grad_(True)
    class A: pass
    args = A(); args.source_prior = "none"; args.source_prior_lambda = 0.0

    def objective(c):
        with torch.no_grad():
            x = dd.euler_flow(model, c, STEPS)
            v, _ = bench.data_objective(bench.sparse_radon(x, angles), y, R, "relative_scaled")
        return float(v)
    opt = bench.ArmijoLBFGS(src, objective, lr=1.0, history=100)

    def closure():
        opt.zero_grad()
        x = dd.euler_flow(model, src, STEPS)
        v, _ = bench.data_objective(bench.sparse_radon(x, angles), y, R, "relative_scaled")
        v.backward()
        return v.detach()
    out = {0: z0.clone()}
    for k in range(1, max(keep) + 1):
        opt.step(closure)
        if k in keep:
            out[k] = src.detach().clone()
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
        x_dlr, (U, S, V), a_dlr = dd.sensitivity_factors(
            model, z, STEPS, rank=RANK, method="enriched", trace_probes=PROBES,
            initialization_oversampling=8, enrichment_probes=4, factor_stride=1,
            products=products, zero_terminal_baseline=False)
        x1, J = exact_J1(z)
        g, rel = loss_grad_x(x1, y)
        target = J.T @ g
        # sanity: dense J^T g must equal autograd through the flow
        zz = z.clone().requires_grad_(True)
        xa = dd.euler_flow(model, zz, STEPS)
        oa, _ = bench.data_objective(bench.sparse_radon(xa, angles), y, R, "relative_scaled")
        ga, = torch.autograd.grad(oa, zz)
        dense_vs_autograd = ((target - ga.flatten()).norm() / ga.flatten().norm()).item()

        low = V @ (S.T @ (U.T @ g))
        trJ = torch.trace(J).item()
        Usvd = torch.linalg.svd(J)[0][:, :RANK]
        comp = lambda Q: (trJ - torch.trace(Q.T @ J @ Q).item()) / (d - RANK)
        gen = torch.Generator(device=DEV).manual_seed(123)
        Pperp = lambda X: X - U @ (U.T @ X)
        hutch, hpp = [], []
        for _ in range(200):
            P = rademacher(PROBES, gen)
            hutch.append((P * (J @ P)).sum().item() / (PROBES * d))
            Q = Pperp(P)
            hpp.append((Q * (J @ Q)).sum().item() / (PROBES * (d - RANK)))
        resid = target - low
        alphas = dict(zero=0.0, dlr_trace=float(a_dlr), exact_trace=trJ / d,
                      comp_dlrU=comp(U), comp_svdU=comp(Usvd), comp_hpp=hpp[0],
                      oracle=(g @ resid / (g @ g)).item())
        err = {n: ((target - (a * g + low)).norm() / target.norm()).item()
               for n, a in alphas.items()}
        cos = {n: (torch.dot(a * g + low, target) /
                   ((a * g + low).norm() * target.norm())).item() for n, a in alphas.items()}
        row = dict(sample=s, iterate=k, residual=rel, z_over_sqrt_d=z.norm().item() / math.sqrt(d),
                   armijo_initial_step_policy=bench.ArmijoLBFGS.initial_step_policy,
                   armijo_progress_policy=bench.ArmijoLBFGS.progress_policy,
                   dense_vs_autograd=dense_vs_autograd, alphas=alphas, rel_error=err, cosine=cos,
                   hutch_trace_std=torch.tensor(hutch).std().item(),
                   hpp_comp_std=torch.tensor(hpp).std().item(),
                   top_sv=torch.linalg.svdvals(J)[:6].tolist(),
                   dlr_vs_exact_x=((x_dlr.flatten() - x1.flatten()).norm() / x1.norm()).item(),
                   seconds=time.time() - t0)
        rows.append(row)
        print(json.dumps({k2: row[k2] for k2 in ("sample", "iterate", "residual", "z_over_sqrt_d",
                                                   "dense_vs_autograd")}),
              "\n  alpha:", {n: f"{a:.4g}" for n, a in alphas.items()},
              "\n  err  :", {n: f"{e:.4f}" for n, e in err.items()},
              f"\n  hutch std {row['hutch_trace_std']:.3g}  hpp std {row['hpp_comp_std']:.3g}"
              f"  {row['seconds']:.0f}s", flush=True)
        (OUT / "results.json").write_text(json.dumps(rows, indent=1))
print("ALPHA TEST DONE")
