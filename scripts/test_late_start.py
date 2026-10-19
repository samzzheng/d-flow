"""Late-start DLR sweep: start evolving the low-rank residual at step m0 instead of 0.

For m0 in the sweep, the flow state is integrated with plain Euler to t0 = m0 h,
the residual is initialised there as Y_{t0} = 0 (J_{t0} ~= alpha_{t0} I, the
proposal's assumption) by the same randomized sketch the standard evolution uses
at t = 0, and the KSL step (enriched._fused_enriched_step, rank 8, 4 enrichment
probes) runs from m0 to 100.  The baseline is the analytic alpha_t = 1 - t
throughout, as the proposal recommends.  m0 = 0 is the standard evolution with
that baseline.  A reduced-rank-early variant evolves rank r_early until m_switch,
then pads the bases to rank 8.

Scored on the terminal transpose action against autograd (alpha_1 = 0), at the
starting source and capped-Armijo exact-gradient iterate 10, samples 0-2, 64px.
Usage: python scripts/test_late_start.py [cuda:1]
"""
import json, math, sys, time
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import test_parallel_bug_integrator as H
import dlr_dflow.dlr_dflow as dd
from dlr_dflow import enriched as E
from dlr_dflow.graphed_products import GraphedAProducts

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:1"); torch.cuda.set_device(dev)
R, N, RANK, P, OVER = 64, 100, 8, 4, 8
h = 1.0 / N
SWEEP = [0, 30, 50, 70, 80, 90, 95]
model, tag = H.load(R, dev)
products = GraphedAProducts(model, torch.zeros(1, 1, R, R, device=dev))
angles = torch.linspace(0, math.pi, 16, device=dev)[:-1]
data = torch.load(ROOT / f"data/circles-{R}-n1.pt", map_location="cpu", weights_only=True, mmap=True)["test"]
analytic = lambda pd, U: -1.0           # d alpha / dt for alpha_t = 1 - t


def evolve(z, m0, rank_early=None, m_switch=None, seed=0):
    """KSL evolution starting at step m0 with Y=0 there; analytic alpha; returns x1, factors, alpha1."""
    d = R * R
    x = z.detach()
    with torch.no_grad():
        for m in range(m0):
            x = x + h * dd.velocity(model, m * h, x)
    probes, gen = E._trace_probes(d, 1, seed + 1, z.device, z.dtype)   # 1 dummy probe column
    sens = probes.clone()
    alpha = 1.0 - m0 * h
    rank0 = rank_early or RANK
    # first step at t0: sketch Y_{t0+h} ~= h alpha A_{t0} + (alpha_{t0} - alpha_{t0+h}) I = h(alpha A + I)
    products.set_state(m0 * h, x); A, AT = products.A, products.AT
    sketch = E._rademacher(d, rank0 + OVER, gen, z.device, z.dtype)
    first = A(torch.cat([sens, sketch], 1))
    sens = sens + h * first[:, :1]
    q, _ = torch.linalg.qr(h * (alpha * first[:, 1:] + sketch), mode="reduced")
    ti = h * (alpha * AT(q) + q)
    rv, s, lt = E._thin_svd(ti)
    factors = (q @ lt.T[:, :rank0], torch.diag(s[:rank0]), rv[:, :rank0])
    with torch.no_grad():
        x = x + h * A.last_primal.reshape_as(x)
    alpha = alpha - h
    for m in range(m0 + 1, N):
        if m_switch is not None and m == m_switch and factors[0].shape[1] < RANK:
            U, S, V = factors; k = RANK - U.shape[1]
            Ue = torch.linalg.qr(E._rademacher(d, k, gen, z.device, z.dtype) - U @ (U.T @ E._rademacher(d, k, gen, z.device, z.dtype)))[0]
            Ve = torch.linalg.qr(E._rademacher(d, k, gen, z.device, z.dtype) - V @ (V.T @ E._rademacher(d, k, gen, z.device, z.dtype)))[0]
            Sp = torch.zeros(RANK, RANK, device=dev); Sp[:S.shape[0], :S.shape[1]] = S
            factors = (torch.cat([U, Ue], 1), Sp, torch.cat([V, Ve], 1))
        products.set_state(m * h, x)
        factors, sd, _ = E._fused_enriched_step(A, AT, factors, sens, probes, h, alpha, P, gen,
                                                alpha_rate=analytic)
        sens = sens + h * sd
        alpha = alpha - h
        with torch.no_grad():
            x = x + h * A.last_primal.reshape_as(x)
    return x, factors, float(alpha)


rows = []
for sample in (0, 1, 2):
    z0, y = H.make_problem(R, sample, data, angles, dev)
    pts = H.exact_iterates(model, z0, y, R, angles, [10])
    for k, z in ((0, pts[0]), (10, pts[10])):
        g_exact, rel, _ = H.exact_gradient(model, z, y, R, angles)
        row = {"sample": sample, "iterate": k, "residual": rel,
               "armijo_initial_step_policy": H.bench.ArmijoLBFGS.initial_step_policy,
               "armijo_progress_policy": H.bench.ArmijoLBFGS.progress_policy,
               "configs": {}}
        configs = [(f"m0={m0}", dict(m0=m0)) for m0 in SWEEP] + [
            ("r3_to_8@50", dict(m0=0, rank_early=3, m_switch=50)),
            ("r3_to_8@80", dict(m0=0, rank_early=3, m_switch=80))]
        for name, kw in configs:
            torch.cuda.synchronize(); t0 = time.perf_counter()
            x, factors, a1 = evolve(z, **kw)
            torch.cuda.synchronize(); dt = time.perf_counter() - t0
            g_x = H.terminal_gradient(x, y, R, angles)
            sc = H.score(factors, 0.0, g_x, g_exact)
            row["configs"][name] = dict(err=sc["err_a0"], cos=sc["cos_a0"], seconds=dt,
                                        factor_steps=N - kw["m0"])
        rows.append(row)
        print(f"sample {sample} it {k:2d} resid {100*rel:6.1f}%  " +
              "  ".join(f"{n}: {100*v['err']:.1f}%/{v['seconds']:.1f}s" for n, v in row["configs"].items()), flush=True)
        (ROOT / "outputs/parallel_bug_test/late_start.json").write_text(json.dumps(rows, indent=1))
print("LATE START SWEEP DONE")
