"""Why is the DLR gradient anti-aligned at a near-converged point?

At 64px, sample 1, exact-gradient Armijo L-BFGS iterate 20 (12% residual),
every integrator variant gave 117% action error with cosine -0.985 and a
trace-based alpha_1 of +15.  Hypothesis: the running trace alpha_t is polluted
by a few huge singular values, so the residual Y = J - alpha I gets a -alpha
on thousands of directions that the rank-8 truncation turns into an
anti-aligned gradient.  Test: same point, KSL evolution with
  trace      the default Hutchinson tr(J_t)/d (m probes)
  analytic   alpha_t = 1 - t  (alpha_1 = 0)
  frozen     alpha_t = 1      (Y = J - I, the notes' simplest split)
  rank32     default trace, rank 32 (does more rank fix it?)
and report error / cosine with alpha_1 = 0 and with the evolved alpha_1.
Also a 64-probe Hutchinson estimate of the exact tr(J_1)/d via autograd JVPs.
"""
import json, math, sys, time
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import test_parallel_bug_integrator as H
import benchmark_dflow_dlr_sparse_ct_circles_gpu as bench
from dlr_dflow import enriched as E
from dlr_dflow.graphed_products import GraphedAProducts

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:1"); torch.cuda.set_device(dev)
R, m = 64, 4
model, _ = H.load(R, dev)
products = GraphedAProducts(model, torch.zeros(1, 1, R, R, device=dev))
angles = torch.linspace(0, math.pi, 16, device=dev)[:-1]
data = torch.load(ROOT / f"data/circles-{R}-n1.pt", map_location="cpu", weights_only=True, mmap=True)["test"]
out = {}
for sample, iterate in ((1, 20), (0, 20)):
    z0, y = H.make_problem(R, sample, data, angles, dev)
    z = H.exact_iterates(model, z0, y, R, angles, [iterate])[iterate]
    g_exact, rel, x_exact = H.exact_gradient(model, z, y, R, angles)
    # exact tr(J_1)/d by Hutchinson on autograd JVPs through the flow
    zz = z.detach().clone()
    tr = 0.0; probes = 64
    gen = torch.Generator(device=dev).manual_seed(7)
    for _ in range(probes):
        v = (torch.randint(0, 2, zz.shape, generator=gen, device=dev) * 2 - 1).float()
        _, jv = torch.func.jvp(lambda s: bench.solve_flow(model, s, H.STEPS, "forward_euler"), (zz,), (v,))
        tr += (v * jv).sum().item()
    exact_trace_over_d = tr / (probes * zz.numel())
    row = {"residual": rel, "z_over_sqrt_d": z.norm().item() / math.sqrt(z.numel()),
           "armijo_initial_step_policy": bench.ArmijoLBFGS.initial_step_policy,
           "armijo_progress_policy": bench.ArmijoLBFGS.progress_policy,
           "exact_trace_over_d": exact_trace_over_d, "variants": {}}
    variants = {
        "trace":    dict(rank=8),
        "analytic": dict(rank=8, alpha_rate=lambda k, pd, U: -1.0),
        "frozen":   dict(rank=8, alpha_rate=lambda k, pd, U: 0.0),
        "rank32":   dict(rank=32),
    }
    for name, kw in variants.items():
        x, factors, a1 = E.evolve_trace_centered_enriched(
            model, z, H.STEPS, trace_probes=m, initialization_oversampling=8,
            enrichment_probes=4, factor_stride=1, products=products,
            zero_terminal_baseline=False, **kw)
        g_x = H.terminal_gradient(x, y, R, angles)
        sc = H.score(factors, a1, g_x, g_exact); sc["alpha1"] = float(a1)
        row["variants"][name] = sc
        print(f"s{sample} it{iterate} {name:9s} err(a1=0) {sc['err_a0']:.3f} cos {sc['cos_a0']:+.3f}  "
              f"err(own a1={a1:+.3g}) {sc['err_own']:.3f}", flush=True)
    print(f"s{sample} it{iterate}: residual {rel:.3f}, exact tr(J1)/d = {exact_trace_over_d:.3f}", flush=True)
    out[f"s{sample}_it{iterate}"] = row
Path(ROOT / "outputs/parallel_bug_test").mkdir(exist_ok=True)
(ROOT / "outputs/parallel_bug_test/alpha_anomaly.json").write_text(json.dumps(out, indent=1))
print("ALPHA ANOMALY CHECK DONE")
