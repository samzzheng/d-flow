"""Flow-step sweep, stage A: per-gradient cost and accuracy of optimized
analytic-alpha DLR (rank 8, 0 probes) vs number of Euler steps N, at 250px.

For each N: median DLR gradient time and forward-only (Armijo trial) time,
interleaved across N in one process; DLR gradient error against exact
checkpointed D-Flow at the same N; and the Euler-N image's distance to an
RK4-400 reference at the same source (the generator's own ODE error).
Default TF32 settings, as in dlr_execution_speed_2026_09_28.md."""
import argparse, gc, json, random, statistics, sys, time
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import benchmark_dflow_dlr_sparse_ct_circles_gpu as bench
import test_parallel_bug_integrator as helper
from dlr_dflow import dlr_dflow as core, enriched, tangent_unet as tangent
from dlr_dflow.graphed_products import GraphedAProducts

pa = argparse.ArgumentParser()
pa.add_argument("--device", default="cuda:1")
pa.add_argument("--steps", type=int, nargs="+", default=[10, 20, 30, 50, 75, 100, 150])
pa.add_argument("--repeats", type=int, default=3)
pa.add_argument("--out", type=Path, default=ROOT / "outputs/flow_steps_sweep_2026_09_28/stage_a.json")
args = pa.parse_args()
dev = torch.device(args.device); torch.cuda.set_device(dev); torch.set_num_threads(4)
model = bench.load_model(250, 100, dev, "outputs/fm_unet/circles-{res}-noattn-ch16-b8/n1")
bench.configure_dlr_kernels(250)
tangent.GN_ACTION_BACKEND = "triton"; tangent.GN_PRIMAL_BACKEND = "full"
enriched.SPECIALIZE_ZERO_ENRICHMENT = True; enriched.SKIP_UNTRUNCATED_SVD = True
core.SHARE_LINEARIZATION = True
angles = torch.linspace(0, torch.pi, 16, device=dev)[:-1]
data = torch.load(ROOT / "data/circles-250-n1.pt", map_location="cpu", weights_only=True, mmap=True)["test"]
_, target = helper.make_problem(250, 0, data, angles, dev)
srcs = torch.load(ROOT / "outputs/accuracy_fast_euler_2026_09_27/exact_from_z0/sources_sample0_exact.pt",
                  map_location="cpu", weights_only=True)["sources"]
points = {"z0": srcs[0].to(dev), "iterate20": srcs[20].to(dev)}
with torch.no_grad():
    reference = {k: bench.solve_flow(model, z, 400, "rk4", False) for k, z in points.items()}

def dlr(products, z, n):
    x, factors, _ = core.sensitivity_factors(model, z, n, rank=8, trace_probes=0, enrichment_probes=0,
        initialization_oversampling=8, seed=0, factor_stride=1, zero_terminal_baseline=True, products=products)
    return x, core.approx_source_gradient(factors, helper.terminal_gradient(x, target, 250, angles), 0.)

def exact(z, n):
    v = z.detach().requires_grad_(True)
    x = bench.solve_flow(model, v, n, "forward_euler", True)
    g, = torch.autograd.grad(x, v, helper.terminal_gradient(x, target, 250, angles).reshape_as(x))
    return x.detach(), g.flatten()

rel = lambda a, b: ((a - b).norm() / b.norm()).item()
report = {"protocol": vars(args) | {"out": str(args.out), "rank": 8, "trace_probes": 0,
          "gpu_load": "shared with another user's job; compare times across N, not to idle baselines"},
          "per_steps": {}, "calls": []}
def make_provider():
    return GraphedAProducts(model, points["z0"], share_linearization=True, full_step_graphs=True,
        max_jvp_columns=20, max_vjp_columns=16, max_step_jvp_columns=6, max_step_vjp_columns=4)

def release(provider):
    del provider
    gc.collect(); torch.cuda.synchronize(dev); torch.cuda.empty_cache()

for n in args.steps:
    row = {}
    provider = make_provider()
    for name, z in points.items():
        xe, ge = exact(z, n)
        x, g = dlr(provider, z, n)
        row[name] = {"dlr_grad_rel_err": rel(g, ge), "dlr_grad_cos": (g @ ge / (g.norm() * ge.norm())).item(),
                     "euler_image_vs_rk4_400": rel(xe, reference[name])}
    release(provider)
    report["per_steps"][str(n)] = row
    print(f"N={n:4d}: grad err z0 {100*row['z0']['dlr_grad_rel_err']:.2f}%  it20 {100*row['iterate20']['dlr_grad_rel_err']:.2f}% | "
          f"ODE err z0 {100*row['z0']['euler_image_vs_rk4_400']:.2f}%  it20 {100*row['iterate20']['euler_image_vs_rk4_400']:.2f}%", flush=True)
    args.out.write_text(json.dumps(report, indent=2))
z = points["iterate20"]
rng = random.Random(20260928)
for rnd in range(2):                      # two shuffled rounds over N to average load drift
    order = list(args.steps); rng.shuffle(order)
    for n in order:
        provider = make_provider()
        dlr(provider, z, n); torch.cuda.synchronize(dev)          # warmup / capture
        for rep in range(args.repeats):
            for kind in ("dlr", "forward"):
                torch.cuda.synchronize(dev); t0 = time.perf_counter()
                if kind == "dlr":
                    dlr(provider, z, n)
                else:
                    with torch.no_grad():
                        bench.solve_flow(model, z, n, "forward_euler", False)
                torch.cuda.synchronize(dev)
                report["calls"].append({"round": rnd, "steps": n, "kind": kind, "seconds": time.perf_counter() - t0})
        release(provider)
for n in args.steps:
    for kind in ("dlr", "forward"):
        report["per_steps"][str(n)][f"{kind}_seconds"] = statistics.median(
            c["seconds"] for c in report["calls"] if c["steps"] == n and c["kind"] == kind)
    r = report["per_steps"][str(n)]
    print(f"N={n:4d}: DLR gradient {r['dlr_seconds']:.3f} s  forward {r['forward_seconds']:.3f} s", flush=True)
report["peak_reserved_GiB"] = torch.cuda.max_memory_reserved(dev) / 2**30
args.out.write_text(json.dumps(report, indent=2))
