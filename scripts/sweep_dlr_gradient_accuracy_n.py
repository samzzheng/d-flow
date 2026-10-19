"""How small can N be before the DLR gradient degrades? 250px, optimized DLR
(analytic alpha, rank 8). Error of the DLR source gradient against exact
D-Flow at the same N, at several source points, plus idle-GPU timings."""
import gc, json, statistics, sys, time
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import benchmark_dflow_dlr_sparse_ct_circles_gpu as bench
import test_parallel_bug_integrator as helper
from dlr_dflow import dlr_dflow as core, enriched, tangent_unet as tangent
from dlr_dflow.graphed_products import GraphedAProducts
dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:0"); torch.cuda.set_device(dev); torch.set_num_threads(4)
STEPS = [int(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else [30, 40, 50, 60, 75, 100, 125, 150, 200]
R = 250
model = bench.load_model(R, 100, dev, "outputs/fm_unet/circles-{res}-noattn-ch16-b8/n1")
bench.configure_dlr_kernels(R)
tangent.GN_ACTION_BACKEND = "triton"; tangent.GN_PRIMAL_BACKEND = "full"
enriched.SPECIALIZE_ZERO_ENRICHMENT = True; enriched.SKIP_UNTRUNCATED_SVD = True
core.SHARE_LINEARIZATION = True
angles = torch.linspace(0, torch.pi, 16, device=dev)[:-1]
data = torch.load(ROOT / f"data/circles-{R}-n1.pt", map_location="cpu", weights_only=True, mmap=True)["test"]
traj = torch.load(ROOT / "outputs/accuracy_fast_euler_2026_09_27/exact_from_z0/sources_sample0_exact.pt",
                  map_location="cpu", weights_only=True)["sources"]
points = {}
targets = {}
for s in (0, 1, 3, 16):
    z0, y = helper.make_problem(R, s, data, angles, dev)
    points[f"s{s}_z0"] = z0; targets[f"s{s}_z0"] = y
for it in (5, 10, 20):
    points[f"s0_it{it}"] = traj[it].to(dev); targets[f"s0_it{it}"] = targets["s0_z0"]

def exact(z, y, n, ckpt):
    v = z.detach().requires_grad_(True)
    x = bench.solve_flow(model, v, n, "forward_euler", ckpt)
    g, = torch.autograd.grad(x, v, helper.terminal_gradient(x, y, R, angles).reshape_as(x))
    return g.flatten()

def dlr(prov, z, y, n):
    x, f, _ = core.sensitivity_factors(model, z, n, rank=8, trace_probes=0, enrichment_probes=0,
        initialization_oversampling=8, seed=0, factor_stride=1, zero_terminal_baseline=True, products=prov)
    return core.approx_source_gradient(f, helper.terminal_gradient(x, y, R, angles), 0.)

def timed(fn, reps=3):
    fn(); torch.cuda.synchronize(dev); out = []
    for _ in range(reps):
        torch.cuda.synchronize(dev); t0 = time.perf_counter(); fn(); torch.cuda.synchronize(dev)
        out.append(time.perf_counter() - t0)
    return statistics.median(out)

out = {"points": list(points), "steps": {}}
for n in STEPS:
    prov = GraphedAProducts(model, points["s0_z0"], share_linearization=True, full_step_graphs=True,
        max_jvp_columns=20, max_vjp_columns=16, max_step_jvp_columns=6, max_step_vjp_columns=4)
    row = {"error": {}, "cosine": {}}
    for name, z in points.items():
        ge = exact(z, targets[name], n, True)
        gd = dlr(prov, z, targets[name], n)
        row["error"][name] = ((gd - ge).norm() / ge.norm()).item()
        row["cosine"][name] = (gd @ ge / (gd.norm() * ge.norm())).item()
    z, y = points["s0_it10"], targets["s0_it10"]
    row["dlr_s"] = timed(lambda: dlr(prov, z, y, n))
    row["dflow_ckpt_s"] = timed(lambda: exact(z, y, n, True))
    if n <= 150:
        try: row["dflow_s"] = timed(lambda: exact(z, y, n, False))
        except torch.cuda.OutOfMemoryError: row["dflow_s"] = None
    del prov; gc.collect(); torch.cuda.empty_cache()
    out["steps"][str(n)] = row
    e = row["error"]
    print(f"N={n:4d} err: " + " ".join(f"{k}={100*v:.1f}" for k, v in e.items()) +
          f" | median {100*statistics.median(e.values()):.1f}% | DLR {row['dlr_s']:.3f}s  D-Flow {row.get('dflow_s') or float('nan'):.3f}s  ckpt {row['dflow_ckpt_s']:.3f}s", flush=True)
    Path(ROOT / "outputs/dlr_small_n_diagnosis_2026_09_28" / (sys.argv[3] if len(sys.argv) > 3 else "gradient_accuracy_vs_n.json")).write_text(json.dumps(out, indent=2))
