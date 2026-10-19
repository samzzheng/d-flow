"""Optimized fast DLR: 4 Hutchinson trace probes vs analytic alpha_t = 1 - t
(trace_probes=0, 8 instead of 12 JVP columns per step), interleaved in one
process so shared-GPU load affects both arms equally. Accuracy is checked
against checkpointed exact D-Flow at the saved sample-0 sources.
Settings otherwise match dlr_execution_speed_2026_09_28.md."""
import argparse, json, random, statistics, sys, time
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
pa.add_argument("--repeats", type=int, default=5)
pa.add_argument("--iterates", type=int, nargs="+", default=[0, 20])
pa.add_argument("--memory-fraction", type=float, default=0.2)
pa.add_argument("--out", type=Path, default=ROOT / "outputs/dlr_execution_2026_09_28/analytic_alpha/results.json")
args = pa.parse_args()
dev = torch.device(args.device); torch.cuda.set_device(dev)
torch.cuda.set_per_process_memory_fraction(args.memory_fraction, dev)
torch.set_num_threads(4)
model = bench.load_model(250, 100, dev, "outputs/fm_unet/circles-{res}-noattn-ch16-b8/n1")
bench.configure_dlr_kernels(250)
tangent.GN_ACTION_BACKEND = "triton"; tangent.GN_PRIMAL_BACKEND = "full"
enriched.SPECIALIZE_ZERO_ENRICHMENT = True; enriched.SKIP_UNTRUNCATED_SVD = True
core.SHARE_LINEARIZATION = True
angles = torch.linspace(0, torch.pi, 16, device=dev)[:-1]
data = torch.load(ROOT / "data/circles-250-n1.pt", map_location="cpu", weights_only=True, mmap=True)["test"]
_, target = helper.make_problem(250, 0, data, angles, dev)
sources = torch.load(ROOT / "outputs/accuracy_fast_euler_2026_09_27/exact_from_z0/sources_sample0_exact.pt",
                     map_location="cpu", weights_only=True)["sources"]
products = GraphedAProducts(model, sources[0].to(dev), share_linearization=True, full_step_graphs=True,
                            max_jvp_columns=20, max_vjp_columns=16, max_step_jvp_columns=6, max_step_vjp_columns=4)

def dlr(source, probes):
    x, factors, _ = core.sensitivity_factors(model, source, 100, rank=8, trace_probes=probes,
        enrichment_probes=0, initialization_oversampling=8, seed=0, factor_stride=1,
        zero_terminal_baseline=True, products=products)
    terminal = helper.terminal_gradient(x, target, 250, angles)
    return x, core.approx_source_gradient(factors, terminal, 0.)

def exact(source):
    v = source.detach().requires_grad_(True)
    x = bench.solve_flow(model, v, 100, "forward_euler", True)
    g, = torch.autograd.grad(x, v, helper.terminal_gradient(x, target, 250, angles).reshape_as(x))
    return x.detach(), g.flatten()

report = {"protocol": vars(args) | {"out": str(args.out), "rank": 8, "steps": 100, "jvp_columns": {"probes4": 12, "analytic": 8},
          "step_chunks": [6, 4], "gpu_load": "shared with another user's job (contended)"}, "accuracy": {}, "calls": []}
for it in args.iterates:
    s = sources[it].to(dev)
    xe, ge = exact(s)
    row = {}
    for name, p in (("probes4", 4), ("analytic", 0)):
        x, g = dlr(s, p)
        row[name] = {"grad_rel_err": ((g - ge).norm() / ge.norm()).item(),
                     "cosine": (g @ ge / (g.norm() * ge.norm())).item(),
                     "image_diff_vs_exact": ((x - xe).norm() / xe.norm()).item()}
    report["accuracy"][str(it)] = row
    print(f"iterate {it}: " + "  ".join(f"{k} err {v['grad_rel_err']*100:.3f}% cos {v['cosine']:.6f}" for k, v in row.items()), flush=True)

s = sources[20].to(dev)
schedule = [("probes4", 4), ("analytic", 0)]
for name, p in schedule:                      # warmup/capture per arm
    dlr(s, p); torch.cuda.synchronize(dev)
rng = random.Random(20260928)
for rep in range(args.repeats):
    order = schedule[:]; rng.shuffle(order)
    for name, p in order:
        torch.cuda.synchronize(dev); t0 = time.perf_counter()
        dlr(s, p); torch.cuda.synchronize(dev)
        report["calls"].append({"arm": name, "seconds": time.perf_counter() - t0})
med = {n: statistics.median(c["seconds"] for c in report["calls"] if c["arm"] == n) for n, _ in schedule}
report["medians"] = med; report["ratio_analytic_over_probes4"] = med["analytic"] / med["probes4"]
report["peak_reserved_GiB"] = torch.cuda.max_memory_reserved(dev) / 2**30
args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(report, indent=2))
print(json.dumps({"medians": med, "ratio": report["ratio_analytic_over_probes4"]}))
