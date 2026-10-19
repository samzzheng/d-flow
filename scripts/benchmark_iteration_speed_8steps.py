"""Per-iteration speed at 8 Euler steps across resolutions: exact D-Flow
(uncheckpointed / activation-checkpointed) vs optimized DLR D-Flow
(analytic alpha, rank 8, fused kernels, 6/4 chunks; with / without local
UNet block checkpointing). One resolution per process; the four arms are
interleaved in random order after one warmup each. One "iteration" = one
source-gradient evaluation (image + sparse-CT terminal gradient + source
gradient); an Armijo forward trial is timed separately."""
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
pa.add_argument("--resolution", type=int, required=True)
pa.add_argument("--device", default="cuda:1")
pa.add_argument("--steps", type=int, default=8)
pa.add_argument("--repeats", type=int, default=5)
pa.add_argument("--out", type=Path, required=True)
args = pa.parse_args()
R, N = args.resolution, args.steps
dev = torch.device(args.device); torch.cuda.set_device(dev); torch.set_num_threads(4)
model = bench.load_model(R, 100, dev, "outputs/fm_unet/circles-{res}-noattn-ch16-b8/n1")
assert not model.mid_attn and not any(model.att_channels)
bench.configure_dlr_kernels(R)
tangent.GN_ACTION_BACKEND = "triton"; tangent.GN_PRIMAL_BACKEND = "full"
enriched.SPECIALIZE_ZERO_ENRICHMENT = True; enriched.SKIP_UNTRUNCATED_SVD = True
core.SHARE_LINEARIZATION = True
angles = torch.linspace(0, torch.pi, 16, device=dev)[:-1]
data = torch.load(ROOT / f"data/circles-{R}-n1.pt", map_location="cpu", weights_only=True, mmap=True)["test"]
z, target = helper.make_problem(R, 0, data, angles, dev)
providers = {ck: GraphedAProducts(model, z, share_linearization=True, full_step_graphs=True,
                 max_jvp_columns=20, max_vjp_columns=16, max_step_jvp_columns=6,
                 max_step_vjp_columns=4, checkpoint_blocks=ck) for ck in (False, True)}

def dflow(checkpointed):
    v = z.detach().requires_grad_(True)
    x = bench.solve_flow(model, v, N, "forward_euler", checkpointed)
    g, = torch.autograd.grad(x, v, helper.terminal_gradient(x, target, R, angles).reshape_as(x))
    return x.detach(), g.flatten()

def dlr(checkpointed):
    x, factors, _ = core.sensitivity_factors(model, z, N, rank=8, trace_probes=0, enrichment_probes=0,
        initialization_oversampling=8, seed=0, factor_stride=1, zero_terminal_baseline=True,
        products=providers[checkpointed], checkpoint_blocks=checkpointed)
    return x, core.approx_source_gradient(factors, helper.terminal_gradient(x, target, R, angles), 0.)

def forward():
    with torch.no_grad():
        return bench.solve_flow(model, z, N, "forward_euler", False), None

arms = {"dflow": lambda: dflow(False), "dflow_checkpointed": lambda: dflow(True),
        "dlr": lambda: dlr(False), "dlr_checkpointed": lambda: dlr(True), "forward_trial": forward}
report = {"protocol": vars(args) | {"out": str(args.out), "rank": 8, "trace_probes": 0, "dlr": "optimized fast profile, analytic alpha",
          "gpu_load": "shared with another user's job"}, "calls": [], "peak_GiB": {}}
ref = None
failed = set()
for name, fn in list(arms.items()):           # warmup + per-arm peak memory
    torch.cuda.synchronize(dev); torch.cuda.reset_peak_memory_stats(dev)
    before = torch.cuda.memory_allocated(dev)
    try:
        x, g = fn(); torch.cuda.synchronize(dev)
    except torch.cuda.OutOfMemoryError:
        report["peak_GiB"][name] = None; failed.add(name); arms.pop(name)
        import gc; gc.collect(); torch.cuda.empty_cache(); continue
    report["peak_GiB"][name] = (torch.cuda.max_memory_allocated(dev) - before) / 2**30
    if name == "dflow_checkpointed": ref = g
    if g is not None and name not in ("dflow", "dflow_checkpointed"):
        report.setdefault("grad_rel_err_vs_dflow", {})[name] = ((g - ref).norm() / ref.norm()).item()
rng = random.Random(R)
for rep in range(args.repeats):
    order = list(arms); rng.shuffle(order)
    for name in order:
        torch.cuda.synchronize(dev); t0 = time.perf_counter()
        arms[name](); torch.cuda.synchronize(dev)
        report["calls"].append({"arm": name, "seconds": time.perf_counter() - t0})
report["median_seconds"] = {n: statistics.median(c["seconds"] for c in report["calls"] if c["arm"] == n) for n in arms}
report["out_of_memory"] = sorted(failed)
args.out.write_text(json.dumps(report, indent=2))
print(R, json.dumps({k: round(v, 4) for k, v in report["median_seconds"].items()}),
      json.dumps({k: round(v, 3) for k, v in report["peak_GiB"].items()}),
      json.dumps({k: round(100*v, 3) for k, v in report.get("grad_rel_err_vs_dflow", {}).items()}), flush=True)
