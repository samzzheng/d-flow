"""DLR-D-Flow rank sweep: gradient accuracy and evolution time vs rank.

Current algorithm (enriched KSL evolution, 4 trace probes, 4 enrichment probes,
100 Euler steps, CUDA graphs), ranks 1..10, 64px, sample 0 (quick test).  Scored on the
terminal transpose action against autograd (alpha_1 = 0, the benchmark default),
at the starting source and the capped-Armijo exact-gradient iterate 10.  Time is
the wall clock of one full evolution (one DLR gradient) on a shared GPU; the
first evolution at each rank is excluded (CUDA-graph capture for new widths).
Usage: python scripts/test_rank_sweep.py [cuda:1]
"""
import json, math, statistics, sys, time
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import test_parallel_bug_integrator as H
from dlr_dflow import enriched as E
from dlr_dflow.graphed_products import GraphedAProducts

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:1"); torch.cuda.set_device(dev)
RANKS = list(range(1, 11))
M, P, OVER = 4, 4, 8
PLAN = {64: [0]}
out = {}
for R, samples in PLAN.items():
    model, tag = H.load(R, dev)
    products = GraphedAProducts(model, torch.zeros(1, 1, R, R, device=dev))
    angles = torch.linspace(0, math.pi, 16, device=dev)[:-1]
    data = torch.load(ROOT / f"data/circles-{R}-n1.pt", map_location="cpu", weights_only=True, mmap=True)["test"]
    rows, times = [], {r: [] for r in RANKS}
    z0, _ = H.make_problem(R, samples[0], data, angles, dev)
    for r in RANKS:                                   # warm-up: capture graphs for this rank's widths
        E.evolve_trace_centered_enriched(model, z0, H.STEPS, rank=r, trace_probes=M,
                                         initialization_oversampling=OVER, enrichment_probes=P,
                                         factor_stride=1, products=products, zero_terminal_baseline=True)
    for s in samples:
        z0, y = H.make_problem(R, s, data, angles, dev)
        pts = H.exact_iterates(model, z0, y, R, angles, [10])
        for k in (0, 10):
            z = pts[k]
            g_exact, rel, _ = H.exact_gradient(model, z, y, R, angles)
            row = {"sample": s, "iterate": k, "residual": rel,
                   "armijo_initial_step_policy": H.bench.ArmijoLBFGS.initial_step_policy,
                   "armijo_progress_policy": H.bench.ArmijoLBFGS.progress_policy,
                   "ranks": {}}
            for r in RANKS:
                torch.cuda.synchronize(); t0 = time.perf_counter()
                x, factors, a1 = E.evolve_trace_centered_enriched(
                    model, z, H.STEPS, rank=r, trace_probes=M, initialization_oversampling=OVER,
                    enrichment_probes=P, factor_stride=1, products=products, zero_terminal_baseline=False)
                torch.cuda.synchronize(); dt = time.perf_counter() - t0
                g_x = H.terminal_gradient(x, y, R, angles)
                sc = H.score(factors, a1, g_x, g_exact)
                row["ranks"][r] = dict(err=sc["err_a0"], cos=sc["cos_a0"], err_own=sc["err_own"], seconds=dt)
                times[r].append(dt)
            rows.append(row)
            print(f"{R}px sample {s} it {k:2d} resid {100*rel:6.1f}%  " +
                  " ".join(f"r{r}:{100*row['ranks'][r]['err']:.1f}%" for r in RANKS), flush=True)
    out[str(R)] = {"model": tag, "points": rows,
                   "median_seconds": {r: statistics.median(v) for r, v in times.items()},
                   "columns_per_step": {r: M + 2 * r + 3 * P for r in RANKS}}
    (ROOT / "outputs/parallel_bug_test/rank_sweep.json").write_text(json.dumps(out, indent=1))
    print(f"{R}px median s/evolution: " + " ".join(f"r{r}:{statistics.median(v):.2f}" for r, v in times.items()), flush=True)
    del products, model; torch.cuda.empty_cache()
print("RANK SWEEP DONE")
