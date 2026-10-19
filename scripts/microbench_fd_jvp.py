"""Finite-difference JVP columns vs the hand-written tangent pass: cost and accuracy.

For k random unit columns Q at a mid-flow state x_t (t = 0.5), compares
    hand      unet_jvp (the current evolution's JVP: one shared primal + k tangent passes)
    fd2       central differences [v(x+eps q) - v(x-eps q)] / 2 eps   (2k batched forwards)
    fd1       one-sided [v(x+eps q) - v(x)] / eps                       (k forwards + shared v(x))
    fwd       one batched forward of k inputs (the "1x per element" unit)
against torch.func.jvp (fp32) and, at 64px, a float64 torch.func.jvp reference.
eps is per column: eps_rel * ||x|| / ||q||.  Timing: interleaved, median of reps,
eager (no CUDA graphs) -- launch overhead inflates every arm equally at 64px.
Usage: python scripts/microbench_fd_jvp.py [cuda:1]
"""
import json, math, statistics, sys, time
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import test_parallel_bug_integrator as H
import dlr_dflow.dlr_dflow as dd
from dlr_dflow.tangent_unet import unet_jvp

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:1"); torch.cuda.set_device(dev)
K, REPS, T = 16, 10, 0.5
out = {}
for R in (64, 250):
    model, tag = H.load(R, dev)
    z = torch.randn(1, 1, R, R, generator=torch.Generator(device=dev).manual_seed(0), device=dev)
    with torch.no_grad():
        x = dd.euler_flow(model, z, 50) if False else None
    # mid-flow state: 50 of 100 Euler steps
    x = z.clone()
    with torch.no_grad():
        for i in range(50):
            x = x + 0.01 * dd.velocity(model, i * 0.01, x)
    g = torch.Generator(device=dev).manual_seed(1)
    Q = (torch.randint(0, 2, (K, 1, R, R), generator=g, device=dev) * 2 - 1).float()
    Q = Q / Q.flatten(1).norm(dim=1).view(K, 1, 1, 1)                  # unit columns
    tvec = torch.full((K,), T, device=dev)
    fwd = lambda X: model(X, tvec[:X.shape[0]])

    def fd(eps_rel, central):
        eps = eps_rel * x.norm()                                         # ||q|| = 1
        with torch.no_grad():
            if central:
                plus, minus = fwd(x + eps * Q), fwd(x - eps * Q)
                return (plus - minus) / (2 * eps)
            base = fwd(x)
            return (fwd(x + eps * Q) - base) / eps

    def hand():
        with torch.no_grad():
            return unet_jvp(model, T, x, Q)[1]

    def func_jvp(dtype=torch.float32):
        m = model
        if dtype != torch.float32:
            m = H.load(R, dev)[0].to(dtype)
            # the sinusoidal timestep embedding is built in float32 inside the model
            m.time_proj.register_forward_pre_hook(lambda mod, inp: (inp[0].to(dtype),))
        xx, QQ = x.to(dtype), Q.to(dtype)
        t1 = torch.full((1,), T, device=dev, dtype=dtype)
        f = lambda X: m(X, t1)
        _, jv = torch.func.vmap(lambda q: torch.func.jvp(f, (xx,), (q[None],)), out_dims=(None, 0))(QQ)
        return jv.reshape(K, 1, R, R).float()

    ref32 = func_jvp()
    ref = func_jvp(torch.float64) if R == 64 else ref32
    err = lambda A: ((A - ref).norm() / ref.norm()).item()
    acc = {"func_fp32_vs_ref": err(ref32), "hand": err(hand())}
    for e in (1e-2, 1e-3, 1e-4, 1e-5):
        acc[f"fd2_eps{e:g}"] = err(fd(e, True))
        acc[f"fd1_eps{e:g}"] = err(fd(e, False))

    arms = {"fwd_k": lambda: fwd(Q), "hand": hand,
            "fd2_eps1e-3": lambda: fd(1e-3, True), "fd1_eps1e-3": lambda: fd(1e-3, False)}
    times = {k: [] for k in arms}
    for k, f in arms.items():
        with torch.no_grad(): f()
    for _ in range(REPS):
        for k, f in arms.items():
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.no_grad(): f()
            torch.cuda.synchronize(); times[k].append(time.perf_counter() - t0)
    med = {k: statistics.median(v) for k, v in times.items()}
    unit = med["fwd_k"] / K
    row = {"model": tag, "k": K, "accuracy_rel_err": acc,
           "median_s": med, "per_column_in_forward_elements":
           {k: v / unit for k, v in med.items()}}
    out[str(R)] = row
    print(f"\n=== {R}px {tag}  k={K} t={T}")
    print("accuracy (rel. Frobenius error vs reference):")
    for k, v in acc.items(): print(f"   {k:20s} {v:.3e}")
    print("cost per column, in units of one batched-forward element:")
    for k, v in row["per_column_in_forward_elements"].items(): print(f"   {k:14s} {v:6.2f}x   ({med[k]*1e3:7.1f} ms per {K} cols)")
    del model; torch.cuda.empty_cache()
Path(ROOT / "outputs/parallel_bug_test").mkdir(exist_ok=True)
(ROOT / "outputs/parallel_bug_test/fd_jvp_microbench.json").write_text(json.dumps(out, indent=1))
print("FD MICROBENCH DONE")
