"""Why is the DLR gradient inaccurate at 8 Euler steps?
(1) spectrum of the exact discrete Jacobian J_1 = prod (I + h A_k) and the
    error of its best rank-r approximation applied to the actual gradient;
(2) DLR error vs rank and trace probes;
(3) top singular value of each one-step map I + h A_k (power iteration):
    values > 1 mean that step amplifies some perturbation."""
import argparse, json, math, sys
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
pa.add_argument("--steps", type=int, required=True)
pa.add_argument("--device", default="cuda:1")
pa.add_argument("--k", type=int, default=48)
pa.add_argument("--batch", type=int, default=16)
pa.add_argument("--out", type=Path, required=True)
args = pa.parse_args()
R, N = args.resolution, args.steps; h = 1.0 / N
dev = torch.device(args.device); torch.cuda.set_device(dev); torch.set_num_threads(4)
model = bench.load_model(R, 100, dev, "outputs/fm_unet/circles-{res}-noattn-ch16-b8/n1")
bench.configure_dlr_kernels(R)
tangent.GN_ACTION_BACKEND = "triton"; tangent.GN_PRIMAL_BACKEND = "full"
enriched.SPECIALIZE_ZERO_ENRICHMENT = True; enriched.SKIP_UNTRUNCATED_SVD = True
core.SHARE_LINEARIZATION = True
angles = torch.linspace(0, torch.pi, 16, device=dev)[:-1]
data = torch.load(ROOT / f"data/circles-{R}-n1.pt", map_location="cpu", weights_only=True, mmap=True)["test"]
z, target = helper.make_problem(R, 0, data, angles, dev)
d = R * R
flow = lambda zz: bench.solve_flow(model, zz, N, "forward_euler", False)

# exact gradient
v = z.detach().requires_grad_(True); x = flow(v)
gx = helper.terminal_gradient(x.detach(), target, R, angles).reshape(-1)
gz, = torch.autograd.grad(x, v, gx.reshape_as(x)); gz = gz.flatten()

def apply(fn, M):
    out = []
    for s in range(0, M.shape[1], args.batch):
        blk = M[:, s:s + args.batch].T.reshape(-1, 1, R, R)
        out.append(fn(blk).reshape(blk.shape[0], -1).T)
    return torch.cat(out, 1)
def jvp(V):
    Z = z.expand(V.shape[0], -1, -1, -1).contiguous()
    with torch.no_grad():
        return torch.func.jvp(flow, (Z,), (V,))[1]
def vjp(Q):
    Z = z.expand(Q.shape[0], -1, -1, -1).contiguous().requires_grad_(True)
    X = bench.solve_flow(model, Z, N, "forward_euler", True)
    return torch.autograd.grad((X * Q).sum(), Z)[0].detach()

# (1) randomized SVD of J_1, two power iterations
g = torch.Generator(device=dev).manual_seed(0)
Q, _ = torch.linalg.qr(apply(jvp, torch.randn(d, args.k + 8, device=dev, generator=g)))
for _ in range(2):
    Qz, _ = torch.linalg.qr(apply(vjp, Q)); Q, _ = torch.linalg.qr(apply(jvp, Qz))
B = apply(vjp, Q)                                   # J^T Q
Us, s, Vt = torch.linalg.svd(B.T, full_matrices=False)
U = Q @ Us; V = Vt.T
rel = lambda a, b: ((a - b).norm() / b.norm()).item()
trunc = {r: rel(V[:, :r] @ (s[:r] * (U[:, :r].T @ gx)), gz) for r in (3, 8, 16, 32, args.k)}
# fraction of J^T g_x lying outside the top-k right singular space
outside_k = ((gz - V @ (V.T @ gz)).norm() / gz.norm()).item()

# (2) DLR variants
dlr = {}
prov = GraphedAProducts(model, z, share_linearization=True, full_step_graphs=True, max_jvp_columns=20,
                        max_vjp_columns=16, max_step_jvp_columns=6, max_step_vjp_columns=4)
for rank, probes in ((8, 0), (8, 4), (16, 0), (32, 0)):
    xd, fac, _ = core.sensitivity_factors(model, z, N, rank=rank, trace_probes=probes, enrichment_probes=0,
        initialization_oversampling=8, seed=0, factor_stride=1, zero_terminal_baseline=True, products=prov)
    gd = core.approx_source_gradient(fac, helper.terminal_gradient(xd, target, R, angles), 0.)
    dlr[f"rank{rank}_probes{probes}"] = rel(gd, gz)

# (3) per-step top singular value of I + h A_k along the trajectory
step_gain = []
xk = z.detach()
for k in range(N):
    t = k * h
    step = lambda y: y + h * bench.velocity(model, t, y)
    q = torch.randn(1, 1, R, R, device=dev, generator=g); q /= q.norm()
    for _ in range(20):
        w = torch.func.jvp(step, (xk,), (q,))[1]
        y0 = xk.detach().requires_grad_(True)
        q = torch.autograd.grad(step(y0), y0, w)[0].detach(); q /= q.norm()
    sigma = torch.func.jvp(step, (xk,), (q,))[1].norm().item()
    with torch.no_grad():
        xk = step(xk)
    step_gain.append({"t": t, "top_singular_value": sigma})

out = {"resolution": R, "steps": N, "singular_values": s[:args.k].tolist(), "truncation_error": trunc,
       "grad_outside_top_k": outside_k, "dlr_error": dlr, "step_gain": step_gain}
args.out.write_text(json.dumps(out, indent=2))
print(f"R={R} N={N}: sigma {[round(v,2) for v in s[:10].tolist()]} ... s[{args.k-1}]={s[args.k-1]:.3f}")
print(f"  best rank-r error on the gradient: " + " ".join(f"r{r}={100*e:.1f}%" for r, e in trunc.items()) + f"  (outside top-{args.k}: {100*outside_k:.1f}%)")
print(f"  DLR error: " + " ".join(f"{k}={100*v:.1f}%" for k, v in dlr.items()))
print(f"  per-step top singular value of I+hA: " + " ".join(f"{e['top_singular_value']:.2f}" for e in step_gain[-min(8,N):]), flush=True)
