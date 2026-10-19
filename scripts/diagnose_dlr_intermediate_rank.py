"""Spectrum of the evolving low-rank part Y_k = J_k - (1 - t_k) I of the exact
discrete Jacobian J_k = prod_{j<k}(I + h A_j), at several k, for N Euler steps.
Reports the relative error of the best rank-8 approximation of Y_k (spectral
tail among the top-k computed values) -- what a rank-8 evolving model must carry."""
import argparse, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import benchmark_dflow_dlr_sparse_ct_circles_gpu as bench
import test_parallel_bug_integrator as helper
pa = argparse.ArgumentParser()
pa.add_argument("--resolution", type=int, default=64); pa.add_argument("--steps", type=int, required=True)
pa.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
pa.add_argument("--k", type=int, default=40); pa.add_argument("--device", default="cuda:1")
pa.add_argument("--out", type=Path, required=True)
args = pa.parse_args()
R, N = args.resolution, args.steps; h = 1.0 / N; d = R * R
dev = torch.device(args.device); torch.cuda.set_device(dev)
model = bench.load_model(R, 100, dev, "outputs/fm_unet/circles-{res}-noattn-ch16-b8/n1")
data = torch.load(ROOT / f"data/circles-{R}-n1.pt", map_location="cpu", weights_only=True, mmap=True)["test"]
angles = torch.linspace(0, torch.pi, 16, device=dev)[:-1]
z, _ = helper.make_problem(R, 0, data, angles, dev)
def partial(zz, k):
    x = zz
    for j in range(k):
        x = x + h * bench.velocity(model, j * h, x)
    return x
res = {}
for f in args.fractions:
    k = round(f * N); a = 1.0 - k * h
    def Y(V):   # (J_k - a I) V, batched columns
        Z = z.expand(V.shape[0], -1, -1, -1).contiguous()
        with torch.no_grad():
            return torch.func.jvp(lambda zz: partial(zz, k), (Z,), (V,))[1] - a * V
    def YT(Q):
        Z = z.expand(Q.shape[0], -1, -1, -1).contiguous().requires_grad_(True)
        return torch.autograd.grad((partial(Z, k) * Q).sum(), Z)[0].detach() - a * Q
    def ap(fn, M):
        return torch.cat([fn(M[:, s:s+16].T.reshape(-1, 1, R, R)).reshape(-1, d).T for s in range(0, M.shape[1], 16)], 1)
    g = torch.Generator(device=dev).manual_seed(0)
    Q, _ = torch.linalg.qr(ap(Y, torch.randn(d, args.k + 8, device=dev, generator=g)))
    for _ in range(2):
        Q2, _ = torch.linalg.qr(ap(YT, Q)); Q, _ = torch.linalg.qr(ap(Y, Q2))
    s = torch.linalg.svdvals(ap(YT, Q).T)[:args.k]
    tail8 = (s[8:].square().sum() / s.square().sum()).sqrt().item()
    res[str(k)] = {"t": k * h, "singular_values": s.tolist(), "rank8_tail_rel_fro_topk": tail8}
    print(f"N={N} k={k} (t={k*h:.3f}): sigma {[round(v,2) for v in s[:10].tolist()]} ... s[{args.k-1}]={s[-1]:.3f}  rank-8 tail {100*tail8:.1f}%", flush=True)
args.out.write_text(json.dumps(res, indent=2))
