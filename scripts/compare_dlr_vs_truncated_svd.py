"""64px: DLR Jacobian J_hat = U S V^T (rank 8, analytic alpha, alpha_1 = 0)
versus the exact discrete Jacobian J = dG/dz of the N-step Euler flow and its
best rank-8 approximation J_8 (truncated SVD). Full J is formed column by
column with forward-mode JVPs (d = 4096). Source points are saved accepted
states of an exact Euler-100 reconstruction (epoch-100 attention-free model)."""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dlr_dflow.api import load_flow_model, sensitivity_factors, terminal_gradient

BASE_SEED = 20260807


def sparse_radon(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    n = angles.numel()
    rep = x[:, None].expand(b, n, c, h, w).reshape(b * n, c, h, w)
    a = angles.repeat(b)
    theta = torch.zeros(b * n, 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 1, 0], theta[:, 1, 1] = a.cos(), -a.sin(), a.sin(), a.cos()
    grid = F.affine_grid(theta, rep.shape, align_corners=False)
    proj = F.grid_sample(rep, grid, mode="bilinear", padding_mode="zeros", align_corners=False).mean(dim=-2)
    return proj.reshape(b, n, c, w).permute(0, 2, 1, 3)


def euler_flow(model: torch.nn.Module, z: torch.Tensor, steps: int) -> torch.Tensor:
    h, x = 1.0 / steps, z
    for k in range(steps):
        x = x + h * model(x, torch.full((x.shape[0],), k * h, device=x.device, dtype=x.dtype))
    return x


def main() -> None:
    pa = argparse.ArgumentParser()
    pa.add_argument("--device", default="cuda:1")
    pa.add_argument("--steps", type=int, nargs="+", default=[150, 100, 8])
    pa.add_argument("--rank", type=int, default=8)
    pa.add_argument("--batch", type=int, default=128)
    args = pa.parse_args()
    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    R, r = 64, args.rank
    d = R * R
    model, _ = load_flow_model(R, 1, 100, str(dev), ROOT / "outputs/fm_unet/circles-64-noattn-ch16-b8/n1")
    runs = ROOT / "outputs/rk4_mechanism_2026_09_27/solves64"
    points = {"s3_start": (3, runs / "res64_s3_euler100/accepted/state_00000.pt"),
              "s3_mid": (3, runs / "res64_s3_euler100/accepted/state_00040.pt"),
              "s3_final": (3, runs / "res64_s3_euler100/accepted/state_00160.pt"),
              "s0_final": (0, runs / "res64_s0_euler100/accepted/state_00109.pt")}
    angles = torch.linspace(0, math.pi, 16, device=dev)[:-1]
    data = torch.load(ROOT / "data/circles-64-n1.pt", map_location="cpu", weights_only=True, mmap=True)["test"]
    eye = torch.eye(d, device=dev)
    out_path = ROOT / "outputs/dlr_vs_tsvd_2026_09_30/results.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for n in args.steps:
        for name, (sample, path) in points.items():
            if f"N{n}_{name}" in results:
                continue
            z = torch.load(path, map_location="cpu", weights_only=False)["source"].to(dev).reshape(1, 1, R, R)
            truth = data[sample:sample + 1].float().div(255.0).to(dev)
            clean = sparse_radon(truth, angles).cpu()
            noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(BASE_SEED + R * 1000 + sample))
            y = (clean + noise / noise.norm() * 0.01 * clean.norm()).to(dev)
            # exact J, column by column: J[:, j] = J e_j
            cols = []
            with torch.no_grad():
                for s in range(0, d, args.batch):
                    e = eye[s:s + args.batch].reshape(-1, 1, R, R)
                    zz = z.expand(e.shape[0], -1, -1, -1).contiguous()
                    cols.append(torch.func.jvp(lambda q: euler_flow(model, q, n), (zz,), (e,))[1].reshape(e.shape[0], d).T)
            J = torch.cat(cols, 1).double()
            U8, s_all, V8t = torch.linalg.svd(J)
            J8 = (U8[:, :r] * s_all[:r]) @ V8t[:r]
            with torch.no_grad():
                x1 = euler_flow(model, z, n)
            x_dlr, (U, S, V), _ = sensitivity_factors(model, z, n, rank=r, trace_probes=0, enrichment_probes=0,
                                                       initialization_oversampling=8, seed=0, factor_stride=1,
                                                       zero_terminal_baseline=True)
            Jh = (U.double() @ S.double()) @ V.double().T
            gx = terminal_gradient(x1, lambda x: sparse_radon(x, angles), y).reshape(-1).double()
            gz = J.T @ gx
            fro = lambda M: torch.linalg.matrix_norm(M).item()
            spec = lambda M: torch.linalg.matrix_norm(M, ord=2).item()
            Ud, sd, Vdt = torch.linalg.svd(Jh)
            left_cos = torch.linalg.svdvals(U8[:, :r].T @ Ud[:, :r]).tolist()
            right_cos = torch.linalg.svdvals(V8t[:r] @ Vdt[:r].T).tolist()
            row = {
                "fro_error_dlr": fro(J - Jh) / fro(J), "fro_error_tsvd": fro(J - J8) / fro(J),
                "spec_error_dlr": spec(J - Jh) / s_all[0].item(), "spec_error_tsvd": s_all[r].item() / s_all[0].item(),
                "dlr_vs_tsvd_fro": fro(Jh - J8) / fro(J8),
                "grad_error_dlr": ((Jh.T @ gx - gz).norm() / gz.norm()).item(),
                "grad_error_tsvd": ((J8.T @ gx - gz).norm() / gz.norm()).item(),
                "true_singular_values": s_all[:r + 4].tolist(), "dlr_singular_values": sd[:r].tolist(),
                "left_subspace_cosines": left_cos, "right_subspace_cosines": right_cos,
                "image_diff_dlr_primal": ((x_dlr - x1).norm() / x1.norm()).item(),
            }
            results[f"N{n}_{name}"] = row
            print(f"N={n:3d} {name:9s} | ||J-J_dlr||F {100*row['fro_error_dlr']:6.2f}%  ||J-J_8||F {100*row['fro_error_tsvd']:6.2f}%"
                  f" | spec {100*row['spec_error_dlr']:6.2f}% vs {100*row['spec_error_tsvd']:6.2f}%"
                  f" | ||J_dlr-J_8||F {100*row['dlr_vs_tsvd_fro']:6.2f}%"
                  f" | grad {100*row['grad_error_dlr']:6.2f}% vs {100*row['grad_error_tsvd']:6.2f}%", flush=True)
            print(f"      sigma true {[round(v, 3) for v in row['true_singular_values'][:r]]}\n"
                  f"      sigma DLR  {[round(v, 3) for v in row['dlr_singular_values']]}\n"
                  f"      left cos  {[round(v, 3) for v in left_cos]}\n      right cos {[round(v, 3) for v in right_cos]}", flush=True)
            out_path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
