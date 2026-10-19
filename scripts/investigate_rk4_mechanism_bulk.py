"""Bulk (random-direction) gain of the full-flow Jacobian per generator, at z0
and at the two solution sources: E||J v|| for unit random v, plus the
null-space fraction of the image response.  Cheap: a few batched JVPs."""
import json, sys, math
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
import investigate_rk4_mechanism as M
import benchmark_dflow_dlr_sparse_ct_circles_gpu as bench
dev = torch.device(sys.argv[1]); torch.cuda.set_device(dev)
model = M.load_model(250, dev); P = M.problem(250, 1, dev); proj = M.Projector(250, dev)
sources = {"z0": P["source0"],
           "E96": torch.load("outputs/rk4_noise_floor_2026_09_27/sample1_euler/accepted/state_00096.pt", map_location="cpu", weights_only=False)["source"].to(dev),
           "R94": torch.load("outputs/rk4_noise_floor_2026_09_27/sample1_rk4/accepted/state_00094.pt", map_location="cpu", weights_only=False)["source"].to(dev)}
gen = torch.Generator(device=dev).manual_seed(7)
V = torch.randn(16, 1, 250, 250, device=dev, generator=gen); V = V / V.flatten(1).norm(dim=1).view(-1,1,1,1)
out = {}
for sname, z in sources.items():
    for g in ("euler100", "rk425", "euler400", "rk4100"):
        method, steps = M.parse_generator(g)
        Z = z.expand(16, -1, -1, -1).contiguous()
        with torch.no_grad():
            _, JV = torch.func.jvp(lambda zz: bench.solve_flow(model, zz, steps, method, False), (Z,), (V,))
        gains = JV.flatten(1).norm(dim=1)
        inv = torch.stack([proj.split(JV[i].flatten())[1].norm() for i in range(16)]) / gains
        out[f"{sname}|{g}"] = {"rms_gain": gains.square().mean().sqrt().item(), "mean_gain": gains.mean().item(),
                               "min_gain": gains.min().item(), "max_gain": gains.max().item(),
                               "null_space_fraction": inv.mean().item()}
        print(f"{sname:4s} {g:9s} rms gain {out[f'{sname}|{g}']['rms_gain']:.4f}  (min {gains.min():.4f} max {gains.max():.4f})  null-space frac {inv.mean():.3f}", flush=True)
M.write_json(out, M.OUT / "spectra" / "bulk_gain_s1.json")
