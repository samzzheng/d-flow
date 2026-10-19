"""Is the difference a property of the generators' samples, independent of the
inverse problem?  (1) Unconditional samples from the same z through each
generator: how binary / how sharp-edged are they vs the data?  (2) Radial edge
profile of the sample-1 reconstructions at the noise floor vs the truth."""
import json, math, sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import investigate_rk4_mechanism as M
import benchmark_dflow_dlr_sparse_ct_circles_gpu as bench
dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:1")
torch.cuda.set_device(dev)
OUT = M.OUT / "samples"; OUT.mkdir(exist_ok=True)
res = 250
model = M.load_model(res, dev)
data = torch.load(ROOT / f"data/circles-{res}-n1.pt", map_location="cpu", weights_only=True, mmap=True)["test"]
truth = bench.normalize_images(data[:16]).to(dev)

def stats(x):
    x = x.clamp(0, 1)
    mid = ((x > 0.05) & (x < 0.95)).float().flatten(1).mean(1)          # fraction of intermediate pixels
    binar = (x - x.round()).flatten(1).norm(dim=1) / x.flatten(1).norm(dim=1)  # distance to nearest binary image
    # edge width proxy: intermediate pixels per unit of estimated circumference
    area = (x > 0.5).float().flatten(1).sum(1)
    circ = 2 * math.pi * (area / math.pi).sqrt()
    return {"intermediate_fraction": mid.tolist(), "binary_distance": binar.tolist(),
            "intermediate_pixels_per_circumference": ((x > 0.05) & (x < 0.95)).float().flatten(1).sum(1).div(circ).tolist()}

results = {"data": stats(truth), "generators": {}}
gen = torch.Generator(device=dev).manual_seed(123)
z = torch.randn(8, 1, res, res, device=dev, generator=gen)
images = {}
with torch.no_grad():
    for g in ("euler100", "rk425", "euler400", "rk4100", "rk4800"):
        method, steps = M.parse_generator(g)
        x = bench.solve_flow(model, z, steps, method, False)
        images[g] = x
        results["generators"][g] = stats(x)
        s = results["generators"][g]
        print(f"{g:9s} intermediate frac {np.mean(s['intermediate_fraction']):.5f}  binary dist {np.mean(s['binary_distance']):.4f}  "
              f"edge px/circ {np.mean(s['intermediate_pixels_per_circumference']):.3f}", flush=True)
    d = results["data"]
    print(f"{'data':9s} intermediate frac {np.mean(d['intermediate_fraction']):.5f}  binary dist {np.mean(d['binary_distance']):.4f}  "
          f"edge px/circ {np.mean(d['intermediate_pixels_per_circumference']):.3f}")
    ref = images["rk4800"]
    for g in ("euler100", "rk425", "euler400", "rk4100"):
        diff = images[g] - ref
        rel = (diff.flatten(1).norm(dim=1) / ref.flatten(1).norm(dim=1)).mean().item()
        # where is the difference? fraction of its energy on the edge band of the reference
        edge = ((ref > 0.05) & (ref < 0.95)).float()
        band = torch.nn.functional.max_pool2d(edge, 5, stride=1, padding=2)   # dilate 2 px
        frac_edge = ((diff * band).flatten(1).norm(dim=1) ** 2 / diff.flatten(1).norm(dim=1) ** 2).mean().item()
        results["generators"][g]["rel_diff_to_rk4800"] = rel
        results["generators"][g]["diff_energy_fraction_in_edge_band"] = frac_edge
        print(f"{g:9s} vs rk4800: rel diff {100*rel:.3f}%  ({100*frac_edge:.1f}% of it within 2px of the edge)")
torch.save({"z": z.cpu(), **{k: v.cpu() for k, v in images.items()}}, OUT / "unconditional_samples.pt")

# ---- radial edge profiles of the sample-1 reconstructions at the floor ----
P = M.problem(res, 1, torch.device("cpu"))
t = P["truth"][0, 0]
ys, xs = torch.meshgrid(torch.arange(res).float(), torch.arange(res).float(), indexing="ij")
mass = t.sum(); cy, cx = (ys * t).sum() / mass, (xs * t).sum() / mass
R = (mass / math.pi).sqrt()
r = ((ys - cy) ** 2 + (xs - cx) ** 2).sqrt()
profiles = {"center": [cy.item(), cx.item()], "radius": R.item(), "bins": [], "truth": [], "euler": [], "rk4": []}
recon = {"euler": torch.load("outputs/rk4_noise_floor_2026_09_27/sample1_euler/accepted/state_00096.pt", map_location="cpu", weights_only=False)["raw_reconstruction"][0, 0],
         "rk4": torch.load("outputs/rk4_noise_floor_2026_09_27/sample1_rk4/accepted/state_00094.pt", map_location="cpu", weights_only=False)["raw_reconstruction"][0, 0]}
edges = torch.arange(R.item() - 6, R.item() + 6, 0.25)
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (r >= lo) & (r < hi)
    profiles["bins"].append(((lo + hi) / 2 - R).item())
    profiles["truth"].append(t[m].mean().item())
    for k, v in recon.items():
        profiles[k].append(v[m].mean().item())
results["edge_profile_sample1"] = profiles
print("radial profile (distance from fitted edge, mean intensity: truth / euler / rk4):")
for b, a, e, k in zip(profiles["bins"], profiles["truth"], profiles["euler"], profiles["rk4"]):
    if abs(b) < 3: print(f"  {b:+5.2f}px  {a:.4f}  {e:.4f}  {k:.4f}")
M.write_json(results, OUT / "sample_stats.json")
