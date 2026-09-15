"""Assets for the 2026-09-09 report: Euler-vs-RK4 noise panels + benchmark plots."""
import json, math, sys
from pathlib import Path
import torch, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cmocean

ROOT = Path("/home/samzheng/d-flow")
ASSETS = ROOT / "outputs/reports/2026-09-09_assets"
ASSETS.mkdir(parents=True, exist_ok=True)
SP = Path("/tmp/claude-1005/-home-samzheng-d-flow/"
          "a92c707f-8bcf-47a6-aa24-5ca3654db5b6/scratchpad/dlr_speed")

def radon(x, n_angles=15):
    angles = torch.linspace(0, math.pi, n_angles + 1)[:-1]
    b, c, h, w = x.shape; n = angles.numel()
    rep = x[:, None].expand(b, n, c, h, w).reshape(b * n, c, h, w)
    ab = angles.repeat(b)
    th = torch.zeros(b * n, 2, 3, dtype=x.dtype)
    th[:, 0, 0], th[:, 0, 1] = torch.cos(ab), -torch.sin(ab)
    th[:, 1, 0], th[:, 1, 1] = torch.sin(ab), torch.cos(ab)
    g = F.affine_grid(th, rep.shape, align_corners=False)
    r = F.grid_sample(rep, g, mode="bilinear", padding_mode="zeros", align_corners=False)
    return r.mean(dim=-2).reshape(b, n, c, w).permute(0, 2, 1, 3)

def panel_figure(x_true, y, x_hat, path):
    """Standard 2x3 diagnostic: GT / recon / |image error| over y / Ax_hat / |residual|."""
    Ax = radon(x_hat)
    gt, rec = x_true[0, 0].numpy(), x_hat[0, 0].numpy()
    obs, pred = y[0, 0].numpy(), Ax[0, 0].numpy()
    m_lo = min(obs.min(), pred.min()); m_hi = max(obs.max(), pred.max())
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.4))
    panels = [
        (axes[0, 0], gt, "Ground truth", cmocean.cm.dense_r, 0.0, 1.0, "Intensity"),
        (axes[0, 1], rec, "Reconstruction", cmocean.cm.dense_r, 0.0, 1.0, "Intensity"),
        (axes[0, 2], abs(rec - gt), "|image error|", cmocean.cm.amp, 0.0, 1.0, "Abs. error"),
        (axes[1, 0], obs, "Observed $Ax$ (noisy)", cmocean.cm.dense_r, m_lo, m_hi, "Projection"),
        (axes[1, 1], pred, "Reconstructed $Ax$", cmocean.cm.dense_r, m_lo, m_hi, "Projection"),
        (axes[1, 2], abs(pred - obs), "|measurement residual|", cmocean.cm.amp, 0.0, 1.0, "Abs. error"),
    ]
    for ax, img, title, cmap, vmin, vmax, lab in panels:
        art = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(art, ax=ax, fraction=0.046, pad=0.04); cb.set_label(lab, fontsize=8)
        cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)

# ---- 1. noise-test panels (2 res x 5 samples x 2 integrators) ----
R = json.load(open(SP / "noise_rk4.json"))
picks = []
for res in (200, 250):
    T = torch.load(SP / f"noise_rk4_res{res}.pt", weights_only=True)
    for meth in ("euler", "rk4"):
        rows = sorted((R[f"{res}_s{si}_{meth}"] for si in range(5)),
                      key=lambda r: r["image_rel"])
        for label, row in (("best", rows[0]), ("median", rows[2]), ("worst", rows[4])):
            si = row["sample"]
            fn = f"noise01_res{res}_{meth}_{label}.jpg"
            panel_figure(T[f"s{si}"]["x_true"], T[f"s{si}"]["y"],
                         T[f"s{si}"][meth], ASSETS / fn)
            picks.append(dict(res=res, method=meth, label=label, sample=si, file=fn,
                              image_rel=row["image_rel"], meas_rel=row["meas_rel"]))
            print("wrote", fn, flush=True)
json.dump(picks, open(ASSETS / "noise_picks.json", "w"), indent=1)

print("ASSETSDONE")
