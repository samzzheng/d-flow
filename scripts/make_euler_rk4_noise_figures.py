"""Figures for the 250px 1%-noise Euler-vs-RK4 sweep, built incrementally.

Two modes, both safe to run while the sweep is still going:

  --sample N   pair figure for one sample: convergence of both integrators plus
               the final metrics, written to figures/pair_sample_NNN.png
  (no flag)    progress figure across every completed pair: paired image errors,
               win/loss tally, per-method distributions, and cost, written to
               figures/progress.png (plus progress.csv)

Per-(sample, method) 3x3 diagnostics are produced by the benchmark itself under
each worker's figures/res_250/sample_NNN/; this script adds the comparison view.
"""
from __future__ import annotations

import argparse, csv, json
from collections import defaultdict
from pathlib import Path

import cmocean
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
try:
    matplotlib.rcParams["font.family"] = "TeX Gyre Pagella"
    matplotlib.rcParams["mathtext.fontset"] = "custom"
    matplotlib.rcParams["mathtext.rm"] = "TeX Gyre Pagella"
except Exception:
    pass

FIELD = cmocean.cm.dense_r
COLOUR = {"forward_euler": FIELD(0.75), "rk4": FIELD(0.35)}
LABEL = {"forward_euler": "Forward Euler (100 evals/flow)",
         "rk4": "RK4 (400 evals/flow)"}
SHORT = {"forward_euler": "Euler", "rk4": "RK4"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path,
                   default=ROOT / "outputs/euler_rk4_250_noise001_cpu")
    p.add_argument("--sample", type=int, help="build the pair figure for this sample")
    p.add_argument("--recon", action="store_true",
                   help="also build the side-by-side reconstruction comparison")
    p.add_argument("--all-samples", action="store_true",
                   help="loop over every sample present, then the progress figure")
    p.add_argument("--noise-level", type=float, default=0.01)
    return p.parse_args()


def load(root: Path):
    """Collect samples.csv / iterations.csv from every per-sample worker dir."""
    samples, iterations = {}, defaultdict(list)
    for path in sorted(root.glob("sample_*/samples.csv")):
        for row in csv.DictReader(path.open()):
            samples[(int(row["sample_index"]), row["method"])] = row
    for path in sorted(root.glob("sample_*/iterations.csv")):
        for row in csv.DictReader(path.open()):
            iterations[(int(row["sample_index"]), row["method"])].append(
                (int(row["outer_step"]), float(row["measurement_relative_l2"])))
    for key in iterations:
        iterations[key].sort()
    return samples, iterations


def pair_figure(root: Path, sample: int, samples, iterations, noise_level: float):
    have = [m for m in ("forward_euler", "rk4") if (sample, m) in samples]
    if len(have) < 2:
        return False
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3))

    for method in have:
        steps, resid = zip(*iterations[(sample, method)])
        axes[0].semilogy(steps, [100 * r for r in resid], color=COLOUR[method],
                         label=LABEL[method], linewidth=1.8)
    axes[0].axhline(100 * noise_level, color="0.35", linestyle=":", linewidth=1.2)
    axes[0].annotate(f"{100*noise_level:g}% noise floor", (0.02, 100 * noise_level),
                     xycoords=("axes fraction", "data"), va="bottom", fontsize=8,
                     color="0.35")
    axes[0].set_xlabel("LBFGS outer step")
    axes[0].set_ylabel("measurement residual (%)")
    axes[0].set_title("Convergence", fontsize=10)
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(frameon=False, fontsize=8)

    image = [100 * float(samples[(sample, m)]["image_relative_l2"]) for m in have]
    meas = [100 * float(samples[(sample, m)]["measurement_relative_l2_noisy"]) for m in have]
    index = np.arange(len(have))
    axes[1].bar(index, image, 0.55, color=[COLOUR[m] for m in have])
    for i, v in enumerate(image):
        axes[1].text(i, v, f" {v:.2f}%", ha="center", va="bottom", fontsize=9)
    axes[1].set_xticks(index); axes[1].set_xticklabels([SHORT[m] for m in have])
    axes[1].set_ylabel("image relative L2 (%)")
    axes[1].set_title("Reconstruction error", fontsize=10)
    axes[1].grid(alpha=0.25, axis="y")

    axes[2].bar(index, meas, 0.55, color=[COLOUR[m] for m in have])
    axes[2].axhline(100 * noise_level, color="0.35", linestyle=":", linewidth=1.2)
    for i, v in enumerate(meas):
        axes[2].text(i, v, f" {v:.3g}%", ha="center", va="bottom", fontsize=9)
    axes[2].set_yscale("log")
    axes[2].set_xticks(index); axes[2].set_xticklabels([SHORT[m] for m in have])
    axes[2].set_ylabel("measurement residual (%)")
    axes[2].set_title("Final data fit (below floor = overfit)", fontsize=10)
    axes[2].grid(alpha=0.25, axis="y", which="both")

    fig.suptitle(f"250×250 sparse CT, {100*noise_level:g}% noise — sample {sample}",
                 fontsize=12)
    fig.tight_layout()
    out = root / "figures"; out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"pair_sample_{sample:03d}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    return True


# Panel boxes (x0, y0, x1, y1 as fractions) within the benchmark's 3x3 diagnostic
# PNG. The benchmark saves no arrays, so the reconstructions can only be recovered
# by cropping its rendered figure.
GT_BOX = (0.006, 0.088, 0.232, 0.357)
PROP_BOX = (0.006, 0.388, 0.232, 0.658)
ERR_BOX = (0.006, 0.692, 0.232, 0.958)
NOISY_SINO_BOX = (0.700, 0.088, 0.926, 0.357)
PRED_SINO_BOX = (0.352, 0.388, 0.578, 0.658)
ERROR_CMAP = cmocean.cm.amp


def _fields(root: Path, sample: int, method: str):
    """Saved fields for one run, or None for runs predating --npz output."""
    path = _diagnostic(root, sample, method)
    if path is None:
        return None
    npz = path.with_suffix(".npz")
    if not npz.exists():
        return None
    with np.load(npz) as data:
        return {k: data[k][0, 0] for k in data.files}


def _panel(path: Path, box):
    import matplotlib.image as mpimg
    im = mpimg.imread(path)
    x0, y0, x1, y1 = box
    h, w = im.shape[:2]
    return im[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def _diagnostic(root: Path, sample: int, method: str):
    hits = sorted(root.glob(f"sample_*/figures/res_250/sample_{sample:03d}/{method}.png"))
    return hits[0] if hits else None


def _errors(samples, sample, method, fields=None):
    """Image and measurement relative L2 as percentages.

    samples.csv reports measurement error on the *unclamped* field while the
    image error and every displayed panel use the clamped one, so the recorded
    number can read 0.0004% for a reconstruction that misfits by 161%. With the
    saved fields the honest clamped-vs-noisy fit is computed here instead.
    """
    row = samples[(sample, method)]
    img = 100 * float(row["image_relative_l2"])
    if fields is not None:
        pred, noisy = fields["predicted_measurement"], fields["noisy_measurement"]
        meas = 100 * float(np.linalg.norm(pred - noisy) / np.linalg.norm(noisy))
    else:
        meas = 100 * float(row["measurement_relative_l2_clean"])
    return img, meas


def recon_figure(root: Path, sample: int, samples, noise_level: float):
    """Columns GT / Euler / RK4: reconstruction, measurement, absolute image error.

    Uses the saved fields when present. Runs predating the .npz output fall back
    to cropping the rendered diagnostic, which cannot supply a true absolute
    error map, so that row is dropped and the caller is told.
    """
    methods = ("forward_euler", "rk4")
    paths = {m: _diagnostic(root, sample, m) for m in methods}
    if not all(paths.values()) or not all((sample, m) in samples for m in methods):
        return False
    fields = {m: _fields(root, sample, m) for m in methods}
    have_fields = all(v is not None for v in fields.values())

    rows = 3 if have_fields else 2
    fig, axes = plt.subplots(rows, 3, figsize=(11.5, 3.9 * rows))

    if have_fields:
        gt = fields["rk4"]["ground_truth"]
        errors = {m: np.abs(fields[m]["reconstruction"] - gt) for m in methods}
        vmax = max(e.max() for e in errors.values()) or 1.0
        axes[0, 0].imshow(gt, cmap=FIELD, vmin=0, vmax=1)
        axes[1, 0].imshow(fields["rk4"]["noisy_measurement"], cmap=FIELD, aspect="auto")
    else:
        axes[0, 0].imshow(_panel(paths["rk4"], GT_BOX))
        axes[1, 0].imshow(_panel(paths["rk4"], NOISY_SINO_BOX))
    axes[0, 0].set_title("Ground truth", fontsize=11)
    axes[1, 0].set_title(f"Observed $Ax$ ({100*noise_level:g}% noise)", fontsize=10)

    for col, method in enumerate(methods, start=1):
        img, meas = _errors(samples, sample, method, fields[method])
        if have_fields:
            axes[0, col].imshow(fields[method]["reconstruction"], cmap=FIELD, vmin=0, vmax=1)
            axes[1, col].imshow(fields[method]["predicted_measurement"], cmap=FIELD,
                                aspect="auto")
        else:
            axes[0, col].imshow(_panel(paths[method], PROP_BOX))
            axes[1, col].imshow(_panel(paths[method], PRED_SINO_BOX))
        axes[0, col].set_title(
            f"{SHORT[method]}\nimage rel. $\\ell_2$ = {img:.2f}%", fontsize=10)
        axes[1, col].set_title(
            f"{SHORT[method]} reconstructed $Ax$\n"
            f"meas. rel. $\\ell_2$ = {meas:.3g}%", fontsize=10)

    if have_fields:
        axes[2, 0].axis("off")
        for col, method in enumerate(methods, start=1):
            im = axes[2, col].imshow(errors[method], cmap=ERROR_CMAP, vmin=0, vmax=vmax)
            axes[2, col].set_title(
                f"{SHORT[method]} $|\\hat{{x}} - x|$   max {errors[method].max():.3f}",
                fontsize=10)
        cax = fig.add_axes([0.13, 0.055, 0.22, 0.016])
        fig.colorbar(im, cax=cax, orientation="horizontal", label="absolute image error")

    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(
        f"250×250 sparse CT, 15 angles, {100*noise_level:g}% noise — sample {sample}",
        fontsize=12.5, y=1.00)
    fig.tight_layout()
    out = root / "figures"; out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"recon_sample_{sample:03d}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    return True if have_fields else "no-fields"


def progress_figure(root: Path, samples, iterations, noise_level: float):
    paired = sorted({s for (s, m) in samples
                     if (s, "forward_euler") in samples and (s, "rk4") in samples})
    if not paired:
        print("no completed pairs yet"); return
    e = np.array([100 * float(samples[(s, "forward_euler")]["image_relative_l2"]) for s in paired])
    r = np.array([100 * float(samples[(s, "rk4")]["image_relative_l2"]) for s in paired])
    em = np.array([100 * float(samples[(s, "forward_euler")]["measurement_relative_l2_noisy"]) for s in paired])
    rm = np.array([100 * float(samples[(s, "rk4")]["measurement_relative_l2_noisy"]) for s in paired])
    et = np.array([float(samples[(s, "forward_euler")]["elapsed_s"]) for s in paired])
    rt = np.array([float(samples[(s, "rk4")]["elapsed_s"]) for s in paired])

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0))

    ax = axes[0, 0]
    lim = [min(e.min(), r.min()) * 0.7, max(e.max(), r.max()) * 1.4]
    ax.loglog(lim, lim, color="0.6", linestyle="--", linewidth=1.0)
    ax.scatter(e, r, s=34, color=FIELD(0.55), edgecolor="white", linewidth=0.6, zorder=3)
    for s, x, y in zip(paired, e, r):
        ax.annotate(str(s), (x, y), fontsize=6, xytext=(3, 3),
                    textcoords="offset points", color="0.35")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Euler image error (%)"); ax.set_ylabel("RK4 image error (%)")
    ax.set_title(f"Paired image error — below the line, RK4 wins ({len(paired)} samples)",
                 fontsize=10)
    ax.grid(alpha=0.25, which="both")

    ax = axes[0, 1]
    wins = int((r < e).sum()); losses = int((r > e).sum())
    ratio = e / r
    ax.bar([0, 1], [wins, losses], 0.5, color=[COLOUR["rk4"], COLOUR["forward_euler"]])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["RK4 better", "Euler better"])
    ax.set_ylabel("samples")
    ax.set_title(f"Win/loss — RK4 {wins}/{len(paired)}   "
                 f"median error ratio {np.median(ratio):.2f}×", fontsize=10)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1, 0]
    ax.boxplot([e, r], labels=["Euler", "RK4"], widths=0.5)
    ax.scatter(np.full_like(e, 1.0), e, s=18, color=COLOUR["forward_euler"], alpha=0.7, zorder=3)
    ax.scatter(np.full_like(r, 2.0), r, s=18, color=COLOUR["rk4"], alpha=0.7, zorder=3)
    ax.set_yscale("log"); ax.set_ylabel("image relative L2 (%)")
    ax.set_title("Distribution of reconstruction error", fontsize=10)
    ax.grid(alpha=0.25, axis="y", which="both")

    ax = axes[1, 1]
    ax.boxplot([em, rm], labels=["Euler", "RK4"], widths=0.5)
    ax.axhline(100 * noise_level, color="0.35", linestyle=":", linewidth=1.2)
    ax.set_yscale("log"); ax.set_ylabel("final measurement residual (%)")
    ax.set_title(f"Data fit vs {100*noise_level:g}% noise floor "
                 f"(below = overfit)", fontsize=10)
    ax.grid(alpha=0.25, axis="y", which="both")

    fig.suptitle(f"250×250 sparse CT, {100*noise_level:g}% noise, attention-free flow — "
                 f"{len(paired)}/30 pairs complete", fontsize=12)
    fig.tight_layout()
    out = root / "figures"; out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "progress.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    with (out / "progress.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample", "euler_image_rel_l2_pct", "rk4_image_rel_l2_pct",
                    "euler_meas_pct", "rk4_meas_pct", "euler_s", "rk4_s", "rk4_better"])
        for i, s in enumerate(paired):
            w.writerow([s, f"{e[i]:.4f}", f"{r[i]:.4f}", f"{em[i]:.5f}", f"{rm[i]:.5f}",
                        f"{et[i]:.0f}", f"{rt[i]:.0f}", int(r[i] < e[i])])
    (out / "progress.json").write_text(json.dumps({
        "pairs_complete": len(paired), "rk4_wins": wins, "euler_wins": losses,
        "median_error_ratio_euler_over_rk4": float(np.median(ratio)),
        "median_euler_image_pct": float(np.median(e)),
        "median_rk4_image_pct": float(np.median(r)),
        "median_rk4_cost_multiple": float(np.median(rt / et)),
        "samples_overfit_euler": int((em < 100 * noise_level / 2).sum()),
        "samples_overfit_rk4": int((rm < 100 * noise_level / 2).sum()),
    }, indent=2) + "\n")
    print(f"progress: {len(paired)}/30 pairs, RK4 better on {wins}, "
          f"median ratio {np.median(ratio):.2f}x, RK4 cost {np.median(rt/et):.1f}x")


def main():
    args = parse_args()
    samples, iterations = load(args.root)
    targets = []
    if args.all_samples:
        targets = sorted({s for (s, _) in samples})
    elif args.sample is not None:
        targets = [args.sample]
    for sample in targets:
        ok = pair_figure(args.root, sample, samples, iterations, args.noise_level)
        print(f"pair figure sample {sample}: {'written' if ok else 'incomplete'}")
        if args.recon:
            ok = recon_figure(args.root, sample, samples, args.noise_level)
            note = {True: "written", False: "incomplete",
                    "no-fields": "written WITHOUT absolute-error row "
                                 "(run predates saved fields)"}[ok]
            print(f"recon figure sample {sample}: {note}")
    progress_figure(args.root, samples, iterations, args.noise_level)


if __name__ == "__main__":
    main()
