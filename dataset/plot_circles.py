"""Visualise sample grids from the fixed-count 64×64 circles datasets."""
import argparse
from pathlib import Path

import cmocean
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.rcParams["font.family"] = "TeX Gyre Pagella"
matplotlib.rcParams["mathtext.fontset"] = "custom"
matplotlib.rcParams["mathtext.rm"] = "TeX Gyre Pagella"

_CMAP = cmocean.cm.dense_r


def plot_all_counts(
    data_dir: Path = Path("data"),
    n_cols: int = 8,
    split: str = "train",
    seed: int = 0,
) -> plt.Figure:
    """4-row grid (one row per circle count), colorbar on the right."""
    rng = np.random.default_rng(seed)
    counts = [1, 2, 3, 4]

    fig = plt.figure(figsize=(n_cols * 1.05 + 1.0, len(counts) * 1.05 + 0.4))
    gs  = fig.add_gridspec(
        len(counts), n_cols + 1,
        hspace=0.04, wspace=0.04,
        width_ratios=[1] * n_cols + [0.06],
    )

    im = None
    for row, n in enumerate(counts):
        path = data_dir / f"circles-64-n{n}.pt"
        raw  = torch.load(path, weights_only=True)
        data = raw[split]                       # (N, 1, 64, 64)
        idx  = rng.choice(len(data), size=n_cols, replace=False)
        imgs = data[idx, 0].numpy()             # (n_cols, 64, 64)

        for col, img in enumerate(imgs):
            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(img, cmap=_CMAP, vmin=0, vmax=1,
                           interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if col == 0:
                ax.set_ylabel(f"n = {n}", fontsize=9, rotation=0,
                              labelpad=28, va="center")

    # colorbar — no tick labels, just the scale
    cax  = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_ticks([0.0, 1.0])
    cbar.set_ticklabels(["", ""])
    cbar.ax.tick_params(length=4)

    fig.suptitle(r"Circles dataset — $r \sim \mathrm{Unif}\{4,\ldots,10\}$",
                 fontsize=11, y=1.01)
    return fig


def main() -> None:
    pa = argparse.ArgumentParser()
    pa.add_argument("--data-dir", type=str, default="data")
    pa.add_argument("--cols",     type=int, default=8)
    pa.add_argument("--split",    type=str, default="train")
    pa.add_argument("--output",   type=str, default=None)
    pa.add_argument("--seed",     type=int, default=0)
    args = pa.parse_args()

    fig = plot_all_counts(
        data_dir=Path(args.data_dir),
        n_cols=args.cols,
        split=args.split,
        seed=args.seed,
    )

    out = Path(args.output) if args.output else Path("outputs") / "circles_preview_counts.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
