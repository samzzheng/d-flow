"""Unconditional flow-matching training on 64×64 circles (UNet velocity model)."""
import argparse
from pathlib import Path

import cmocean
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from tqdm import tqdm

from models.unet.default import Unet
from sampler.ode_samplers import ode_integrate
import dataset.circles_64 as circles_64

# ── aesthetics ────────────────────────────────────────────────────────────────
matplotlib.rcParams["font.family"] = "TeX Gyre Pagella"
matplotlib.rcParams["mathtext.fontset"] = "custom"
matplotlib.rcParams["mathtext.rm"] = "TeX Gyre Pagella"
_CMAP = cmocean.cm.dense_r

# ── constants (overridable via CLI) ──────────────────────────────────────────
DATA_PATH  = Path("data/circles-64-uniform.pt")
OUT_DIR    = Path("outputs/fm_unet")

BATCH       = 128
EPOCHS      = 150
LR          = 2e-4
WD          = 1e-4
EMA_DECAY   = 0.999
EULER_STEPS = 200

# sampler configs used in the post-training comparison
SAMPLER_CONFIGS = [
    ("Euler  200 steps", "euler", 200),
    ("Euler 1000 steps", "euler", 1000),
    ("RK4   100 steps",  "rk4",   100),
]


# ── dataset ───────────────────────────────────────────────────────────────────
def load_dataset(path: Path):
    return torch.load(path, weights_only=True)


# ── loss ──────────────────────────────────────────────────────────────────────
def fm_loss(model, x1, device):
    B  = x1.shape[0]
    t  = torch.rand(B, device=device)
    x0 = torch.randn_like(x1)
    xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * x1
    return torch.mean((model(xt, t) - (x1 - x0)) ** 2)


# ── sampling ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def sample(model, n, device, method="euler", n_steps=200):
    """Generate n samples via fixed-step ODE integration (euler or rk4)."""
    x0 = torch.randn(n, 1, 64, 64, device=device)
    step_size = 1.0 / n_steps

    def ode_func(t, x):
        # ode_integrate passes scalar t; UNet expects batched (B,)
        t_batch = t.expand(x.shape[0])
        return model(x, t_batch)

    x1 = ode_integrate(
        ode_func=ode_func,
        init_x=x0,
        ode_opts={"options": {"step_size": step_size}},
        sampler=method,
    )
    return x1.clamp(0.0, 1.0)


# ── eval grid + checkpoint ────────────────────────────────────────────────────
def save_eval_grid(model, epoch, device, eval_dir: Path, ckpt_dir: Path):
    model.eval()
    imgs = sample(model, 25, device, method="rk4", n_steps=100)[:, 0].cpu().numpy()
    fig, axes = plt.subplots(5, 5, figsize=(5.5, 5.5),
                             gridspec_kw={"hspace": 0.04, "wspace": 0.04})
    for ax, img in zip(axes.flat, imgs):
        ax.imshow(img, cmap=_CMAP, vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle(f"Epoch {epoch}", fontsize=10)
    eval_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(eval_dir / f"epoch_{epoch:04d}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch:04d}.pt")
    tqdm.write(f"  → eval + checkpoint saved (epoch {epoch})")


# ── sampler comparison ────────────────────────────────────────────────────────
def save_sampler_comparison(model, device, eval_dir: Path):
    """3-column grid: Euler-200 | Euler-1000 | RK4-100, 5 rows each."""
    n = 5
    cols = len(SAMPLER_CONFIGS)
    fig, axes = plt.subplots(n, cols, figsize=(cols * 2.5 + 0.3, n * 2.5 + 0.5),
                             gridspec_kw={"hspace": 0.04, "wspace": 0.04})

    model.eval()
    torch.manual_seed(0)
    x0_fixed = torch.randn(n, 1, 64, 64, device=device)

    for col, (label, method, n_steps) in enumerate(tqdm(SAMPLER_CONFIGS, desc="Sampler comparison")):
        step_size = 1.0 / n_steps

        def ode_func(t, x, _model=model):
            return _model(x, t.expand(x.shape[0]))

        with torch.no_grad():
            x1 = ode_integrate(
                ode_func=ode_func,
                init_x=x0_fixed.clone(),
                ode_opts={"options": {"step_size": step_size}},
                sampler=method,
            ).clamp(0.0, 1.0)

        imgs = x1[:, 0].cpu().numpy()
        axes[0, col].set_title(label, fontsize=9)
        for row, img in enumerate(imgs):
            ax = axes[row, col]
            ax.imshow(img, cmap=_CMAP, vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)

    fig.suptitle("Sampler comparison (same initial noise)", fontsize=11, y=1.01)
    eval_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(eval_dir / "sampler_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved sampler comparison")


# ── loss plot ─────────────────────────────────────────────────────────────────
def save_loss_plot(train_losses, val_losses, out_dir: Path):
    c_train = _CMAP(0.15)
    c_val   = _CMAP(0.75)
    epochs  = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for ax, scale in [(ax1, "linear"), (ax2, "log")]:
        ax.plot(epochs, train_losses, color=c_train, label="train")
        ax.plot(epochs, val_losses,   color=c_val,   label="val")
        ax.set_yscale(scale)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("FM Loss")
        ax.set_title(f"Loss ({scale} scale)")
        ax.legend()
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)




# ── main ──────────────────────────────────────────────────────────────────────
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--data",    type=str, default=str(DATA_PATH),
                    help="Path to .pt dataset file")
    pa.add_argument("--out-dir", type=str, default=str(OUT_DIR),
                    help="Root output directory for this run")
    pa.add_argument("--epochs",  type=int, default=EPOCHS)
    pa.add_argument("--batch",   type=int, default=BATCH)
    pa.add_argument("--device",  type=str, default=None,
                    help="e.g. cuda:0 or cuda:1 (default: cuda if available)")
    args = pa.parse_args()

    data_path = Path(args.data)
    out_dir   = Path(args.out_dir)
    eval_dir  = out_dir / "eval"
    ckpt_dir  = out_dir / "checkpoints"
    epochs    = args.epochs
    batch     = args.batch
    eval_every = max(1, epochs // 10)

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  data: {data_path}  |  out: {out_dir}")

    splits     = load_dataset(data_path)
    train_data = splits["train"]
    val_data   = splits["val"].to(device)
    print(f"Split sizes — train: {len(train_data)}  "
          f"val: {len(splits['val'])}  test: {len(splits['test'])}")

    loader = data.DataLoader(
        data.TensorDataset(train_data),
        batch_size=batch,
        shuffle=True,
        pin_memory=device.startswith("cuda"),
        num_workers=4,
    )

    torch.backends.cuda.enable_flash_sdp(True)
    model = Unet(ch=24, ch_mul=[1, 2, 2], att_channels=[0, 1, 1],
                 groups=8, dropout=0.0).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    print("Compiling model (first epoch will be slow) …")
    model = torch.compile(model)

    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    train_losses, val_losses = [], []

    epoch_bar = tqdm(range(1, epochs + 1), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        batch_losses = []
        for (x1,) in tqdm(loader, desc=f"  Epoch {epoch:4d}", leave=False, unit="batch"):
            x1 = x1.to(device)
            optimizer.zero_grad()
            loss = fm_loss(model, x1, device)
            loss.backward()
            optimizer.step()
            ema_model.update_parameters(model)
            batch_losses.append(loss.item())
        scheduler.step()

        train_loss = sum(batch_losses) / len(batch_losses)

        ema_model.eval()
        with torch.no_grad():
            val_loss = fm_loss(ema_model, val_data, device).item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epoch_bar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}")

        if epoch % eval_every == 0:
            save_eval_grid(ema_model, epoch, device, eval_dir, ckpt_dir)

    save_sampler_comparison(ema_model, device, eval_dir)
    save_loss_plot(train_losses, val_losses, out_dir)
    print(f"Done. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
