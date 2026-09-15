"""Unconditional flow-matching training on circles datasets (UNet velocity model)."""
import argparse
import json
import math
import re
from pathlib import Path
from datetime import datetime

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

BATCH       = 512
EPOCHS      = 100
LR          = 2e-4
WD          = 1e-4
EMA_DECAY   = 0.999
SAMPLE_STEPS = 100
ARCH        = {
    "ch": 44,
    "ch_mul": [1, 2, 2],
    "att_channels": [0, 0, 1],
    "groups": 4,
}


# ── dataset ───────────────────────────────────────────────────────────────────
def load_dataset(path: Path):
    return torch.load(path, weights_only=True)


def normalize_images(images: torch.Tensor) -> torch.Tensor:
    """Convert compact uint8 datasets to the [0, 1] float convention."""
    if images.dtype == torch.uint8:
        return images.float().div_(255.0)
    return images.float()


# ── loss ──────────────────────────────────────────────────────────────────────
def fm_loss(model, x1, device):
    B  = x1.shape[0]
    t  = torch.rand(B, device=device)
    x0 = torch.randn_like(x1)
    xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * x1
    return torch.mean((model(xt, t) - (x1 - x0)) ** 2)


# ── sampling ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def sample(model, n, device, im_size: int, method="euler", n_steps=100,
           initial_noise: torch.Tensor | None = None):
    """Generate n samples via fixed-step ODE integration (euler or rk4)."""
    if initial_noise is None:
        x0 = torch.randn(n, 1, im_size, im_size, device=device)
    else:
        expected = (n, 1, im_size, im_size)
        if tuple(initial_noise.shape) != expected:
            raise ValueError(
                f"initial_noise has shape {tuple(initial_noise.shape)}, expected {expected}"
            )
        x0 = initial_noise.to(device)
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
def save_eval_grid(model, epoch, device, eval_dir: Path, ckpt_dir: Path,
                   im_size: int, train_losses=None, val_losses=None,
                   out_dir: Path = None, sample_steps: int = SAMPLE_STEPS):
    model.eval()
    initial_noise = torch.randn(2, 1, im_size, im_size, device=device)
    methods = ("euler", "rk4")
    generated = {
        method: sample(
            model, 2, device, im_size=im_size, method=method,
            n_steps=sample_steps, initial_noise=initial_noise,
        )[:, 0].cpu().numpy()
        for method in methods
    }
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 5.2),
                             gridspec_kw={"hspace": 0.18, "wspace": 0.08})
    for row, method in enumerate(methods):
        for column, ax in enumerate(axes[row]):
            img = generated[method][column]
            ax.imshow(img, cmap=_CMAP, vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(
                f"{method.upper()} · {sample_steps} steps · sample {column + 1}",
                fontsize=8,
            )
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
    fig.suptitle(f"Epoch {epoch} unconditional samples", fontsize=10)
    eval_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(eval_dir / f"epoch_{epoch:04d}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch:04d}.pt")
    tqdm.write(f"  → eval + checkpoint saved (epoch {epoch})")

    if train_losses is not None and out_dir is not None:
        save_loss_plot(train_losses, val_losses, out_dir)


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
    fig.savefig(out_dir / "loss_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_loss_history(train_losses, val_losses, out_dir: Path) -> None:
    history = {
        "epoch": list(range(1, len(train_losses) + 1)),
        "train_loss": train_losses,
        "val_loss": val_losses,
    }
    with (out_dir / "loss_history.json").open("w") as f:
        json.dump(history, f, indent=2)


def recover_loss_history(log_path: Path, through_epoch: int):
    """Recover completed epoch losses from tqdm output in an interrupted log."""
    ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    text = ansi.sub("", log_path.read_text(errors="ignore")).replace("\r", "\n")
    by_epoch = {}
    pattern = re.compile(
        r"Training:.*?\|\s*(\d+)/\d+.*?train=([0-9.eE+-]+), val=([0-9.eE+-]+)"
    )
    for match in pattern.finditer(text):
        epoch = int(match.group(1)) + 1
        if epoch <= through_epoch:
            by_epoch[epoch] = (float(match.group(2)), float(match.group(3)))
    missing = [epoch for epoch in range(1, through_epoch + 1) if epoch not in by_epoch]
    if missing:
        raise RuntimeError(f"Could not recover loss history for epochs: {missing}")
    return (
        [by_epoch[epoch][0] for epoch in range(1, through_epoch + 1)],
        [by_epoch[epoch][1] for epoch in range(1, through_epoch + 1)],
    )


def save_config(
    out_dir: Path,
    data_path: Path,
    n_circles: int,
    im_size: int,
    epochs: int,
    batch: int,
    lr: float,
    wd: float,
    ema_decay: float,
    sample_steps: int,
    micro_batch: int,
    arch: dict,
    params: int,
    dataset_metadata: dict | None = None,
) -> None:
    config = {
        "im_size": im_size,
        "data": str(data_path),
        "epochs": epochs,
        "batch": batch,
        "lr": lr,
        "wd": wd,
        "ema_decay": ema_decay,
        "lr_schedule": {
            "name": "CosineAnnealingLR",
            "t_max": epochs,
            "eta_min": 1e-5,
        },
        "checkpoint_sampling": {
            "samples_per_integrator": 2,
            "integrators": ["euler", "rk4"],
            "steps": sample_steps,
            "shared_initial_noise": True,
        },
        "micro_batch": micro_batch,
        "arch": arch,
        "params": params,
        "dataset_metadata": dataset_metadata or {},
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    if n_circles >= 0:
        config["n_circles"] = n_circles
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2)


def prepare_run_dir(out_dir: Path) -> None:
    for subdir, pattern in (("checkpoints", "epoch_*.pt"), ("eval", "epoch_*.png")):
        for path in (out_dir / subdir).glob(pattern):
            path.unlink()
    for name in ("loss_curves.png", "loss_history.json", "run_summary.json", "config.json"):
        path = out_dir / name
        if path.exists():
            path.unlink()


def infer_n_circles(path: Path) -> int:
    match = re.search(r"-n(\d+)(?:$|-)", path.stem)
    if match is None:
        return -1
    return int(match.group(1))


def autocast_context(device: str):
    if device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


def default_micro_batch(im_size: int, batch: int) -> int:
    if im_size >= 256:
        return min(batch, 32)
    # Deepest-level attention cost grows with (im_size / 4)^2 tokens, so 200 and
    # 250 need smaller chunks than 160. These are the largest that fit in 48 GiB
    # (measured peaks: 38.3 GiB at 200/40 and 35.5 GiB at 250/16; the next step
    # up, 200/44 and 250/20, both OOM).
    if im_size >= 224:
        return min(batch, 16)
    if im_size >= 192:
        return min(batch, 40)
    if im_size >= 128:
        return min(batch, 128)
    return min(batch, 128)


def batch_fm_loss(model, x1, device, micro_batch: int, backward: bool = False):
    total = x1.shape[0]
    total_loss = 0.0
    for start in range(0, total, micro_batch):
        x1_micro = x1[start:start + micro_batch]
        weight = x1_micro.shape[0] / total
        loss = fm_loss(model, x1_micro, device)
        if backward:
            (loss * weight).backward()
        total_loss += loss.item() * weight
    return total_loss


def evaluate_split(model, images: torch.Tensor, device: str,
                   batch: int, micro_batch: int) -> float:
    model.eval()
    with torch.no_grad(), autocast_context(device):
        return sum(
            batch_fm_loss(
                model, normalize_images(images[i:i + batch]).to(device),
                device, micro_batch, backward=False,
            )
            for i in range(0, len(images), batch)
        ) / math.ceil(len(images) / batch)



# ── main ──────────────────────────────────────────────────────────────────────
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--data",    type=str, default=str(DATA_PATH),
                    help="Path to .pt dataset file")
    pa.add_argument("--out-dir", type=str, default=str(OUT_DIR),
                    help="Root output directory for this run")
    pa.add_argument("--epochs",  type=int, default=EPOCHS)
    pa.add_argument("--batch",   type=int, default=BATCH)
    pa.add_argument("--lr",      type=float, default=LR)
    pa.add_argument("--wd",      type=float, default=WD)
    pa.add_argument("--ema-decay", type=float, default=EMA_DECAY)
    pa.add_argument(
        "--sample-steps", "--euler-steps", dest="sample_steps",
        type=int, default=SAMPLE_STEPS,
        help="Fixed integration steps used by both checkpoint samplers.",
    )
    pa.add_argument("--num-workers", type=int, default=4)
    pa.add_argument("--micro-batch", type=int, default=None,
                    help="Forward/backward chunk size; defaults by image size")
    pa.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    pa.add_argument("--resume-checkpoint", type=str, default=None,
                    help="EMA checkpoint to resume from without deleting prior outputs")
    pa.add_argument("--start-epoch", type=int, default=0,
                    help="Last completed epoch represented by the resume checkpoint")
    pa.add_argument("--resume-log", type=str, default=None,
                    help="Interrupted log used to recover the earlier loss curve")
    pa.add_argument("--device",  type=str, default=None,
                    help="e.g. cuda:0 or cuda:1 (default: cuda if available)")
    args = pa.parse_args()

    data_path = Path(args.data)
    out_dir   = Path(args.out_dir)
    eval_dir  = out_dir / "eval"
    ckpt_dir  = out_dir / "checkpoints"
    resume_checkpoint = Path(args.resume_checkpoint) if args.resume_checkpoint else None
    start_epoch = args.start_epoch
    if resume_checkpoint is None:
        if start_epoch != 0 or args.resume_log is not None:
            pa.error("--start-epoch/--resume-log require --resume-checkpoint")
        prepare_run_dir(out_dir)
    else:
        if not resume_checkpoint.exists():
            pa.error(f"Resume checkpoint does not exist: {resume_checkpoint}")
        if not 0 < start_epoch < args.epochs:
            pa.error("--start-epoch must be between 1 and epochs-1 when resuming")
    epochs    = args.epochs
    batch     = args.batch
    lr        = args.lr
    wd        = args.wd
    ema_decay = args.ema_decay
    sample_steps = args.sample_steps
    num_workers = args.num_workers
    eval_every = max(1, epochs // 10)

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  data: {data_path}  |  out: {out_dir}")

    splits     = load_dataset(data_path)
    train_data = splits["train"]
    val_data   = splits["val"]
    im_size    = train_data.shape[-1]
    assert train_data.shape[-2] == im_size, "Only square images are supported"
    micro_batch = args.micro_batch or default_micro_batch(im_size, batch)
    micro_batch = max(1, min(micro_batch, batch))
    print(f"Split sizes — train: {len(train_data)}  "
          f"val: {len(splits['val'])}  test: {len(splits['test'])}")
    print(f"Effective batch: {batch}  |  micro-batch: {micro_batch}")

    loader = data.DataLoader(
        data.TensorDataset(train_data),
        batch_size=batch,
        shuffle=True,
        pin_memory=device.startswith("cuda"),
        num_workers=num_workers,
    )

    if device.startswith("cuda"):
        torch.backends.cuda.enable_flash_sdp(True)
    model = Unet(ch=ARCH["ch"], ch_mul=ARCH["ch_mul"],
                 att_channels=ARCH["att_channels"],
                 groups=ARCH["groups"], dropout=0.0).to(device)
    resume_state = None
    if resume_checkpoint is not None:
        resume_state = torch.load(resume_checkpoint, map_location=device, weights_only=True)
        model_state = {
            key.removeprefix("module._orig_mod.").removeprefix("module."): value
            for key, value in resume_state.items()
            if key != "n_averaged"
        }
        model.load_state_dict(model_state)
        print(f"Resuming model and EMA weights from {resume_checkpoint} at epoch {start_epoch}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    if resume_checkpoint is None:
        save_config(
            out_dir=out_dir,
            data_path=data_path,
            n_circles=infer_n_circles(data_path),
            im_size=im_size,
            epochs=epochs,
            batch=batch,
            lr=lr,
            wd=wd,
            ema_decay=ema_decay,
            sample_steps=sample_steps,
            micro_batch=micro_batch,
            arch=ARCH,
            params=n_params,
            dataset_metadata=splits.get("metadata", {}),
        )
    else:
        config_path = out_dir / "config.json"
        with config_path.open() as f:
            config = json.load(f)
        config.setdefault("initial_micro_batch", config.get("micro_batch"))
        config["micro_batch"] = micro_batch
        config.setdefault("resumes", []).append({
            "checkpoint": str(resume_checkpoint),
            "start_epoch": start_epoch,
            "resumed_at": datetime.now().isoformat(timespec="seconds"),
            "micro_batch": micro_batch,
            "device": device,
            "optimizer_state": "reinitialized; not present in legacy checkpoint",
        })
        with config_path.open("w") as f:
            json.dump(config, f, indent=2)

    if args.compile:
        print("Compiling model (first epoch will be slow) …")
        model = torch.compile(model)

    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))
    if resume_state is not None:
        ema_model.load_state_dict(resume_state)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )
    if start_epoch:
        resumed_lr = 1e-5 + (lr - 1e-5) * (1 + math.cos(math.pi * start_epoch / epochs)) / 2
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = resumed_lr
        scheduler.last_epoch = start_epoch
        scheduler._last_lr = [resumed_lr for _ in optimizer.param_groups]
        print(f"Reconstructed cosine schedule at epoch {start_epoch}: lr={resumed_lr:.6g}")

    if start_epoch and args.resume_log:
        train_losses, val_losses = recover_loss_history(Path(args.resume_log), start_epoch)
    elif start_epoch:
        history_path = out_dir / "loss_history.json"
        if not history_path.exists():
            raise FileNotFoundError("Resume requires --resume-log or an existing loss_history.json")
        with history_path.open() as f:
            prior_history = json.load(f)
        train_losses = prior_history["train_loss"][:start_epoch]
        val_losses = prior_history["val_loss"][:start_epoch]
    else:
        train_losses, val_losses = [], []

    epoch_bar = tqdm(range(start_epoch + 1, epochs + 1), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        batch_losses = []
        for (x1,) in tqdm(loader, desc=f"  Epoch {epoch:4d}", leave=False, unit="batch"):
            x1 = normalize_images(x1).to(device)
            optimizer.zero_grad()
            with autocast_context(device):
                loss = batch_fm_loss(model, x1, device, micro_batch, backward=True)
            optimizer.step()
            ema_model.update_parameters(model)
            batch_losses.append(loss)
        scheduler.step()

        train_loss = sum(batch_losses) / len(batch_losses)

        val_loss = evaluate_split(
            ema_model, val_data, device, batch, micro_batch,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        save_loss_history(train_losses, val_losses, out_dir)
        epoch_bar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}")

        if epoch % eval_every == 0:
            save_eval_grid(ema_model, epoch, device, eval_dir, ckpt_dir,
                           im_size, train_losses, val_losses, out_dir,
                           sample_steps)

    save_loss_plot(train_losses, val_losses, out_dir)
    test_loss = evaluate_split(
        ema_model, splits["test"], device, batch, micro_batch,
    )
    summary = {
        "status": "complete",
        "epochs_completed": epochs,
        "final_train_loss": train_losses[-1],
        "final_validation_loss": val_losses[-1],
        "final_test_loss": test_loss,
        "checkpoint_count": len(list(ckpt_dir.glob("epoch_*.pt"))),
        "evaluation_grid_count": len(list(eval_dir.glob("epoch_*.png"))),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    with (out_dir / "run_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Final test FM loss: {test_loss:.6f}")
    print(f"Done. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
