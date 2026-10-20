"""Checkpoint loading for pretrained flow-matching models."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from models.unet.default import Unet

ROOT = Path(__file__).resolve().parents[2]


def load_flow_model(resolution: int = 64, n_circles: int = 1, epoch: int = 50,
                    device: str = "cpu", run_dir: str | Path | None = None
                    ) -> tuple[torch.nn.Module, int]:
    """Load the pretrained flow-matching UNet (smallest 64x64 n1 model by default).

    run_dir overrides the conventional outputs/fm_unet/circles-<res>/n<k> path,
    for runs that do not follow it (e.g. the attention-free variants).
    """
    run = Path(run_dir) if run_dir is not None else (
        ROOT / f"outputs/fm_unet/circles-{resolution}/n{n_circles}")
    arch = json.loads((run / "config.json").read_text())["arch"]
    # mid_attn is absent from configs written before the attention-free runs;
    # its default (True) is what those checkpoints were trained with.
    model = Unet(ch=arch["ch"], ch_mul=arch["ch_mul"], att_channels=arch["att_channels"],
                 groups=arch["groups"], mid_attn=arch.get("mid_attn", True),
                 time_embed_dim=arch.get("time_embed_dim"),
                 dropout=0.0).to(device)
    state = torch.load(run / f"checkpoints/epoch_{epoch:04d}.pt",
                       map_location=device, weights_only=True)
    state = {k.removeprefix("module._orig_mod.").removeprefix("module."): v
             for k, v in state.items() if k != "n_averaged"}
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, resolution
