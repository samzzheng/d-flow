"""Short CPU FP32 timing projection for the 117k separated-shapes model."""

from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.unet.default import Unet
from train_fm_unet import batch_fm_loss


def main() -> None:
    torch.set_num_interop_threads(1)
    output = ROOT / "outputs/shapes_cpu_timing_20261021"
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "device": "cpu", "dtype": "float32", "effective_batch": 512,
        "micro_batch": 32, "epochs": 500, "training_images": 80000,
        "affinity": sorted(os.sched_getaffinity(0)), "rows": [],
        "scope": "Synthetic in-memory batches; real FM forward/backward, AdamW and EMA. Project microbatch timing to 80k images for 500 epochs. Excludes validation, sampling, I/O and sustained-load effects; no CPU tuning or BF16.",
    }
    for threads in (4, 16):
        torch.set_num_threads(threads)
        for resolution in (64, 96, 128, 160, 200, 250):
            torch.manual_seed(20261021)
            model = Unet(ch=6, ch_mul=[1, 2, 3, 3], att_channels=[0]*4,
                         groups=2, mid_attn=False, time_embed_dim=32, dropout=0.)
            model.to(memory_format=torch.channels_last)
            ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(.999))
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
            images = torch.rand(32, 1, resolution, resolution).contiguous(memory_format=torch.channels_last)
            forward_backward, updates = [], []
            for repeat in range(4):
                start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                loss = batch_fm_loss(model, images, "cpu", 32, backward=True)
                middle = time.perf_counter()
                optimizer.step()
                ema.update_parameters(model)
                end = time.perf_counter()
                if repeat:
                    forward_backward.append(middle - start)
                    updates.append(end - middle)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite CPU training loss")
            fb = statistics.median(forward_backward)
            update = statistics.median(updates)
            row = dict(threads=threads, resolution=resolution,
                       parameters=sum(p.numel() for p in model.parameters()),
                       microbatch_seconds=fb, optimizer_ema_seconds=update,
                       microbatch_repeats=forward_backward,
                       training_hours=(2500 * fb + 157 * update) * 500 / 3600)
            report["rows"].append(row)
            (output / "timings.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(row), flush=True)
            del model, ema, optimizer, images


if __name__ == "__main__":
    main()
