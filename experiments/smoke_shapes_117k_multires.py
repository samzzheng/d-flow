"""Check the scheduled 117k shape models with real data and graph training."""

from __future__ import annotations

import gc
import argparse
import json
from pathlib import Path
import sys
import time

import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dlr_dflow.implementation.model_io import load_flow_model
from models.unet.default import Unet
from train_fm_unet import fm_loss, normalize_images
from training.fm_acceleration import GraphedMicrobatches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, required=True, choices=(96, 128, 160, 200, 250))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    output = ROOT / "outputs/separated_shapes_117k_multires_20261021"
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for resolution, gpu, micro in ((250, 1, 32), (200, 0, 32), (160, 0, 64), (128, 1, 128), (96, 0, 128)):
        if resolution != args.resolution:
            continue
        device = f"cuda:{gpu}"
        torch.cuda.set_device(device)
        torch.manual_seed(20261015)
        data = torch.load(ROOT / f"data/separated-shapes-{resolution}.pt", weights_only=True,
                          map_location="cpu", mmap=True)
        resident = {key: data[key].to(device) for key in ("train", "val", "test")}
        model = Unet(ch=6, ch_mul=[1, 2, 3, 3], att_channels=[0]*4, groups=2,
                     mid_attn=False, time_embed_dim=32, dropout=0.).to(device, memory_format=torch.channels_last)
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(.999))
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4, fused=True)
        graph = GraphedMicrobatches(model, fm_loss, device)
        torch.cuda.reset_peak_memory_stats(device)
        durations = []
        for batch_size in (512, 512, 128):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            images = normalize_images(resident["train"][:batch_size]).contiguous(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=False)
            loss = graph(images, micro)
            optimizer.step()
            ema.update_parameters(model)
            torch.cuda.synchronize(device)
            durations.append(time.perf_counter() - start)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss at {resolution}")
        training_peak = torch.cuda.max_memory_reserved(device) / 2**30
        reference, _ = load_flow_model(resolution=resolution, epoch=200, device=device,
                                      run_dir=str(ROOT / f"outputs/fm_unet/separated-shapes-{resolution}-noattn-ch16-b8-e200"))
        with torch.no_grad():
            prediction = reference(torch.randn(micro, 1, resolution, resolution, device=device),
                                   torch.full((micro,), .5, device=device))
        if not torch.isfinite(prediction).all() or prediction.shape != (micro, 1, resolution, resolution):
            raise RuntimeError(f"Invalid reference evaluation at {resolution}")
        results.append(dict(resolution=resolution, device=device, micro_batch=micro,
                            parameters=sum(p.numel() for p in model.parameters()),
                            update_seconds=durations, training_peak_reserved_gib=training_peak,
                            combined_reference_peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30,
                            finite_training_and_reference=True))
        (output / f"smoke_{resolution}.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results[-1]), flush=True)
        del model, ema, optimizer, graph, images, resident, data, loss, reference, prediction
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
