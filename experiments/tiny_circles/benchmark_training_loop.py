"""Check deferred loss logging against the original loop, then time paired rounds."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import statistics
import time

import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from models.unet.default import Unet
from train_fm_unet import autocast_context, batch_fm_loss, fm_loss


def original_loss(model: torch.nn.Module, images: torch.Tensor, device: str,
                  micro_batch: int, backward: bool = True) -> float:
    total = 0.0
    for start in range(0, len(images), micro_batch):
        chunk = images[start:start + micro_batch]
        weight = len(chunk) / len(images)
        loss = fm_loss(model, chunk, device)
        if backward:
            (loss * weight).backward()
        total += loss.item() * weight
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--out', type=Path, default=Path('outputs/training_loop_optimization_20261019'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    cuda = args.device.startswith('cuda')
    if cuda:
        torch.cuda.set_device(args.device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(761)
    model = Unet(ch=4, ch_mul=[1, 2, 2], att_channels=[0, 0, 0],
                 groups=2, mid_attn=False, dropout=0.).to(args.device, memory_format=torch.channels_last)
    models = [model, deepcopy(model)]
    optimizers = [torch.optim.AdamW(m.parameters(), lr=2e-4, fused=cuda) for m in models]
    emas = [AveragedModel(m, multi_avg_fn=get_ema_multi_avg_fn(.999)) for m in models]
    loss_functions = (original_loss, batch_fm_loss)
    max_loss_difference = 0.0
    for update, count in enumerate((8, 7, 8, 3)):
        images = torch.rand(count, 1, 64, 64, device=args.device)
        losses = []
        for m, opt, ema, loss_fn in zip(models, optimizers, emas, loss_functions):
            torch.manual_seed(1100 + update)
            opt.zero_grad(set_to_none=True)
            with autocast_context(args.device):
                loss = loss_fn(m, images, args.device, 3, backward=True)
            losses.append(float(loss))
            opt.step()
            ema.update_parameters(m)
        max_loss_difference = max(max_loss_difference, abs(losses[0] - losses[1]))
        for left, right in zip(models[0].parameters(), models[1].parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        for left, right in zip(emas[0].state_dict().values(), emas[1].state_dict().values()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        for left, right in zip(optimizers[0].state.values(), optimizers[1].state.values()):
            for key in left:
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
    assert max_loss_difference < 1e-12
    with torch.no_grad(), autocast_context(args.device):
        torch.manual_seed(912)
        old_val = original_loss(models[0], images, args.device, 2, backward=False)
        torch.manual_seed(912)
        new_val = float(batch_fm_loss(models[1], images, args.device, 2))
    assert abs(old_val - new_val) < 1e-12
    results = dict(correctness='passed', updates=4, max_loss_difference=max_loss_difference,
                   parameters_ema_adam='bitwise equal', timings=[],
                   timing_note='Short alternating rounds; existing GPU training is left running. Not an isolated whole-epoch throughput claim.')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'verification.json').write_text(json.dumps(results, indent=2))
    print('Correctness passed', flush=True)
    if cuda:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        for resolution, batch, micro in ((64, 512, 512), (64, 512, 128), (250, 64, 16)):
            images = torch.rand(batch, 1, resolution, resolution, device=args.device).contiguous(memory_format=torch.channels_last)
            times: list[list[float]] = [[], []]
            for round_index in range(4):
                for variant in ((0, 1) if round_index % 2 == 0 else (1, 0)):
                    m, opt, ema = models[variant], optimizers[variant], emas[variant]
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    losses = []
                    for _ in range(12):
                        opt.zero_grad(set_to_none=True)
                        with autocast_context(args.device):
                            loss = loss_functions[variant](m, images, args.device, micro, backward=True)
                        opt.step()
                        ema.update_parameters(m)
                        losses.append(loss)
                    if variant:
                        float(torch.stack(losses).mean())
                    torch.cuda.synchronize()
                    elapsed = 1000 * (time.perf_counter() - start) / 12
                    if round_index:
                        times[variant].append(elapsed)
            row = dict(resolution=resolution, batch=batch, micro_batch=micro,
                       old_ms=statistics.median(times[0]), new_ms=statistics.median(times[1]),
                       old_rounds_ms=times[0], new_rounds_ms=times[1])
            results['timings'].append(row)
            print(json.dumps(row), flush=True)
        (args.out/'verification.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
