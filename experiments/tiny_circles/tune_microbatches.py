"""Calibrate the 22k FM model's microbatches at a fixed effective batch of 512."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import time
from unittest.mock import patch

import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from models.unet.default import Unet
from train_fm_unet import autocast_context, batch_fm_loss, normalize_images


def gpu_processes(uuid: str) -> list[int]:
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader,nounits'], text=True)
    return [int(pid) for row in rows.splitlines() for gpu, pid in [row.split(', ')]
            if gpu == uuid and int(pid) != os.getpid()]


def verify_accumulation(device: str) -> list[dict]:
    """Hold noise/time draws fixed; changing chunking otherwise changes RNG draws."""
    model = Unet(ch=4, ch_mul=[1, 2, 2], att_channels=[0, 0, 0], groups=2,
                 mid_attn=False, dropout=0.).to(device, memory_format=torch.channels_last)
    images = torch.rand(17, 1, 64, 64, device=device)
    noise = torch.randn_like(images)
    times = torch.rand(17, device=device)
    records = []
    for precision in ('fp32', 'bf16'):
        gradients = []
        for micro in (17, 8, 3):
            offset = 0

            def fixed_loss(m: torch.nn.Module, x: torch.Tensor, device: str,
                           t_alpha: float = 1.) -> torch.Tensor:
                nonlocal offset
                t = times[offset:offset + len(x)]
                z = noise[offset:offset + len(x)]
                offset += len(x)
                xt = (1 - t[:, None, None, None]) * z + t[:, None, None, None] * x
                return (m(xt, t) - (x - z)).square().mean()

            model.zero_grad(set_to_none=True)
            with patch('train_fm_unet.fm_loss', fixed_loss):
                with autocast_context(device) if precision == 'bf16' else nullcontext():
                    loss = batch_fm_loss(model, images, device, micro, backward=True)
            gradients.append(torch.cat([p.grad.flatten() for p in model.parameters()]))
            relative = float((gradients[-1] - gradients[0]).norm() / gradients[0].norm())
            assert relative < (0.02 if precision == 'bf16' else 1e-4), relative
            assert offset == len(images) and torch.isfinite(loss)
            records.append(dict(precision=precision, micro_batch=micro, relative_gradient_error=relative))
    return records


def trial(images: torch.Tensor, micro: int, device: str) -> dict:
    torch.manual_seed(912)
    model = Unet(ch=4, ch_mul=[1, 2, 2], att_channels=[0, 0, 0], groups=2,
                 mid_attn=False, dropout=0.).to(device, memory_format=torch.channels_last)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(.999))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4, fused=True)
    indices = torch.arange(512, device=device)

    def update() -> torch.Tensor:
        x = normalize_images(images[indices]).contiguous(memory_format=torch.channels_last)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device):
            loss = batch_fm_loss(model, x, device, micro, backward=True)
        optimizer.step()
        ema.update_parameters(model)
        return loss

    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(2):
        update()
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    losses = [update() for _ in range(4)]
    value = float(torch.stack(losses).mean())
    torch.cuda.synchronize(device)
    milliseconds = 1000 * (time.perf_counter() - start) / len(losses)
    assert torch.isfinite(torch.tensor(value))
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    return dict(ms=milliseconds, peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--out', type=Path, default=Path('outputs/microbatch_tuning_20261019'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_device(args.device)
    torch.cuda.set_per_process_memory_fraction(.85, args.device)
    reservation = torch.zeros(1, device=args.device)
    index = args.device.split(':')[-1]
    uuid = subprocess.check_output(['nvidia-smi', f'--id={index}', '--query-gpu=uuid',
                                   '--format=csv,noheader'], text=True).strip()
    print('Waiting for existing GPU work to finish; reservation prevents queue handoff.', flush=True)
    while gpu_processes(uuid):
        time.sleep(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(881)
    correctness = verify_accumulation(args.device)
    print('Gradient accumulation verified', correctness, flush=True)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    args.out.mkdir(parents=True, exist_ok=True)
    result = dict(status='running', gpu=torch.cuda.get_device_name(args.device), torch=torch.__version__,
                  memory_limit_fraction=.85, effective_batch=512, correctness=correctness,
                  trainer_sha256=hashlib.sha256(Path('train_fm_unet.py').read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  settings=dict(parameters=22037, ch_mul=[1,2,2], precision='bf16', tf32=False,
                                channels_last=True, fused_adam=True, gpu_resident=True, threads=4),
                  rows=[], selections={})
    for resolution in (64, 96, 128, 160, 200, 250):
        data_path = Path(f'data/circles-{resolution}-n1.pt')
        raw = torch.load(data_path, mmap=True, map_location='cpu', weights_only=True)
        splits = {key: raw[key].to(args.device) for key in ('train', 'val', 'test')}
        assert [len(splits[key]) for key in ('train','val','test')] == [48000,1000,1000]
        assert splits['train'].dtype == torch.uint8
        del raw
        candidates = [16, 32, 64, 128, 256, 384, 512]
        if resolution == 200:
            candidates.append(40)
        rejected: set[int] = set()
        for repeat in range(3):
            order = list(candidates)
            random.Random(910 + repeat).shuffle(order)
            for micro in order:
                if micro in rejected:
                    continue
                while gpu_processes(uuid):
                    time.sleep(1)
                gc.collect()
                torch.cuda.empty_cache()
                row = dict(resolution=resolution, micro_batch=micro, repeat=repeat)
                try:
                    row.update(trial(splits['train'], micro, args.device), status='ok')
                except torch.cuda.OutOfMemoryError:
                    row.update(status='oom')
                    rejected.add(micro)
                if gpu_processes(uuid):
                    raise RuntimeError('Another process started during measurement; results are incomplete')
                result['rows'].append(row)
                (args.out/'results.json').write_text(json.dumps(result, indent=2))
                print(json.dumps(row), flush=True)
        summaries = []
        for micro in candidates:
            rows = [r for r in result['rows'] if r['resolution']==resolution and r['micro_batch']==micro and r['status']=='ok']
            if len(rows) == 3:
                summaries.append(dict(micro_batch=micro, median_ms=statistics.median(r['ms'] for r in rows),
                                      peak_reserved_gib=max(r['peak_reserved_gib'] for r in rows),
                                      peak_allocated_gib=max(r['peak_allocated_gib'] for r in rows)))
        fastest = min(r['median_ms'] for r in summaries)
        selected = min((r for r in summaries if r['median_ms'] <= 1.02*fastest),
                       key=lambda r:r['peak_reserved_gib'])
        result['selections'][resolution] = dict(selected=selected, candidates=summaries,
            rule='Lowest peak reserved memory among candidates within 2% of the fastest median')
        print('Selected',resolution,selected,flush=True)
        del splits
    result['status']='complete'
    (args.out/'results.json').write_text(json.dumps(result, indent=2))
    del reservation


if __name__ == '__main__':
    main()
