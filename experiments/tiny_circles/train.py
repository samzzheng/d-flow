"""GPU-resident small circle FM pilot, with a replayable full training step."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time
from typing import Any

import torch
from torch import Tensor, nn

from models.unet.default import Unet


class TrainingStep:
    """Static full batches use CUDA graphs; the last partial batch stays eager."""

    def __init__(self, model: nn.Module, batch: int, lr: float, graphed: bool) -> None:
        self.model = model
        self.ema = deepcopy(model).eval().requires_grad_(False)
        self.lr = torch.tensor(lr, device='cuda:1')
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr,
                                           weight_decay=1e-4, fused=True, capturable=True)
        self.x1 = torch.empty(batch, 1, 64, 64, device='cuda:1', memory_format=torch.channels_last)
        self.x0 = torch.empty_like(self.x1)
        self.t = torch.empty(batch, device='cuda:1')
        self.total = torch.zeros((), device='cuda:1')
        self.updates = torch.zeros((), device='cuda:1')
        self.graph: torch.cuda.CUDAGraph | None = None
        if graphed:
            # Populate lazy Adam state on a side stream, then restore the initial state.
            initial = deepcopy(model.state_dict())
            self.x1.zero_()
            self.x0.normal_()
            self.t.uniform_()
            stream = torch.cuda.Stream(device=1)
            stream.wait_stream(torch.cuda.current_stream(1))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.eager(self.x1, self.x0, self.t)
            torch.cuda.current_stream(1).wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.eager(self.x1, self.x0, self.t)
            model.load_state_dict(initial)
            self.ema.load_state_dict(initial)
            for state in self.optimizer.state.values():
                for value in state.values():
                    if isinstance(value, Tensor):
                        value.zero_()
            self.total.zero_()
            self.updates.zero_()

    def eager(self, x1: Tensor, x0: Tensor, t: Tensor) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            time_grid = t[:, None, None, None]
            xt = (1 - time_grid) * x0 + time_grid * x1
            loss = (self.model(xt, t) - (x1 - x0)).square().mean()
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            decay = torch.where(self.updates == 0, 0.0, 0.999)
            torch._foreach_mul_(list(self.ema.parameters()), decay)
            torch._foreach_add_(list(self.ema.parameters()),
                                torch._foreach_mul(list(self.model.parameters()), 1 - decay))
            self.total.add_(loss.detach() * len(x1))
            self.updates.add_(1)

    def __call__(self, images: Tensor) -> None:
        if self.graph is not None and len(images) == len(self.x1):
            self.x1.copy_(images)
            self.x0.normal_()
            self.t.uniform_()
            self.graph.replay()
        else:
            self.eager(images, torch.randn_like(images), torch.rand(len(images), device=images.device))

    def restore(self, state: dict[str, Any]) -> None:
        self.model.load_state_dict(state['model'])
        self.ema.load_state_dict(state['ema'])
        if self.graph is None:
            self.optimizer.load_state_dict(state['optimizer'])
            self.optimizer.param_groups[0]['lr'] = self.lr
        else:
            # CUDA graph replay requires the original Adam buffer addresses.
            saved = state['optimizer']
            for parameter, index in zip(self.optimizer.param_groups[0]['params'], saved['param_groups'][0]['params']):
                for key, value in saved['state'][index].items():
                    self.optimizer.state[parameter][key].copy_(value)
        self.updates.copy_(state['updates'])


@torch.no_grad()
def evaluate(model: nn.Module, images: Tensor, batch: int, seed: int) -> float:
    """Identical fixed noise/time draws for each model and evaluation epoch; FP32."""
    model.eval()
    generator = torch.Generator(device=images.device).manual_seed(seed)
    total = torch.zeros((), device=images.device)
    for start in range(0, len(images), batch):
        x1 = images[start:start + batch].float().div(255).contiguous(memory_format=torch.channels_last)
        x0 = torch.randn(x1.shape, device=images.device, generator=generator).contiguous(memory_format=torch.channels_last)
        t = torch.rand(len(x1), device=images.device, generator=generator)
        xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * x1
        total.add_((model(xt, t) - (x1 - x0)).square().mean() * len(x1))
    return float(total / len(images))


@torch.no_grad()
def samples(model: nn.Module, out: Path, name: str, steps: int = 100) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    model.eval()
    generator = torch.Generator(device='cuda:1').manual_seed(4271)
    x = torch.randn(16, 1, 64, 64, device='cuda:1', generator=generator).contiguous(memory_format=torch.channels_last)
    for i in range(steps):
        x.add_(model(x, torch.full((len(x),), i / steps, device=x.device)), alpha=1 / steps)
    raw = x.cpu()
    torch.save(raw, out / f'{name}_{steps}steps.pt')
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for ax, img in zip(axes.flat, raw[:, 0]):
        ax.imshow(img.numpy(), vmin=0, vmax=1, cmap='gray')
        ax.axis('off')
    fig.suptitle(f'{name}: {steps}-step Euler (display [0,1]; raw arrays saved)')
    fig.tight_layout()
    fig.savefig(out / f'{name}_{steps}steps.png', dpi=130)
    plt.close(fig)


def atomic_save(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--seed', type=int, default=20261019)
    parser.add_argument('--graph', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_device(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'latest.pt').exists() and not args.resume:
        raise RuntimeError('Existing training state; use --resume or a new output directory.')
    arch = dict(ch=4, ch_mul=[1, 2, 2], att_channels=[0, 0, 0], groups=2, mid_attn=False)
    model = Unet(**arch, dropout=0.).to('cuda:1', memory_format=torch.channels_last)
    initial_rng = torch.cuda.get_rng_state(1)
    started = time.monotonic()
    step = TrainingStep(model, args.batch, args.lr, args.graph)
    torch.cuda.set_rng_state(initial_rng, 1)
    splits = torch.load('data/circles-64-n1.pt', mmap=True, weights_only=True)
    images = {k: splits[k].to('cuda:1') for k in ('train', 'val', 'test')}
    config = dict(arch=arch, params=sum(p.numel() for p in model.parameters()), im_size=64,
                  n_circles=1, data='data/circles-64-n1.pt', epochs=args.epochs,
                  batch=args.batch, lr=args.lr, wd=1e-4, ema_decay=.999,
                  precision='BF16 autocast training; FP32 fixed-draw evaluation; TF32 off',
                  schedule='cosine per epoch to 1e-5', gpu=1, graph=args.graph, seed=args.seed,
                  split_sizes={k: len(v) for k,v in images.items()},
                  source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                                 ('experiments/tiny_circles/train.py', 'models/unet/default.py', 'models/unet/unet.py')})
    if args.resume:
        previous = json.loads((args.out / 'config.json').read_text())
        for key in ('arch', 'epochs', 'batch', 'lr', 'seed', 'graph'):
            if previous[key] != config[key]:
                raise ValueError(f'Resume configuration mismatch: {key}')
    else:
        (args.out / 'config.json').write_text(json.dumps(config, indent=2))
    print(json.dumps(config), flush=True)
    if args.smoke:
        # Exercise full and partial batches, FP32 validation and portable checkpoint reload.
        for _ in range(8):
            step(images['train'][:args.batch].float().div(255).contiguous(memory_format=torch.channels_last))
        torch.cuda.synchronize()
        start = time.monotonic()
        step.total.zero_()
        for _ in range(30):
            step(images['train'][:args.batch].float().div(255).contiguous(memory_format=torch.channels_last))
        torch.cuda.synchronize()
        seconds = (time.monotonic() - start) / 30
        step(images['train'][:31].float().div(255).contiguous(memory_format=torch.channels_last))
        loss = evaluate(step.ema, images['val'][:64], 32, 193)
        result = dict(seconds_per_batch=seconds, estimated_train_epoch_seconds=seconds*math.ceil(48000/args.batch),
                      startup_seconds=time.monotonic()-started, validation_loss=loss,
                      peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                      peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                      finite=all(bool(torch.isfinite(p).all()) for p in model.parameters()))
        (args.out / 'smoke.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        return
    history: list[dict[str, float | int]] = []
    best = float('inf')
    start_epoch = 0
    if args.resume:
        state = torch.load(args.out / 'latest.pt', weights_only=False, map_location='cuda:1')
        step.restore(state)
        torch.cuda.set_rng_state(state['cuda_rng'].cpu(), 1)
        torch.set_rng_state(state['cpu_rng'].cpu())
        start_epoch, best, history = state['epoch'], state['best'], state['history']
    else:
        baseline_dir = Path('outputs/fm_unet/circles-64-noattn-ch16-b8-e200/n1')
        baseline_config = json.loads((baseline_dir / 'config.json').read_text())
        baseline = Unet(**baseline_config['arch'], dropout=0.).to('cuda:1', memory_format=torch.channels_last)
        checkpoint = baseline_dir / 'checkpoints/epoch_0200.pt'
        weights = torch.load(checkpoint, weights_only=True, map_location='cuda:1')
        baseline.load_state_dict({k.removeprefix('module._orig_mod.').removeprefix('module.'):v
                                  for k,v in weights.items() if k != 'n_averaged'})
        reference = dict(parameters=baseline_config['params'], checkpoint=str(checkpoint),
                         checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                         validation_loss=evaluate(baseline, images['val'], 128, 193),
                         test_loss=evaluate(baseline, images['test'], 128, 194))
        (args.out / 'baseline.json').write_text(json.dumps(reference, indent=2))
        samples(baseline, args.out, 'baseline')
        samples(baseline, args.out, 'baseline', 5)
        del baseline, weights
        torch.cuda.empty_cache()
        print('BASELINE', reference, flush=True)
    stop = False

    def request_stop(signum: int, frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        step.lr.fill_(1e-5 + (args.lr - 1e-5)*(1 + math.cos(math.pi*(epoch-1)/args.epochs))/2)
        step.total.zero_()
        permutation = torch.randperm(len(images['train']), device='cuda:1')
        torch.cuda.synchronize()
        start = time.monotonic()
        for indices in permutation.split(args.batch):
            batch = images['train'][indices].float().div_(255).contiguous(memory_format=torch.channels_last)
            step(batch)
        train_loss = float(step.total / len(images['train']))
        train_seconds = time.monotonic() - start
        validation = evaluate(step.ema, images['val'], 128, 193)
        if not math.isfinite(train_loss + validation):
            raise RuntimeError('Nonfinite training or validation loss')
        history.append(dict(epoch=epoch, train_loss=train_loss, validation_loss=validation,
                            train_seconds=train_seconds))
        if validation < best:
            best = validation
            atomic_save(step.ema.state_dict(), args.out / 'best.pt')
        atomic_save(dict(model=model.state_dict(), ema=step.ema.state_dict(), optimizer=step.optimizer.state_dict(),
                         updates=step.updates, cuda_rng=torch.cuda.get_rng_state(1), cpu_rng=torch.get_rng_state(),
                         epoch=epoch, best=best, history=history), args.out / 'latest.pt')
        (args.out / 'history.json').write_text(json.dumps(history, indent=2))
        (args.out / 'status.json').write_text(json.dumps(dict(pid=os.getpid(), epoch=epoch,
            status='paused' if stop else 'running', best_validation_loss=best,
            updated_at=datetime.now().astimezone().isoformat()), indent=2))
        print(json.dumps(history[-1]), flush=True)
        if epoch % 100 == 0 or epoch == args.epochs:
            atomic_save(step.ema.state_dict(), args.out / f'epoch_{epoch:04d}.pt')
            samples(step.ema, args.out, f'epoch_{epoch:04d}')
            samples(step.ema, args.out, f'epoch_{epoch:04d}', 5)
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.semilogy([row['epoch'] for row in history], [row['train_loss'] for row in history], label='Tiny model train')
            ax.semilogy([row['epoch'] for row in history], [row['validation_loss'] for row in history], label='Tiny EMA validation (fixed draws)')
            reference = json.loads((args.out / 'baseline.json').read_text())
            ax.axhline(reference['validation_loss'], color='black', linestyle='--', label='472k model validation')
            ax.set(xlabel='Epoch', ylabel='Flow-matching MSE')
            ax.legend()
            fig.tight_layout()
            fig.savefig(args.out / 'loss_curves.png', dpi=150)
            plt.close(fig)
        if stop:
            return
    step.ema.load_state_dict(torch.load(args.out / 'best.pt', weights_only=True, map_location='cuda:1'))
    result = dict(status='complete', epochs=args.epochs, best_validation_loss=best,
                  best_checkpoint_test_loss=evaluate(step.ema, images['test'], 128, 194),
                  elapsed_seconds=time.monotonic()-started)
    samples(step.ema, args.out, 'best')
    samples(step.ema, args.out, 'best', 5)
    (args.out / 'status.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
