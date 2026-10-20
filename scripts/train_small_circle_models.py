"""Train a small circle model through the standard FM trainer."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--resolution', type=int, choices=(64, 96, 128, 160, 200, 250), default=64)
    parser.add_argument('--n-circles', type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument('--ch', type=int, choices=(4, 6, 8), default=4,
                        help='Base channel width')
    parser.add_argument('--ch-mul', type=int, nargs='+', default=[1, 2, 2],
                        help='Channel multipliers, one per spatial level')
    parser.add_argument('--time-embed-dim', type=int, default=None,
                        help='Sinusoidal time feature count, independent of base width')
    parser.add_argument('--micro-batch', type=int, help='Override the calibrated microbatch; effective batch stays 512')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root))
    data_path = Path(f'data/circles-{args.resolution}-n{args.n_circles}.pt')
    time_suffix = f'-time{args.time_embed_dim}' if args.time_embed_dim is not None else ''
    depth_suffix = '-mul' + '-'.join(map(str, args.ch_mul)) if args.ch_mul != [1, 2, 2] else ''
    args.out_dir = args.out_dir or Path(f'outputs/fm_unet/circles-{args.resolution}-noattn-ch{args.ch}{depth_suffix}{time_suffix}-e{args.epochs}/n{args.n_circles}')
    profile = json.loads((root / 'experiments/tiny_circles/microbatch_profile.json').read_text())
    micro_batch = args.micro_batch or profile['micro_batches'][str(args.resolution)]
    if args.micro_batch is not None and not 1 <= args.micro_batch <= 512:
        parser.error('--micro-batch must be between 1 and 512')
    if not data_path.exists():
        parser.error(f'Missing dataset: {data_path}')
    if any(multiplier < 1 for multiplier in args.ch_mul):
        parser.error('--ch-mul entries must be positive')
    torch.manual_seed(20261019)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    command = ['train_fm_unet.py', '--data', str(data_path),
        '--out-dir', str(args.out_dir), '--device', args.device, '--epochs', str(args.epochs),
        '--batch', '512', '--micro-batch', str(micro_batch), '--lr', '0.0002', '--wd', '0.0001',
        '--ema-decay', '0.999', '--t-alpha', '1', '--sample-steps', '100',
        '--ch', str(args.ch), '--ch-mul', ','.join(map(str, args.ch_mul)),
        '--att-channels', ','.join('0' for _ in args.ch_mul), '--groups', '2',
        '--no-mid-attn', '--num-workers', '0', '--threads', '4', '--no-compile',
        '--gpu-resident', '--channels-last', '--fused-adam', '--save-training-state',
        '--graph-microbatches']
    if args.time_embed_dim is not None:
        command.extend(['--time-embed-dim', str(args.time_embed_dim)])
    checkpoint = args.out_dir / 'checkpoints/training_state.pt'
    if checkpoint.exists():
        state = torch.load(checkpoint, weights_only=True, map_location='cpu', mmap=True)
        config = json.loads((args.out_dir / 'config.json').read_text())
        if (config['epochs'] != args.epochs or Path(config['data']).resolve() != data_path.resolve()
                or config['arch']['ch'] != args.ch
                or config['arch']['ch_mul'] != args.ch_mul
                or config['arch'].get('time_embed_dim', args.ch)
                != (args.ch if args.time_embed_dim is None else args.time_embed_dim)):
            parser.error('Resume settings do not match the existing run')
        if state['epoch'] >= args.epochs:
            parser.error('Training has already reached the requested epoch count')
        command.extend(['--resume-training-state', str(checkpoint), '--start-epoch', str(state['epoch'])])
        del state
    elif (args.out_dir / 'config.json').exists():
        parser.error('Existing run has no full training state; refusing to overwrite it')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    record = dict(pid=os.getpid(), started_at=datetime.now().astimezone().isoformat(),
                  command=command, seed=20261019,
                  trainer_sha256=hashlib.sha256((root / 'train_fm_unet.py').read_bytes()).hexdigest())
    (args.out_dir / 'launch.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)
    sys.argv = command
    runpy.run_path(str(root / command[0]), run_name='__main__')


if __name__ == '__main__':
    main()
