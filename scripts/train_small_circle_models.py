"""Train the 22k circles-64 pilot through the standard FM trainer."""
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
    parser.add_argument('--out-dir', type=Path, default=Path('outputs/fm_unet/circles-64-noattn-ch4-e1000/n1'))
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root))
    torch.manual_seed(20261019)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    command = ['train_fm_unet.py', '--data', 'data/circles-64-n1.pt',
        '--out-dir', str(args.out_dir), '--device', args.device, '--epochs', str(args.epochs),
        '--batch', '512', '--micro-batch', '512', '--lr', '0.0002', '--wd', '0.0001',
        '--ema-decay', '0.999', '--t-alpha', '1', '--sample-steps', '100',
        '--ch', '4', '--ch-mul', '1,2,2', '--att-channels', '0,0,0', '--groups', '2',
        '--no-mid-attn', '--num-workers', '0', '--threads', '4', '--no-compile',
        '--gpu-resident', '--channels-last', '--fused-adam', '--save-training-state']
    checkpoint = args.out_dir / 'checkpoints/training_state.pt'
    if checkpoint.exists():
        state = torch.load(checkpoint, weights_only=True, map_location='cpu', mmap=True)
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
