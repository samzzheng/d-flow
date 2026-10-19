"""Compare ordinary training with checkpoint-resumed CUDA graph training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from unittest.mock import patch

import torch

import train_fm_unet as trainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resolution', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--variant', choices=('native', 'graph'))
    parser.add_argument('--stop-epoch', type=int, default=2)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dataset_path = args.out / 'data.pt'
    if args.variant:
        torch.manual_seed(191)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        run = args.out / args.variant
        argv = ['train_fm_unet.py', '--data', str(dataset_path), '--out-dir', str(run),
                '--device', 'cuda:1', '--epochs', '20', '--batch', '512',
                '--micro-batch', '512' if args.resolution == 64 else '32',
                '--lr', '0.0002', '--wd', '0.0001', '--ema-decay', '0.999',
                '--sample-steps', '2', '--ch', '4', '--ch-mul', '1,2,2',
                '--att-channels', '0,0,0', '--groups', '2', '--no-mid-attn',
                '--num-workers', '0', '--threads', '4', '--no-compile',
                '--gpu-resident', '--channels-last', '--fused-adam', '--save-training-state']
        if args.variant == 'graph':
            argv.append('--graph-microbatches')
        if args.resume:
            argv.extend(['--resume-training-state', str(run / 'checkpoints/training_state.pt'),
                         '--start-epoch', '1'])
        save_history = trainer.save_loss_history

        def save_and_stop(train_losses: list[float], val_losses: list[float], out_dir: Path) -> None:
            save_history(train_losses, val_losses, out_dir)
            if len(train_losses) == args.stop_epoch:
                os.kill(os.getpid(), signal.SIGTERM)

        with patch.object(sys, 'argv', argv), patch.object(trainer, 'save_loss_history', save_and_stop):
            trainer.main()
        return

    data = torch.load(f'data/circles-{args.resolution}-n1.pt', mmap=True, weights_only=True)
    torch.save({key: data[key][:count].clone() for key, count in
                (('train', 600), ('val', 16), ('test', 16))}, dataset_path)
    for variant, stop_epoch, resume in (('native', 2, False), ('graph', 1, False), ('graph', 2, True)):
        command = [sys.executable, '-u', '-m', 'experiments.tiny_circles.verify_graph_trainer',
                   '--resolution', str(args.resolution), '--out', str(args.out),
                   '--variant', variant, '--stop-epoch', str(stop_epoch)]
        if resume:
            command.append('--resume')
        with (args.out / f'{variant}_{stop_epoch}.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    states = [torch.load(args.out / name / 'checkpoints/training_state.pt',
                         map_location='cpu', weights_only=True) for name in ('native', 'graph')]
    torch.testing.assert_close(states[0], states[1], rtol=0, atol=0)
    histories = [json.loads((args.out / name / 'loss_history.json').read_text())
                 for name in ('native', 'graph')]
    assert histories[0] == histories[1]
    for name in ('native', 'graph'):
        run = args.out / name
        assert (run / 'loss_curves.png').exists()
        assert list((run / 'eval').glob('epoch_*.png'))
        assert list((run / 'checkpoints').glob('epoch_*.pt'))
    result = dict(status='passed', resolution=args.resolution, epochs=2,
                  train_samples=600, updates=int(states[0]['updates']),
                  comparison='bitwise full training state, RNG, optimizer, EMA and loss history',
                  graph_resumed_after_epoch=1, sample_steps_for_smoke_only=2)
    (args.out / 'verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
