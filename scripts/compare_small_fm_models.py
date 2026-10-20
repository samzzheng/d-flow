"""Paired held-out FM losses and unconditional samples for a model-capacity pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.unet.default import Unet
from sampler.ode_samplers import ode_integrate
from train_fm_unet import evaluate_split


@torch.no_grad()
def compare_models(baseline: Path, candidate: Path, device: str, micro_batch: int = 128,
                   sample_count: int = 128) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    torch.set_num_threads(4)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records: dict[str, dict] = {}
    generated: dict[str, torch.Tensor] = {}
    configs = {name: json.loads((folder / 'config.json').read_text())
               for name, folder in [('baseline', baseline), ('small', candidate)]}
    data_path = Path(configs['small']['data'])
    if Path(configs['baseline']['data']).resolve() != data_path.resolve():
        raise ValueError('Models must use the same dataset')
    data = torch.load(data_path, map_location='cpu', weights_only=True, mmap=True)
    generator = torch.Generator(device=device).manual_seed(4271)
    initial = torch.randn(sample_count, 1, configs['small']['im_size'], configs['small']['im_size'],
                          device=device, generator=generator)
    for name, folder in [('baseline', baseline), ('small', candidate)]:
        config = configs[name]
        checkpoint = folder / 'checkpoints' / f'epoch_{config["epochs"]:04d}.pt'
        state = torch.load(checkpoint, weights_only=True, map_location=device)
        model = Unet(**config['arch'], dropout=0.).to(device).eval()
        model.load_state_dict({key.removeprefix('module._orig_mod.').removeprefix('module.'): value
                               for key, value in state.items() if key != 'n_averaged'})
        record = dict(parameters=sum(p.numel() for p in model.parameters()), epochs=config['epochs'],
                      checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        for split, seed in [('val', 193), ('test', 194)]:
            # The same batch partition, t and source draws for both networks.
            torch.manual_seed(seed)
            record[split + '_fm_mse'] = evaluate_split(model, data[split], device, batch=128, micro_batch=micro_batch)
        def velocity(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return model(x, t.expand(len(x)))
        for steps in (5, 100):
            generated[f'{name}_{steps}'] = torch.cat([
                ode_integrate(ode_func=velocity, init_x=batch.clone(),
                              ode_opts={'options': {'step_size': 1 / steps}},
                              sampler='euler').cpu()
                for batch in initial.split(micro_batch)
            ])
        records[name] = record
        del model, state
    report = dict(models=records, validation_ratio=records['small']['val_fm_mse']/records['baseline']['val_fm_mse'],
                  test_ratio=records['small']['test_fm_mse']/records['baseline']['test_fm_mse'],
                  sample_count=sample_count, source_seed=4271,
                  source_sha256=hashlib.sha256(initial.cpu().numpy().tobytes()).hexdigest(),
                  protocol=f'Final EMA checkpoints; paired full validation/test sets, batch128 and microbatch{micro_batch}; standard BF16 FM evaluation, {sample_count} paired FP32 Euler samples; TF32 off.',
                  limitation='Unequal training epochs are intentional. FM loss and unconditional samples do not establish inverse-reconstruction accuracy.')
    (candidate / 'baseline_comparison.json').write_text(json.dumps(report, indent=2))
    torch.save(generated, candidate / 'eval' / 'capacity_comparison_samples.pt')
    torch.save(initial.cpu(), candidate / 'eval' / 'capacity_comparison_sources.pt')
    for steps in (5, 100):
        for start in range(0, sample_count, 8):
            fig, axes = plt.subplots(2, 8, figsize=(16, 4))
            for row, name in enumerate(('baseline', 'small')):
                for column, ax in enumerate(axes[row]):
                    index = start + column
                    if index >= sample_count:
                        ax.set_axis_off()
                        continue
                    ax.imshow(generated[f'{name}_{steps}'][index, 0].numpy(), cmap='gray', vmin=0, vmax=1)
                    ax.set_title(f'Sample {index}')
                    ax.set_xticks([]); ax.set_yticks([])
                    if column == 0:
                        ax.set_ylabel(f'{records[name]["parameters"]:,} parameters')
            fig.suptitle(f'{steps}-step Euler; paired sources; display [0,1], raw samples saved')
            fig.tight_layout()
            suffix = '' if start == 0 else f'_page{start // 8 + 1:02d}'
            fig.savefig(candidate / 'eval' / f'capacity_comparison_{steps}steps{suffix}.png', dpi=150)
            plt.close(fig)
    print('CAPACITY COMPARISON', json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--sample-count', type=int, default=128)
    args = parser.parse_args()
    compare_models(args.baseline, args.candidate, args.device, sample_count=args.sample_count)
