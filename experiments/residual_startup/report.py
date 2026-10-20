"""Summarize completed paired startup trials and render standard panels."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from scripts.reconstruction_figures import plot_reconstruction_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads((args.out / 'reconstructions.json').read_text())
    summaries = []
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
    for index, problem in enumerate(('ct', 'mri')):
        records = []
        for row in filter(lambda row: row['problem'] == problem, rows):
            path = args.out / f'circles_{problem}' / f"{row['variant']}.pt"
            result = torch.load(path, weights_only=True, map_location='cpu')
            records.append((row, result))
        common_step = min(record['history'][-1]['step'] for _, record in records)
        for row, record in records:
            history = record['history']
            summaries.append({**row, 'final_image_percent': 100 * row['image_relative_l2'],
                              'common_step': common_step,
                              'common_step_image_percent': 100 * history[common_step]['image_relative_l2']})
            inputs = torch.load(args.out / f'circles_{problem}' / 'paired_inputs.pt',
                                weights_only=True, map_location='cpu')
            panel = plot_reconstruction_comparison(
                inputs['truth'], inputs['clean'],
                {'DLR D-Flow': (record['image'], record['measurement'])},
                observation=inputs['observed'], measurement_log=problem == 'mri',
                title=f"Resolution: 64², Inverse Problem: {problem.upper()}, Noise Level 5%, Sample: 0\n"
                      f"Initialization: {row['variant']}")
            panel.savefig(path.with_suffix('.png'), dpi=120)
            plt.close(panel)
            axes[index, 0].plot([h['step'] for h in history],
                                [100 * h['image_relative_l2'] for h in history], label=row['variant'])
            axes[index, 1].semilogy([h['step'] for h in history], [h['phi'] for h in history],
                                   label=row['variant'])
        axes[index, 0].set_ylabel(f'{problem.upper()} image error (%)')
        axes[index, 1].set_ylabel(f'{problem.upper()} measurement objective')
    for ax in axes.flat:
        ax.set_xlabel('Optimizer attempts')
        ax.legend()
        ax.grid(alpha=.2)
    figure.savefig(args.out / 'convergence.png', dpi=150)
    figure.savefig(args.out / 'convergence.pdf')
    plt.close(figure)
    (args.out / 'summary.json').write_text(json.dumps(summaries, indent=2))
    with (args.out / 'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == '__main__':
    main()
