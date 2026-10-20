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
    labels = {'legacy': 'Existing startup', 'gaussian_one': 'Gaussian, 1 step',
              'gaussian_two': 'Gaussian, 2 steps'}
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
                      f"Initialization: {labels[row['variant']]}")
            panel.savefig(args.out / f'circles_{problem}' / f"{row['variant']}.png", dpi=120)
            plt.close(panel)
            axes[index, 0].plot([h['step'] for h in history],
                                [100 * h['image_relative_l2'] for h in history], label=labels[row['variant']])
            axes[index, 1].semilogy([h['step'] for h in history], [h['phi'] for h in history],
                                   label=labels[row['variant']])
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
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(summaries)
    diagnostics = json.loads((args.out / 'diagnostics.json').read_text())
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
    variants = ('legacy', 'gaussian_one', 'gaussian_two')
    for ax, (dataset, problem) in zip(axes.flat, (
            ('circles', 'ct'), ('circles', 'mri'), ('shapes', 'ct'), ('shapes', 'mri'))):
        for offset, steps in ((-.18, 5), (.18, 100)):
            selected = {row['variant']: row for row in diagnostics if row['dataset'] == dataset
                        and row['problem'] == problem and row['flow_steps'] == steps}
            ax.bar([index + offset for index in range(3)],
                   [100 * selected[name]['gradient_relative_error'] for name in variants],
                   width=.35, label=f'{steps} Euler steps')
        ax.set_xticks(range(3), [labels[name] for name in variants], rotation=15)
        ax.set_title(f'{dataset}, {problem.upper()}')
        ax.set_ylabel('Source-gradient relative error (%)')
        ax.legend()
    figure.savefig(args.out / 'gradient_errors.png', dpi=150)
    plt.close(figure)


if __name__ == '__main__':
    main()
