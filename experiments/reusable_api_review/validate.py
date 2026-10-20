"""CPU-only terminal-core timings and paired short CT/MRI trajectories."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from dlr_dflow.api import approx_source_gradient, load_flow_model, sensitivity_factors
from dlr_dflow.implementation.enriched import _canonicalize_factors
from experiments.ct_mri_sweep.operators import CT, MRI, noisy_measurement
from experiments.ct_mri_sweep.optimizer import SweepLBFGS
from experiments.ct_mri_sweep.run import Evaluator
from p_flow import endpoint_gradient
from sampler.ode_samplers import euler_flow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(191)
    timings = []
    for rank in (8, 256, 512):
        u, v = (torch.linalg.qr(torch.randn(4096, rank), mode='reduced')[0] for _ in range(2))
        factors = u, torch.randn(rank, rank), v
        gx = torch.randn(4096)
        elapsed = [[], []]
        actions = []
        for repeat in range(5):
            for variant in ((0, 1) if repeat % 2 else (1, 0)):
                start = time.perf_counter()
                owned = _canonicalize_factors(factors) if variant == 0 else tuple(t.clone() for t in factors)
                action = approx_source_gradient(owned, gx, 0.)
                seconds = time.perf_counter() - start
                if repeat:
                    elapsed[variant].append(seconds)
                if repeat == 4:
                    actions.append((variant, action))
        actions.sort(key=lambda item: item[0])
        relative = float((actions[0][1] - actions[1][1]).norm() / actions[0][1].norm())
        assert relative < 1e-5
        timings.append(dict(rank=rank, dimension=4096,
                            canonical_seconds=statistics.median(elapsed[0]),
                            general_owned_seconds=statistics.median(elapsed[1]),
                            action_relative_difference=relative))
    (args.out / 'tail_timings.json').write_text(json.dumps(timings, indent=2))
    run = Path('outputs/fm_unet/circles-64-noattn-ch4-e500-mbtuned-20261019/n1')
    model, _ = load_flow_model(run_dir=run, epoch=500)
    truth = torch.load('data/circles-64-n1.pt', weights_only=True, mmap=True)['test'][0:1].float() / 255
    initial = torch.randn_like(truth)
    dlr = dict(rank=8, trace_probes=0, enrichment_probes=0, initialization_oversampling=8,
               zero_terminal_baseline=True)
    results = []
    for operator in (CT(64, 10, torch.device('cpu')), MRI(64, 4, 191, torch.device('cpu'))):
        clean = operator(truth)
        observed = noisy_measurement(clean, .05, 192, getattr(operator, 'mask', None))
        with torch.no_grad():
            endpoint = euler_flow(model, initial, 5)
        _, gx = endpoint_gradient(endpoint, operator, observed)
        gradients = []
        for canonical in (True, False):
            _, factors, alpha = sensitivity_factors(
                model, initial, 5, **dlr, seed=193, _canonicalize_terminal=canonical)
            gradients.append(approx_source_gradient(factors, gx, alpha))
        fixed_error = float((gradients[0] - gradients[1]).norm() / gradients[0].norm())
        assert fixed_error < 1e-5
        trajectories = []
        for canonical in (True, False):
            source = initial.clone().requires_grad_(True)
            settings = dict(flow_steps=5, dlr={**dlr, '_canonicalize_terminal': canonical},
                            dlr_reuse_previous_v=True)
            evaluator = Evaluator(model, operator, observed, settings, 193)
            evaluator.method = 'DLR D-Flow'
            optimizer = SweepLBFGS(source, evaluator.objective, lr=1., history_size=100,
                                  max_ls=30, c1=1e-4, step_cap=.64, project=False,
                                  backtracking='geometric')

            def closure() -> torch.Tensor:
                value = evaluator.objective(source)
                source.grad = evaluator.gradient(source, 'DLR D-Flow')
                return source.new_tensor(value)

            rows = []
            for step in range(6):
                optimizer.step(closure)
                evaluator.objective(source)
                rows.append(dict(step=step + 1, phi=evaluator.value,
                                 image_relative_l2=float((evaluator.image - truth).norm() / truth.norm()),
                                 function_evaluations=optimizer.last_function_evaluations))
            trajectories.append(dict(canonical=canonical, rows=rows, source=source.detach().clone()))
        source_error = float((trajectories[0]['source'] - trajectories[1]['source']).norm()
                             / trajectories[0]['source'].norm())
        assert source_error < 1e-4
        assert [row['function_evaluations'] for row in trajectories[0]['rows']] == [
            row['function_evaluations'] for row in trajectories[1]['rows']]
        for trajectory in trajectories:
            del trajectory['source']
        record = dict(problem=operator.metadata['kind'], fixed_source_gradient_relative_error=fixed_error,
                      final_source_relative_difference=source_error, trajectories=trajectories)
        results.append(record)
        print(json.dumps(record), flush=True)
    report = dict(device='cpu', threads=2, steps=6, flow_steps=5, noise=.05,
                  checkpoint_sha256=hashlib.sha256((run / 'checkpoints/epoch_0500.pt').read_bytes()).hexdigest(),
                  interpretation='Short numerical pairing only; not a converged accuracy study or GPU benchmark.',
                  cases=results)
    (args.out / 'paired_trajectories.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
