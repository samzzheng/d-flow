"""Paired initialization pilot; existing checkpoints and optimizer, no training."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import torch

from dlr_dflow.api import approx_source_gradient, load_flow_model, sensitivity_factors
from dlr_dflow.implementation.graphed_products import GraphedAProducts
from experiments.ct_mri_sweep.operators import CT, MRI, noisy_measurement
from experiments.ct_mri_sweep.run import Evaluator, solve, write_json
from p_flow import endpoint_gradient
from sampler.ode_samplers import euler_flow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-updates', type=int, default=1000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(.25, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    variants = {'legacy': None, 'gaussian_one': 1, 'gaussian_two': 2}
    diagnostics, reconstructions = [], []
    sources = ['dlr_dflow/implementation/enriched.py',
               'dlr_dflow/implementation/residual_startup.py', __file__]
    hashes = {str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in sources}
    for dataset, directory in (
        ('circles', 'five_step_noise005_n124_20261013'),
        ('shapes', 'separated_shapes_noiseless_20261016'),
    ):
        settings = json.loads(Path('outputs', directory, 'production/manifest.json').read_text())['settings']
        run = Path(settings['run_template'].format(resolution=64, circles=1))
        model, _ = load_flow_model(run_dir=run, epoch=settings['epoch'], device=args.device)
        data = Path('data/circles-64-n1.pt' if dataset == 'circles' else 'data/separated-shapes-64.pt')
        truth = torch.load(data, weights_only=True, mmap=True)['test'][0:1].to(device).float() / 255
        initial = torch.randn(truth.shape, generator=torch.Generator().manual_seed(606)).to(device)
        checkpoint_hash = hashlib.sha256((run / f"checkpoints/epoch_{settings['epoch']:04d}.pt").read_bytes()).hexdigest()
        products = GraphedAProducts(model, initial) if device.type == 'cuda' else None
        for problem, operator in (('ct', CT(64, 10, device)), ('mri', MRI(64, 4, 607, device))):
            case = args.out / f'{dataset}_{problem}'
            case.mkdir(exist_ok=True)
            clean = operator(truth)
            noise = .05 if dataset == 'circles' else 0.
            observed = noisy_measurement(clean, noise, 608, getattr(operator, 'mask', None))
            torch.save(dict(truth=truth.cpu(), initial=initial.cpu(), clean=clean.cpu(),
                            observed=observed.cpu(), operator=operator.metadata), case / 'paired_inputs.pt')
            for steps in (5, 100):
                source = initial.clone().requires_grad_(True)
                image = euler_flow(model, source, steps)
                gx_loss = .5 * (operator(image) - observed).abs().square().sum()
                exact, = torch.autograd.grad(gx_loss, source)
                _, gx = endpoint_gradient(image.detach(), operator, observed)
                for variant, startup in variants.items():
                    options = {**settings['dlr'], 'residual_sketch_steps': startup, 'seed': 609}
                    endpoint, factors, alpha = sensitivity_factors(
                        model, initial, steps, **options, products=products)
                    approx = approx_source_gradient(factors, gx, alpha)
                    v = factors[2]
                    projected = (v @ (v.T @ exact.flatten())).reshape_as(exact)
                    record = dict(dataset=dataset, problem=problem, flow_steps=steps, variant=variant,
                                  gradient_relative_error=float((approx - exact).norm() / exact.norm()),
                                  outside_v_relative=float((exact - projected).norm() / exact.norm()),
                                  coefficient_relative=float((approx - projected).norm() / exact.norm()),
                                  rank=factors[1].shape[0],
                                  endpoint_relative_error=float((endpoint - image.detach()).norm() / image.detach().norm()),
                                  cosine=float(torch.nn.functional.cosine_similarity(approx.flatten(), exact.flatten(), dim=0)))
                    if steps == 5:
                        eager = sensitivity_factors(model, initial, steps, **options)
                        eager_gradient = approx_source_gradient(eager[1], gx, eager[2])
                        record['graph_vs_eager_gradient_relative'] = float(
                            (approx - eager_gradient).norm() / eager_gradient.norm())
                        assert record['graph_vs_eager_gradient_relative'] < .01
                    assert record['endpoint_relative_error'] < 1e-4
                    diagnostics.append(record)
                    write_json(args.out / 'diagnostics.json', diagnostics)
            # Complete the circle CT/MRI pilot first; shape fixed-source checks
            # broaden the diagnostic without launching a second large sweep.
            if dataset != 'circles':
                continue
            for variant, startup in variants.items():
                config = deepcopy(settings)
                config['dlr']['residual_sketch_steps'] = startup
                config['dlr_reuse_previous_v'] = False
                config['stopping']['max_updates'] = args.max_updates
                config['stopping']['min_updates'] = args.max_updates
                config['execution']['checkpoint_every'] = 50
                protocol = dict(checkpoint_sha256=checkpoint_hash, sources=hashes, settings=config,
                                source_seed=606, mask_seed=607, noise_seed=608, dlr_seed=609,
                                sample=0, noise=noise, variant=variant, device=args.device)
                evaluator = Evaluator(model, operator, observed, config, 609, products)
                evaluator.method = 'DLR D-Flow'
                # Warm the complete refresh, then time repeated fixed-source calls.
                evaluator.gradient(initial, 'DLR D-Flow')
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                for _ in range(5):
                    evaluator.gradient(initial, 'DLR D-Flow')
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                gradient_seconds = (time.perf_counter() - start) / 5
                evaluator = Evaluator(model, operator, observed, config, 609, products)
                result = solve(evaluator, 'DLR D-Flow', initial, truth, clean,
                               case / f'{variant}.pt', protocol, [False])
                row = dict(dataset=dataset, problem=problem, variant=variant, status=result['status'],
                           steps=result['history'][-1]['step'], gradient_seconds=gradient_seconds,
                           elapsed_seconds=result['elapsed_seconds'], **{
                               name: result['history'][-1][name] for name in
                               ('image_relative_l2', 'measurement_relative_l2_clean', 'phi')})
                reconstructions.append(row)
                write_json(args.out / 'reconstructions.json', reconstructions)
                print(json.dumps(row), flush=True)
        del products, model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    write_json(args.out / 'complete.json', dict(status='complete', diagnostics=len(diagnostics),
                                               solves=len(reconstructions), device=args.device))


if __name__ == '__main__':
    main()
