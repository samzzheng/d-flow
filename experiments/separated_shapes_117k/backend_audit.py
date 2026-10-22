"""Check corrected accelerated derivatives against native autodiff at each resolution."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from dlr_dflow.api import approx_source_gradient, load_flow_model, sensitivity_factors
from dlr_dflow.implementation import enriched, tangent_unet
from dlr_dflow.implementation.graphed_products import GraphedAProducts
from experiments.ct_mri_sweep.queue import source_hashes
from experiments.ct_mri_sweep.run import write_json
from experiments.ct_mri_sweep.validation import ChunkedAutodiffProducts
from sampler.ode_samplers import euler_flow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
        default=Path("outputs/separated_shapes_117k_noiseless_20261021/production/manifest.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[64, 96, 128, 160, 200, 250])
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda:1")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.15, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    tangent_unet.GN_STATS_MODE = "native"
    tangent_unet.GN_ACTION_BACKEND = "triton"
    tangent_unet.GN_PRIMAL_BACKEND = "full"
    enriched.SPECIALIZE_ZERO_ENRICHMENT = True
    enriched.SKIP_UNTRUNCATED_SVD = True
    settings = json.loads(args.manifest.read_text())["settings"]
    report = dict(status="running", source_hashes=source_hashes(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        settings=settings, rows=[],
        scope="Backend execution equivalence, not low-rank versus exact-gradient accuracy")
    for resolution in args.resolutions:
        folder = Path(settings["run_template"].format(resolution=resolution))
        model, _ = load_flow_model(resolution, 0, settings["epoch"], str(device), folder)
        seed = settings["seed"] + 10000*resolution
        generator = torch.Generator().manual_seed(seed)
        source = torch.randn(1, 1, resolution, resolution, generator=generator).to(device)
        directions = torch.randn(resolution**2, 3, generator=generator).to(device)
        cotangents = torch.randn(resolution**2, 3, generator=generator).to(device)
        products = GraphedAProducts(model, source, share_linearization=True,
            full_step_graphs=True, max_jvp_columns=20, max_vjp_columns=16,
            max_step_jvp_columns=6, max_step_vjp_columns=4)
        reference = ChunkedAutodiffProducts(model)
        row = dict(resolution=resolution, seed=seed,
            checkpoint_sha256=hashlib.sha256((folder / f"checkpoints/epoch_{settings['epoch']:04d}.pt").read_bytes()).hexdigest(),
            source_sha256=hashlib.sha256(source.cpu().numpy().tobytes()).hexdigest(),
            local=[], factor_gradients=[])
        point = source
        steps = settings["flow_steps"]
        for index in range(steps):
            time = index/steps
            products.set_state(time, point)
            reference.set_state(time, point)
            forward, transpose = products.A(directions).clone(), products.AT(cotangents).clone()
            expected_forward, expected_transpose = reference.A(directions), reference.AT(cotangents)
            row["local"].append(dict(time=time,
                jvp_relative_error=float((forward-expected_forward).norm()/expected_forward.norm()),
                vjp_relative_error=float((transpose-expected_transpose).norm()/expected_transpose.norm()),
                adjoint_error=float(((forward*cotangents).sum()-(directions*transpose).sum()).abs()
                    /(forward.norm()*cotangents.norm()+directions.norm()*transpose.norm()))))
            with torch.no_grad():
                point = point + model(point, point.new_tensor([time]))/steps
        with torch.no_grad():
            endpoint = euler_flow(model, source, steps)
        terminal = cotangents[:, 0].reshape_as(source)
        basis = None
        for seeded in (False, True):
            fast_image, factors, alpha = sensitivity_factors(model, source, steps,
                **settings["dlr"], seed=seed+30000000, products=products, seed_basis=basis)
            native_image, native_factors, native_alpha = sensitivity_factors(model, source, steps,
                **settings["dlr"], seed=seed+30000000, products=reference, seed_basis=basis)
            actual = approx_source_gradient(factors, terminal, alpha)
            expected = approx_source_gradient(native_factors, terminal, native_alpha)
            row["factor_gradients"].append(dict(seeded=seeded,
                relative_error=float((actual-expected).norm()/expected.norm()),
                endpoint_relative_error=float((fast_image-endpoint).norm()/endpoint.norm()),
                native_endpoint_relative_error=float((native_image-endpoint).norm()/endpoint.norm()),
                alpha=float(alpha), native_alpha=float(native_alpha)))
            basis = factors[2].detach().clone()
        row["passed"] = (all(item["jvp_relative_error"] < 1e-3 and item["vjp_relative_error"] < 1e-3
                             and item["adjoint_error"] < 1e-5 for item in row["local"])
            and all(item["relative_error"] < .01 and item["endpoint_relative_error"] < 1e-5
                    and item["native_endpoint_relative_error"] < 1e-5
                    and item["alpha"] == item["native_alpha"] == 0. for item in row["factor_gradients"]))
        report["rows"].append(row)
        write_json(args.out, report)
        print(json.dumps(row), flush=True)
        del model, products, reference, factors, native_factors, basis
        torch.cuda.empty_cache()
    report.update(status="complete", passed=all(row["passed"] for row in report["rows"]))
    write_json(args.out, report)
    if not report["passed"]:
        raise RuntimeError("Backend equivalence failed; inspect the per-resolution audit")


if __name__ == "__main__":
    main()
