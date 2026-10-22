"""Paired fixed-source diagnostics and reconstruction controls for 117k DLR failures."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from dlr_dflow.api import approx_source_gradient, load_flow_model, sensitivity_factors
from dlr_dflow.implementation import enriched, tangent_unet
from dlr_dflow.implementation.graphed_products import GraphedAProducts
from experiments.ct_mri_sweep import run
from experiments.ct_mri_sweep.operators import CT, MRI
from experiments.ct_mri_sweep.queue import source_hashes
from sampler.ode_samplers import euler_flow
from scripts.reconstruction_figures import RESIDUAL_VMAX, plot_reconstruction_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("diagnose", "reconstruct"))
    parser.add_argument("--baseline", type=Path,
        default=Path("outputs/separated_shapes_117k_noiseless_20261021/production"))
    parser.add_argument("--out", type=Path,
        default=Path("outputs/separated_shapes_117k_diagnosis_20261022"))
    parser.add_argument("--cases", nargs="+", default=[
        "r250_shapes_ct10/sample_000", "r250_shapes_mri4/sample_001",
        "r250_shapes_mri4/sample_000"])
    parser.add_argument("--variants", nargs="+", choices=("baseline", "prefix2", "prefix5", "measurement", "random_mix"),
                        default=["prefix2", "prefix5"])
    parser.add_argument("--backend", choices=("fast", "native", "native_primal"), default="fast")
    parser.add_argument("--rank", type=int)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda:1")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.2, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    manifest = json.loads((args.baseline / "manifest.json").read_text())
    settings = manifest["settings"]
    if args.rank is not None:
        settings["dlr"]["rank"] = args.rank
    args.out.mkdir(parents=True, exist_ok=True)
    evidence = dict(baseline=str(args.baseline.resolve()), cases=args.cases,
                    variants=args.variants, backend=args.backend, rank=settings["dlr"]["rank"], source_sha256=source_hashes(),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    run.write_json(args.out / f"{args.action}_config.json", evidence)
    rows: list[dict[str, Any]] = []
    for name in args.cases:
        case = args.baseline / name
        inputs = torch.load(case / "inputs.pt", map_location=device, weights_only=True)
        protocol = inputs["protocol"]
        resolution = protocol["job"]["resolution"]
        folder = Path(settings["run_template"].format(resolution=resolution))
        assert hashlib.sha256((folder / "checkpoints/epoch_0500.pt").read_bytes()).hexdigest() == protocol["checkpoint_sha256"]
        model, _ = load_flow_model(resolution, 0, 500, str(device), folder)
        info = protocol["operator"]
        operator = (CT(resolution, protocol["job"]["level"], device)
                    if protocol["job"]["problem"] == "ct" else
                    MRI(resolution, 4, info["mask_seed"], device, info["pattern"]))
        if isinstance(operator, MRI):
            assert torch.equal(operator.mask, inputs["mask"])
        assert torch.equal(inputs["clean"], inputs["observed"])
        tangent_unet.GN_STATS_MODE = "native" if resolution <= 96 or args.backend != "fast" else "manual"
        tangent_unet.GN_ACTION_BACKEND = "native" if args.backend == "native" else "triton"
        tangent_unet.GN_PRIMAL_BACKEND = "native" if args.backend == "native" else "full"
        enriched.SPECIALIZE_ZERO_ENRICHMENT = True
        enriched.SKIP_UNTRUNCATED_SVD = True
        products = GraphedAProducts(model, inputs["initial"], share_linearization=True,
            full_step_graphs=True, max_jvp_columns=20, max_vjp_columns=16,
            max_step_jvp_columns=6, max_step_vjp_columns=4)
        if args.action == "diagnose":
            saved = torch.load(case / "dlr_d_flow.pt", map_location=device, weights_only=True)
            for stage, source in (("initial", inputs["initial"]), ("dlr_final", saved["source"])):
                variable = source.detach().requires_grad_(True)
                image = euler_flow(model, variable, settings["flow_steps"])
                residual = operator(image) - inputs["observed"]
                loss = .5 * residual.abs().square().sum(dtype=torch.float64)
                exact, = torch.autograd.grad(loss, variable)
                _, terminal = run.endpoint_gradient(image.detach(), operator, inputs["observed"])
                norm = float(exact.norm())
                for prefix in (1, 2, 5):
                    options = dict(settings["dlr"])
                    if prefix > 1:
                        options["residual_sketch_steps"] = prefix
                    endpoint, factors, alpha = sensitivity_factors(model, source, settings["flow_steps"],
                        **options, seed=protocol["dlr_seed"], products=products)
                    estimate = approx_source_gradient(factors, terminal, alpha)
                    v = factors[2]
                    projected = (v @ (v.T @ exact.reshape(-1, 1))).reshape_as(exact)
                    row = dict(case=name, stage=stage, prefix=prefix, backend=args.backend, retained_rank=v.shape[1],
                        exact_norm=norm, estimated_norm=float(estimate.norm()),
                        relative_gradient_error=float((estimate-exact).norm())/norm if norm else None,
                        relative_subspace_error=float((projected-exact).norm())/norm if norm else None,
                        relative_coefficient_error=float((estimate-projected).norm())/norm if norm else None,
                        cosine=float(torch.nn.functional.cosine_similarity(exact.flatten(), estimate.flatten(), dim=0)),
                        endpoint_relative_error=float((endpoint-image.detach()).norm()/image.detach().norm().clamp_min(1e-20)),
                        objective=float(loss.detach()), image_error=float((image.detach()-inputs["truth"]).norm()/inputs["truth"].norm()),
                        predicted_norm_ratio=float(image.detach().norm()/inputs["truth"].norm()),
                        blank_objective=float(.5*inputs["observed"].abs().square().sum(dtype=torch.float64)),
                        source_sha256=hashlib.sha256(source.detach().cpu().numpy().tobytes()).hexdigest(),
                        note="Fresh sketch at fixed source; no optimizer trajectory or previous-V reuse in this diagnostic")
                    rows.append(row)
                    run.write_json(args.out / "gradients.json", dict(status="running", rows=rows))
                    print(json.dumps(row), flush=True)
            del saved, image, exact, terminal, factors, estimate, endpoint, projected, variable, residual, loss, source, v
        else:
            for variant in args.variants:
                configuration = deepcopy(settings)
                if variant.startswith("prefix"):
                    configuration["dlr"]["residual_sketch_steps"] = int(variant[-1])
                initial = inputs["initial"]
                if variant in ("measurement", "random_mix"):
                    if variant == "measurement":
                        variable = torch.zeros_like(initial, requires_grad=True)
                        pairing = (operator(variable).conj() * inputs["observed"]).real.sum()
                        direction, = torch.autograd.grad(pairing, variable)
                    else:
                        direction = torch.randn(initial.shape, generator=torch.Generator().manual_seed(
                            protocol["source_seed"] + 40000000)).to(device)
                    direction = direction - direction.mean()
                    direction = direction * (math.sqrt(initial.numel()) / direction.norm())
                    initial = math.sqrt(.75)*initial + .5*direction
                path = args.out / variant / name
                path.mkdir(parents=True, exist_ok=True)
                paired = {**protocol, "variant": variant, "backend": args.backend, "settings": configuration,
                    "initial_source_sha256": hashlib.sha256(initial.cpu().numpy().tobytes()).hexdigest(),
                    "diagnostic_script_sha256": evidence["script_sha256"]}
                run.save_tensor(path / "inputs.pt", {**inputs, "initial": initial, "protocol": paired})
                results = {}
                methods = run.METHODS if variant in ("measurement", "random_mix") else ("DLR D-Flow",)
                for method in methods:
                    evaluator = run.Evaluator(model, operator, inputs["observed"], configuration,
                                              protocol["dlr_seed"], products)
                    stem = method.lower().replace("-", "_").replace(" ", "_")
                    results[method] = run.solve(evaluator, method, initial, inputs["truth"], inputs["clean"],
                                               path / f"{stem}.pt", paired, [False])
                    print(name, variant, method, results[method]["history"][-1], flush=True)
                fig = plot_reconstruction_comparison(inputs["truth"], inputs["clean"],
                    {method: (result["image"], result["measurement"]) for method, result in results.items()},
                    observation=inputs["observed"], include_errors=True,
                    measurement_log=isinstance(operator, MRI), residual_vmax=RESIDUAL_VMAX[protocol["job"]["problem"]],
                    title=f"{name}: {variant}, noiseless")
                fig.savefig(path / "comparison.png", dpi=120)
                run.plt.close(fig)
                del evaluator, results
        del products, model, inputs
        torch.cuda.empty_cache()
    if args.action == "diagnose":
        run.write_json(args.out / "gradients.json", dict(status="complete", rows=rows))


if __name__ == "__main__":
    main()
