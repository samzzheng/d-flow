"""Paired fixed-source, fixed-rank audits before expensive reconstruction trials."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch

from dlr_dflow.api import approx_source_gradient, load_flow_model, sensitivity_factors
from dlr_dflow.implementation import enriched, tangent_unet
from experiments.ct_mri_sweep import run
from experiments.ct_mri_sweep.operators import CT, MRI
from experiments.ct_mri_sweep.validation import ChunkedAutodiffProducts
from experiments.separated_shapes_117k.bounded_products import BoundedGraphedProducts
from experiments.separated_shapes_117k.gradient_audit import scores
from sampler.ode_samplers import euler_flow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path(
        "outputs/separated_shapes_117k_noiseless_20261021/production/manifest.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=[32, 256, 1024, 2048])
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda:1")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.30, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    inputs = torch.load(args.inputs, weights_only=True, map_location=device)
    source = inputs["initial"]
    if args.source:
        saved = torch.load(args.source, weights_only=True, map_location=device)
        assert saved["protocol"]["checkpoint_sha256"] == inputs["protocol"]["checkpoint_sha256"]
        assert saved["protocol"]["operator"] == inputs["protocol"]["operator"]
        source = saved["source"].detach().clone()
        del saved
    settings = json.loads(args.manifest.read_text())["settings"]
    protocol = inputs["protocol"]
    resolution = protocol["job"]["resolution"]
    folder = Path(settings["run_template"].format(resolution=resolution))
    assert hashlib.sha256((folder/"checkpoints/epoch_0500.pt").read_bytes()).hexdigest() == protocol["checkpoint_sha256"]
    assert settings["flow_steps"] == 5
    assert settings["dlr"]["trace_probes"] == settings["dlr"]["enrichment_probes"] == 0
    assert 0 < min(args.ranks) <= max(args.ranks) < source.numel()
    model, _ = load_flow_model(resolution, 0, 500, str(device), folder)
    info = protocol["operator"]
    operator = (CT(resolution, protocol["job"]["level"], device) if protocol["job"]["problem"] == "ct"
                else MRI(resolution, 4, info["mask_seed"], device, info["pattern"]))
    if isinstance(operator, MRI):
        assert torch.equal(inputs["mask"], operator.mask)
    variable = source.detach().requires_grad_(True)
    image = euler_flow(model, variable, settings["flow_steps"])
    loss = .5*(operator(image)-inputs["observed"]).abs().square().sum(dtype=torch.float64)
    exact, = torch.autograd.grad(loss, variable)
    image = image.detach()
    exact = exact.reshape(-1, 1)
    _, terminal = run.endpoint_gradient(image, operator, inputs["observed"])
    enriched.SPECIALIZE_ZERO_ENRICHMENT = enriched.SKIP_UNTRUNCATED_SVD = True
    tangent_unet.GN_STATS_MODE = tangent_unet.GN_ACTION_BACKEND = tangent_unet.GN_PRIMAL_BACKEND = "native"
    products = BoundedGraphedProducts(model, source, cache_size=4, share_linearization=True,
        full_step_graphs=False, max_jvp_columns=20, max_vjp_columns=16)
    # One nested Rademacher sketch: changes in rank do not change its shared columns.
    sketch = torch.randint(0, 2, (source.numel(), max(args.ranks)+8), device=device,
        generator=torch.Generator(device=device).manual_seed(protocol["dlr_seed"]), dtype=torch.int8).float()
    sketch.mul_(2).sub_(1)
    report: dict[str, Any] = dict(status="running", inputs=str(args.inputs), source=str(args.source),
        source_sha256=hashlib.sha256(source.cpu().numpy().tobytes()).hexdigest(),
        checkpoint_sha256=protocol["checkpoint_sha256"], settings=settings, ranks=args.ranks,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sketch_sha256=hashlib.sha256(sketch.cpu().numpy().tobytes()).hexdigest(),
        objective=float(loss.detach()), exact_gradient_norm=float(exact.norm()),
        image_error=float((image-inputs["truth"]).norm()/inputs["truth"].norm()), rows=[],
        note="Fresh nested Rademacher sketches, fixed rank, native products, FP32 with TF32 off. Exact gradient differentiates the same five-step FP32 map; it is not a high-precision reference. Single cold refresh times under shared GPU load include graph capture; not speed benchmarks.")
    run.write_json(args.out, report)
    for rank in args.ranks:
        options = {**settings["dlr"], "rank": rank, "adaptive_rank": False}
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()

        def progress(step: int, alpha: float, factors: tuple[torch.Tensor, ...]) -> None:
            print(f"rank={rank} flow_step={step} seconds={time.perf_counter()-started:.2f}", flush=True)

        try:
            endpoint, factors, alpha = sensitivity_factors(model, source, settings["flow_steps"],
                **options, seed=protocol["dlr_seed"], products=products,
                seed_basis=sketch[:, :rank+8], factor_callback=progress)
            estimate = approx_source_gradient(factors, terminal, alpha).reshape(-1, 1)
            torch.cuda.synchronize(device)
            row: dict[str, Any] = dict(rank=rank, status="complete",
                cold_refresh_seconds=time.perf_counter()-started,
                peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,
                terminal_rank=factors[1].shape[0], **scores(estimate, exact, factors[2], terminal))
            torch.testing.assert_close(endpoint, image, rtol=0, atol=0)
            assert float(alpha) == 0 and row["terminal_rank"] == rank
            assert torch.isfinite(estimate).all()
            decomposition = abs(row["relative_error"]**2-row["subspace_relative_error"]**2
                                -row["coefficient_relative_error"]**2)
            assert decomposition < 1e-4*max(1, row["relative_error"]**2)
            row["orthogonal_decomposition_defect"] = decomposition
            if rank == min(args.ranks):
                native_image, native_factors, native_alpha = sensitivity_factors(model, source,
                    settings["flow_steps"], **options, seed=protocol["dlr_seed"],
                    products=ChunkedAutodiffProducts(model, columns=4), seed_basis=sketch[:, :rank+8])
                reference = approx_source_gradient(native_factors, terminal, native_alpha).reshape_as(estimate)
                row["native_backend_relative_error"] = float((estimate-reference).norm()/reference.norm())
                row["native_backend_validated"] = row["native_backend_relative_error"] < 1e-3
                row["native_gradient_scores"] = scores(reference, exact, native_factors[2], terminal)
                report["backend_calibration_passed"] = row["native_backend_validated"]
                torch.testing.assert_close(native_image, image, rtol=0, atol=0)
                del native_image, native_factors, reference
            del endpoint, factors, estimate
        except torch.OutOfMemoryError as error:
            row = dict(rank=rank, status="out_of_memory", message=str(error),
                elapsed_seconds=time.perf_counter()-started)
            report["rows"].append(row)
            report["status"] = "incomplete_memory_limit"
            run.write_json(args.out, report)
            print(json.dumps(row), flush=True)
            return
        report["rows"].append(row)
        run.write_json(args.out, report)
        print(json.dumps(row), flush=True)
    report["status"] = "complete"
    run.write_json(args.out, report)


if __name__ == "__main__":
    main()
