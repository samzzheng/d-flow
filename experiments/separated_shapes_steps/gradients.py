"""Compare exact and DLR gradients at identical separated-shapes sources."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from dlr_dflow.api import approx_source_gradient, load_flow_model, sensitivity_factors
from experiments.ct_mri_sweep.operators import CT, MRI
from experiments.ct_mri_sweep.run import write_json
from experiments.ct_mri_sweep.validation import ChunkedEagerProducts
from p_flow import endpoint_gradient
from sampler.ode_samplers import euler_flow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.15, device)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    out = Path("outputs/separated_shapes_steps_20261019/gradients")
    rows = []
    for resolution, problem, sample in ((64, "mri4", 1), (128, "ct10", 0)):
        name = f"r{resolution}_shapes_{problem}/sample_{sample:03d}"
        case = Path("outputs/separated_shapes_noiseless_20261016/production") / name
        inputs = torch.load(case / "inputs.pt", weights_only=True, map_location=device)
        saved = torch.load(case / "d_flow.pt", weights_only=True, map_location="cpu", mmap=True)
        protocol = inputs["protocol"]
        folder = Path(f"outputs/fm_unet/separated-shapes-{resolution}-noattn-ch16-b8-e200")
        assert hashlib.sha256((folder / "checkpoints/epoch_0200.pt").read_bytes()).hexdigest() == protocol["checkpoint_sha256"]
        model, _ = load_flow_model(resolution, 0, 200, str(device), folder)
        info = protocol["operator"]
        operator = CT(resolution, 10, device) if problem == "ct10" else MRI(resolution, 4, info["mask_seed"], device, info["pattern"])
        if problem == "mri4":
            assert torch.equal(operator.mask, inputs["mask"])
        for stage, source in (("initial", inputs["initial"]), ("failed_dflow_final", saved["source"].to(device))):
            for steps in (5, 10, 20):
                variable = source.detach().requires_grad_(True)
                image = euler_flow(model, variable, steps)
                loss = .5 * (operator(image) - inputs["observed"]).abs().square().sum(dtype=torch.float64)
                exact, = torch.autograd.grad(loss, variable)
                image = image.detach()
                _, gx = endpoint_gradient(image, operator, inputs["observed"])
                exact_norm = float(exact.norm())
                direction = exact / exact_norm
                finite_differences = []
                with torch.no_grad():
                    for epsilon in (.01, .003, .001):
                        plus = operator(euler_flow(model, source + epsilon * direction, steps)) - inputs["observed"]
                        minus = operator(euler_flow(model, source - epsilon * direction, steps)) - inputs["observed"]
                        numerical = float((plus.abs().square().sum(dtype=torch.float64) - minus.abs().square().sum(dtype=torch.float64)) / (4 * epsilon))
                        finite_differences.append(dict(epsilon=epsilon, numerical_derivative=numerical,
                                                       relative_error=abs(numerical-exact_norm)/exact_norm))
                    for rank in (32, 128):
                        endpoint, factors, alpha = sensitivity_factors(model, source, steps, rank=rank,
                            trace_probes=0, enrichment_probes=0, initialization_oversampling=8,
                            zero_terminal_baseline=True, speed_profile="fast", seed=protocol["dlr_seed"],
                            products=ChunkedEagerProducts(model))
                        assert alpha == 0
                        estimate = approx_source_gradient(factors, gx, alpha)
                        _, _, v = factors
                        projection = (v @ (v.T @ exact.reshape(-1, 1))).reshape_as(exact)
                        error = float((estimate-exact).norm())
                        coefficient = float((estimate-projection).norm())
                        subspace = float((projection-exact).norm())
                        row = dict(case=name, source=stage, steps=steps, rank=rank,
                            source_sha256=hashlib.sha256(source.cpu().numpy().tobytes()).hexdigest(),
                            checkpoint_sha256=protocol["checkpoint_sha256"], dlr_seed=protocol["dlr_seed"],
                            exact_gradient_norm=exact_norm, endpoint_gradient_norm=float(gx.norm()),
                            relative_gradient_error=error/exact_norm,
                            norm_ratio=float(estimate.norm())/exact_norm,
                            cosine=float(torch.nn.functional.cosine_similarity(estimate.flatten(), exact.flatten(), dim=0)),
                            coefficient_error_over_exact=coefficient/exact_norm,
                            subspace_error_over_exact=subspace/exact_norm,
                            decomposition_error_over_exact_squared=abs(error**2-coefficient**2-subspace**2)/exact_norm**2,
                            endpoint_error_over_gt=float((endpoint-image).norm()/inputs["truth"].norm()),
                            exact_finite_differences=finite_differences,
                            note="Same native endpoint and gx for both gradients; fresh DLR sketch, no previous-V reuse in this fixed-point diagnostic.")
                        rows.append(row)
                        write_json(out / "results.json", dict(status="running", rows=rows))
                        print(name, stage, steps, rank, "relative_error", round(row["relative_gradient_error"], 4),
                              "cosine", round(row["cosine"], 4), "subspace", round(row["subspace_error_over_exact"], 4), flush=True)
                del image, exact, loss, gx, factors, estimate, endpoint, projection, variable, direction
                torch.cuda.empty_cache()
        del model, inputs, saved
        torch.cuda.empty_cache()
    write_json(out / "results.json", dict(status="complete", rows=rows))


if __name__ == "__main__":
    main()
