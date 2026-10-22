"""Compare native FP32/FP64 derivatives at identical failed and successful sources."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch

from dlr_dflow.api import load_flow_model
from experiments.ct_mri_sweep.operators import MRI
from sampler.ode_samplers import euler_flow


def group_norm64(module: torch.nn.GroupNorm, value: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.group_norm(value.double(), module.num_groups,
        module.weight.double() if module.weight is not None else None,
        module.bias.double() if module.bias is not None else None, module.eps).to(value.dtype)


def mixed_euler(model: torch.nn.Module, source: torch.Tensor, steps: int) -> torch.Tensor:
    value = source
    for index in range(steps):
        time = torch.full((value.shape[0],), index/steps, device=value.device, dtype=torch.float32)
        value = value+model(value.float(), time).double()/steps
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridges", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=Path(
        "outputs/separated_shapes_117k_noiseless_20261021/production"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mixed", action="store_true", help="Include isolated higher-precision normalization and Euler accumulation")
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device("cuda:1")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.1, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    settings = json.loads((args.baseline/"manifest.json").read_text())["settings"]
    assert settings["flow_steps"] == 5
    folder = Path(settings["run_template"].format(resolution=250))
    checkpoint_hash = hashlib.sha256((folder/"checkpoints/epoch_0500.pt").read_bytes()).hexdigest()
    rows = []
    for bridge in json.loads(args.bridges.read_text())["rows"]:
        directory = args.baseline/f"r250_shapes_mri4/sample_{bridge['sample']:03d}"
        inputs = torch.load(directory/"inputs.pt", weights_only=True, map_location=device)
        protocol = inputs["protocol"]
        assert checkpoint_hash == protocol["checkpoint_sha256"]
        info = protocol["operator"]
        operator = MRI(250, 4, info["mask_seed"], device, info["pattern"])
        assert torch.equal(operator.mask, inputs["mask"])
        for label, path in (("failed", Path(bridge["failed_source"])), ("successful", directory/"p_flow.pt")):
            if any(row["source_file"] == str(path) for row in rows):
                continue
            saved = torch.load(path, weights_only=True, map_location=device)
            assert all(saved["protocol"][key] == value for key, value in protocol.items())
            values = {}
            modes = ["float32", "norm64", "euler64", "norm_euler64", "float64"] if args.mixed else ["float32", "float64"]
            for mode in modes:
                model, _ = load_flow_model(250, 0, 500, str(device), folder)
                precision = torch.float64 if mode == "float64" else torch.float32
                if precision == torch.float64:
                    model.double()
                    model.time_proj.register_forward_pre_hook(lambda module, arguments: (arguments[0].double(),))
                use_double_state = mode in ("float64", "euler64", "norm_euler64")
                source = saved["source"].to(torch.float64 if use_double_state else torch.float32).detach().requires_grad_(True)
                context = patch.object(torch.nn.GroupNorm, "forward", group_norm64) if mode in ("norm64", "norm_euler64") else nullcontext()
                with context:
                    image = (mixed_euler if mode in ("euler64", "norm_euler64") else euler_flow)(model, source, 5)
                    observed = inputs["observed"].to(torch.complex128 if use_double_state else torch.complex64)
                    phi = .5*(operator(image)-observed).abs().square().sum(dtype=torch.float64)
                    gradient, = torch.autograd.grad(phi, source)
                values[mode] = (float(phi.detach()), image.detach().double(), gradient.detach().double())
                del model
            phi32, image32, gradient32 = values["float32"]
            phi64, image64, gradient64 = values["float64"]
            if label == "failed":
                assert abs(float(gradient64.norm())/bridge["exact_gradient_norm"]-1) < 1e-8
                assert hashlib.sha256(source.detach().cpu().numpy().tobytes()).hexdigest() == bridge["source_sha256"]
            rows.append(dict(sample=bridge["sample"], label=label, source_file=str(path),
                source_sha256=hashlib.sha256(source.detach().cpu().numpy().tobytes()).hexdigest(),
                checkpoint_sha256=checkpoint_hash, phi32=phi32, phi64=phi64,
                image_error_percent=float(100*(image64-inputs["truth"].double()).norm()/inputs["truth"].double().norm()),
                endpoint_relative_difference=float((image32-image64).norm()/image64.norm()),
                gradient_relative_difference=float((gradient32-gradient64).norm()/gradient64.norm()),
                gradient_cosine=float((gradient32*gradient64).sum()/(gradient32.norm()*gradient64.norm())),
                gradient32_norm=float(gradient32.norm()), gradient64_norm=float(gradient64.norm()),
                modes={name:dict(phi=value[0],
                    endpoint_relative_difference=float((value[1]-image64).norm()/image64.norm()),
                    gradient_relative_difference=float((value[2]-gradient64).norm()/gradient64.norm()),
                    gradient_cosine=float((value[2]*gradient64).sum()/(value[2].norm()*gradient64.norm())))
                    for name,value in values.items()}))
            print(json.dumps(rows[-1]), flush=True)
            del source, image, phi, gradient, values
            torch.cuda.empty_cache()
    args.out.write_text(json.dumps(dict(status="complete", rows=rows,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note="Native derivative comparison of FP32, optional mixed precision, and promoted FP64 maps at the exact same sources. TF32 disabled; no optimizer updates. All scalar reductions are FP64. Mixed Euler accumulation also promotes the MRI FFT. This does not compare DLR approximation error."), indent=2)+"\n")


if __name__ == "__main__":
    main()
