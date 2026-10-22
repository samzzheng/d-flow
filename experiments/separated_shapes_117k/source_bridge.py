"""Inspect native objective geometry between failed and successful saved sources."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from dlr_dflow.api import load_flow_model
from experiments.ct_mri_sweep.operators import MRI
from sampler.ode_samplers import euler_flow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(
        "outputs/separated_shapes_117k_diagnosis_20261022"))
    parser.add_argument("--baseline", type=Path, default=Path(
        "outputs/separated_shapes_117k_noiseless_20261021/production"))
    parser.add_argument("--out", type=Path, required=True)
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
    geometry = json.loads((args.root/"dataset_difficulty/geometry_results.json").read_text())["rows"]
    selected = [min(row["sample"] for row in geometry if row["resolution"] == 250
        and row["problem"] == "mri" and row["objects"] == count
        and row["method"] == "d_flow" and row["image_error_percent"] > 50)
        for count in range(1, 5)]
    cases = [(sample, args.baseline/f"r250_shapes_mri4/sample_{sample:03d}/d_flow.pt")
             for sample in selected]
    cases += [(sample, args.root/f"native_reconstruction/baseline/r250_shapes_mri4/sample_{sample:03d}/dlr_d_flow.pt")
              for sample in (0, 1)]
    folder = Path(settings["run_template"].format(resolution=250))
    checkpoint_hash = hashlib.sha256((folder/"checkpoints/epoch_0500.pt").read_bytes()).hexdigest()
    results = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for sample, failed_path in cases:
        directory = args.baseline/f"r250_shapes_mri4/sample_{sample:03d}"
        inputs = torch.load(directory/"inputs.pt", map_location=device, weights_only=True)
        failed = torch.load(failed_path, map_location=device, weights_only=True)
        success = torch.load(directory/"p_flow.pt", map_location=device, weights_only=True)
        protocol = inputs["protocol"]
        for key, value in protocol.items():
            assert failed["protocol"][key] == success["protocol"][key] == value
        assert protocol["checkpoint_sha256"] == checkpoint_hash
        assert success["history"][-1]["image_relative_l2"] < .1
        info = protocol["operator"]
        operator = MRI(250, 4, info["mask_seed"], device, info["pattern"])
        assert torch.equal(operator.mask, inputs["mask"])
        model, _ = load_flow_model(250, 0, 500, str(device), folder)
        with torch.no_grad():
            original_image = euler_flow(model, failed["source"], 5)
            endpoint_drift = float((original_image-failed["image"]).norm()/failed["image"].norm())
        model.double()
        model.time_proj.register_forward_pre_hook(lambda module, arguments: (arguments[0].double(),))
        start = failed["source"].double().detach().requires_grad_(True)
        target = success["source"].double().detach()
        observed, truth = inputs["observed"].to(torch.complex128), inputs["truth"].double()
        image = euler_flow(model, start, 5)
        loss = .5*(operator(image)-observed).abs().square().sum()
        terminal, = torch.autograd.grad(loss, image, retain_graph=True)
        exact, = torch.autograd.grad(loss, start)
        direction = target-start.detach()
        exact_norm, terminal_norm = float(exact.norm()), float(terminal.norm())
        assert exact_norm > 0 and terminal_norm > 0
        row = dict(sample=sample, failed_source=str(failed_path),
            source_sha256=hashlib.sha256(start.detach().cpu().numpy().tobytes()).hexdigest(),
            target_sha256=hashlib.sha256(target.cpu().numpy().tobytes()).hexdigest(),
            checkpoint_sha256=checkpoint_hash, saved_endpoint_relative_drift=endpoint_drift,
            fp64_endpoint_relative_change=float((image.detach()-original_image.double()).norm()/original_image.norm()),
            exact_gradient_norm=exact_norm, terminal_gradient_norm=terminal_norm,
            pullback_norm_ratio=exact_norm/terminal_norm,
            exact_proxy_cosine=float((exact*terminal).sum())/(exact_norm*terminal_norm),
            bridge_length=float(direction.norm()), initial_phi=float(loss.detach()),
            bridge_slope=float((exact*direction).sum()), path=[], probes=[], finite_differences=[])
        with torch.no_grad():
            for fraction in (0., 1e-6, 1e-5, 1e-4, .001, .01, .05, .1, .2, .4, .6, .8, 1.):
                value = euler_flow(model, start+fraction*direction, 5)
                residual = operator(value)-observed
                row["path"].append(dict(fraction=fraction, phi=float(.5*residual.abs().square().sum()),
                    image_error_percent=float(100*(value-truth).norm()/truth.norm()),
                    residual_percent=float(100*residual.norm()/observed.norm())))
            for label, vector in (("bridge", direction/direction.norm()),
                                  ("negative_exact", -exact/exact.norm()),
                                  ("negative_proxy", -terminal/terminal.norm())):
                slope = float((exact*vector).sum())
                for length in (1e-8, 1e-7, 1e-6):
                    plus = euler_flow(model, start+length*vector, 5)
                    minus = euler_flow(model, start-length*vector, 5)
                    phi_plus = float(.5*(operator(plus)-observed).abs().square().sum())
                    phi_minus = float(.5*(operator(minus)-observed).abs().square().sum())
                    central = (phi_plus-phi_minus)/(2*length)
                    represented = (start+length*vector).float()-start.float()
                    row["finite_differences"].append(dict(direction=label, length=length,
                        exact_slope=slope, central_slope=central,
                        slope_error_over_gradient_norm=abs(central-slope)/exact_norm,
                        actual_change=phi_plus-row["initial_phi"],
                        fp32_represented_displacement=float(represented.norm())))
                for length in (1e-5, .001, .1, 2.5):
                    value = euler_flow(model, start+length*vector, 5)
                    phi = float(.5*(operator(value)-observed).abs().square().sum())
                    row["probes"].append(dict(direction=label, length=length, slope=slope,
                        phi=phi, actual_change=phi-row["initial_phi"], linear_change=length*slope))
        results.append(row)
        args.out.write_text(json.dumps(dict(status="complete" if len(results) == len(cases) else "running",
            selection="Lowest-index D-Flow failure above 50% per object count, plus corrected native DLR cases 0/1. Retrospective selected-case diagnostics; no optimizer updates.",
            note="Straight source interpolation is not an optimization trajectory. No barrier here proves absence only on sampled points; a barrier does not prove all paths are blocked. FP64 promotes original FP32 model weights and time features.",
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            rows=results), indent=2)+"\n")
        print(json.dumps({key:row[key] for key in ("sample", "failed_source", "pullback_norm_ratio", "exact_proxy_cosine", "bridge_slope")}), flush=True)
        del model, inputs, failed, success, image, loss, exact, terminal
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
