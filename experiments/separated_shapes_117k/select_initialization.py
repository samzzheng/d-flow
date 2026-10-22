"""Cache objective-selected sources before any reconstruction or image scoring."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from dlr_dflow.api import load_flow_model
from experiments.ct_mri_sweep.operators import CT, MRI
from experiments.ct_mri_sweep.run import save_tensor, write_json
from sampler.ode_samplers import euler_flow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path,
        default=Path("outputs/separated_shapes_117k_noiseless_20261021/production"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--score", choices=("raw", "profiled_gain"), default="raw")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda:1")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.1, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    manifest = json.loads((args.baseline / "manifest.json").read_text())
    settings = manifest["settings"]
    rows = []
    for case in args.cases:
        inputs = torch.load(args.baseline / case / "inputs.pt", map_location=device, weights_only=True)
        protocol = inputs["protocol"]
        resolution = protocol["job"]["resolution"]
        folder = Path(settings["run_template"].format(resolution=resolution))
        assert hashlib.sha256((folder / "checkpoints/epoch_0500.pt").read_bytes()).hexdigest() == protocol["checkpoint_sha256"]
        model, _ = load_flow_model(resolution, 0, 500, str(device), folder)
        info = protocol["operator"]
        operator = (CT(resolution, protocol["job"]["level"], device) if protocol["job"]["problem"] == "ct"
                    else MRI(resolution, 4, info["mask_seed"], device, info["pattern"]))
        if isinstance(operator, MRI):
            assert torch.equal(inputs["mask"], operator.mask)
        z = inputs["initial"]
        variable = torch.zeros_like(z, requires_grad=True)
        pairing = (operator(variable).conj() * inputs["observed"]).real.sum()
        adjoint, = torch.autograd.grad(pairing, variable)
        random = torch.randn(z.shape, generator=torch.Generator().manual_seed(
            protocol["source_seed"] + 40000000)).to(device)
        for variant, direction in (("measurement", adjoint), ("random_mix", random)):
            directory = args.out / variant / case
            if (directory / "inputs.pt").exists():
                raise FileExistsError(f"Refusing to regenerate cached source: {directory}")
            directory.mkdir(parents=True, exist_ok=True)
            direction = direction - direction.mean()
            direction = direction * (math.sqrt(z.numel()) / direction.norm())
            candidates = []
            objectives = []
            candidate_metrics = []
            with torch.no_grad():
                for rho in (0., .25, .5, .75, 1.):
                    candidate = math.sqrt(1-rho*rho) * z + rho * direction
                    image = euler_flow(model, candidate, settings["flow_steps"])
                    predicted = operator(image)
                    residual = predicted - inputs["observed"]
                    raw = float(.5 * residual.abs().square().sum(dtype=torch.float64))
                    squared_norm = float(predicted.abs().square().sum(dtype=torch.float64))
                    gain = (max(0., float((predicted.conj()*inputs["observed"]).real.sum(dtype=torch.float64))/squared_norm)
                            if squared_norm else 0.)
                    profiled = float(.5 * (gain*predicted-inputs["observed"]).abs().square().sum(dtype=torch.float64))
                    objectives.append(raw if args.score == "raw" else profiled)
                    candidate_metrics.append(dict(rho=rho, objective=raw, profiled_objective=profiled, gain=gain))
                    candidates.append((rho, candidate, image))
            selected = min(range(len(candidates)), key=objectives.__getitem__)
            rho, initial, _ = candidates[selected]
            selection = dict(rule="Lowest selected measurement score among five source mixtures; no image scoring",
                score=args.score, gain_applied_to_reconstruction=False,
                direction=variant, rho=rho, candidates=candidate_metrics,
                forward_evaluations=len(candidates), source_sha256=hashlib.sha256(initial.cpu().numpy().tobytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            # Freeze the selected tensor before reading ground truth for diagnostics.
            save_tensor(directory / "inputs.pt", {**inputs, "initial": initial,
                "protocol": {**protocol, "initialization_selection": selection}})
            diagnostic_errors = [float((image-inputs["truth"]).norm()/inputs["truth"].norm())
                                 for _, _, image in candidates]
            row = dict(case=case, **selection, selected_image_error=diagnostic_errors[selected],
                       candidate_image_errors_diagnostic_only=diagnostic_errors)
            rows.append(row)
            write_json(directory / "selection.json", row)
            write_json(args.out / "selection_summary.json", dict(status="running", rows=rows))
            print(json.dumps(dict(case=case, variant=variant, rho=rho, objective=objectives[selected],
                                  image_error=diagnostic_errors[selected])), flush=True)
        del model, inputs
    write_json(args.out / "selection_summary.json", dict(status="complete", rows=rows))


if __name__ == "__main__":
    main()
