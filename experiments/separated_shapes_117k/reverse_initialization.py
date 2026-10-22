"""Cache measurement-only source candidates obtained by backward Euler flow."""
from __future__ import annotations

import argparse
import hashlib
import json
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
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda:1")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.1, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    settings = json.loads((args.baseline / "manifest.json").read_text())["settings"]
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
        variable = torch.zeros_like(inputs["initial"], requires_grad=True)
        pairing = (operator(variable).conj() * inputs["observed"]).real.sum()
        adjoint, = torch.autograd.grad(pairing, variable)
        with torch.no_grad():
            predicted = operator(adjoint)
            gain = ((predicted.conj()*inputs["observed"]).real.sum(dtype=torch.float64)
                    / predicted.abs().square().sum(dtype=torch.float64))
            target = adjoint * gain.to(adjoint.dtype)
            random = torch.randn(target.shape, generator=torch.Generator().manual_seed(
                protocol["source_seed"] + 40000000)).to(device)
            random -= random.mean()
            random = random * ((target-target.mean()).norm()/random.norm()) + target.mean()
            for variant, image_start in (("measurement", target), ("random_mix", random)):
                directory = args.out / variant / case
                if (directory / "inputs.pt").exists():
                    raise FileExistsError(f"Refusing to replace cached source: {directory}")
                directory.mkdir(parents=True, exist_ok=True)
                candidates = {"original": inputs["initial"]}
                for steps in (5, 20, 100):
                    source = image_start.clone()
                    for index in range(steps):
                        time = source.new_full((source.shape[0],), 1-index/steps)
                        source = source - model(source, time)/steps
                    if not torch.isfinite(source).all():
                        raise FloatingPointError("Nonfinite backward-flow source")
                    candidates[f"reverse_{steps}"] = source
                images = {name: euler_flow(model, source, settings["flow_steps"])
                          for name, source in candidates.items()}
                objectives = {name: float(.5*(operator(image)-inputs["observed"]).abs().square().sum(dtype=torch.float64))
                              for name, image in images.items()}
                selected = min(objectives, key=objectives.__getitem__)
                initial = candidates[selected]
                selection = dict(rule="Minimum canonical measurement loss; no ground-truth selection",
                    construction="Backward Euler from scaled adjoint or matched random image",
                    selected=selected, objectives=objectives, adjoint_gain=float(gain),
                    source_sha256=hashlib.sha256(initial.cpu().numpy().tobytes()).hexdigest(),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    initialization_model_evaluations=125+4*settings["flow_steps"],
                    note="Backward Euler is not the exact inverse of the discrete forward Euler map")
                save_tensor(directory / "inputs.pt", {**inputs, "initial": initial,
                    "protocol": {**protocol, "initialization_selection": selection}})
                save_tensor(directory / "candidates.pt", candidates)
                row = dict(case=case, variant=variant, **selection,
                    diagnostic_image_errors={name: float((image-inputs["truth"]).norm()/inputs["truth"].norm())
                                             for name, image in images.items()})
                rows.append(row)
                write_json(directory / "selection.json", row)
                write_json(args.out / "summary.json", dict(status="running", rows=rows))
                print(json.dumps(row), flush=True)
        del model, inputs
    write_json(args.out / "summary.json", dict(status="complete", rows=rows))


if __name__ == "__main__":
    main()
