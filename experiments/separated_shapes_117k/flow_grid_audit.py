"""Reevaluate saved sources on Euler grids without performing optimization."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from dlr_dflow.api import load_flow_model
from experiments.ct_mri_sweep.operators import CT, MRI
from experiments.ct_mri_sweep.run import metrics, write_json
from sampler.ode_samplers import euler_flow


def euler_grid(model: torch.nn.Module, source: torch.Tensor,
               times: tuple[float, ...]) -> torch.Tensor:
    image = source
    for start, end in zip(times[:-1], times[1:]):
        time = image.new_full((image.shape[0],), start)
        image = image + (end-start) * model(image, time)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folders", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
        default=Path("outputs/separated_shapes_117k_noiseless_20261021/production/manifest.json"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda:1")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.1, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    settings = json.loads(args.manifest.read_text())["settings"]
    grids = {f"uniform_{count}": tuple(index/count for index in range(count+1)) for count in (5, 10, 20, 100)}
    grids.update(end_clustered_5=(0., .4, .7, .85, .95, 1.),
                 moderate_5=(0., .3, .55, .75, .9, 1.))
    rows = []
    for folder in args.folders:
        inputs = torch.load(folder / "inputs.pt", map_location=device, weights_only=True)
        protocol = inputs["protocol"]
        resolution = protocol["job"]["resolution"]
        model_folder = Path(settings["run_template"].format(resolution=resolution))
        assert hashlib.sha256((model_folder / "checkpoints/epoch_0500.pt").read_bytes()).hexdigest() == protocol["checkpoint_sha256"]
        model, _ = load_flow_model(resolution, 0, 500, str(device), model_folder)
        info = protocol["operator"]
        operator = (CT(resolution, protocol["job"]["level"], device) if protocol["job"]["problem"] == "ct"
                    else MRI(resolution, 4, info["mask_seed"], device, info["pattern"]))
        sources = {"initial": inputs["initial"]}
        for name in ("d_flow", "p_flow", "dlr_d_flow"):
            path = folder / f"{name}.pt"
            if path.exists():
                saved = torch.load(path, map_location=device, weights_only=True)
                if saved["status"] in {"stalled", "step_limit", "objective_plateau"}:
                    sources[name] = saved["source"]
        with torch.no_grad():
            for name, source in sources.items():
                reference = euler_flow(model, source, 5)
                for label, grid in grids.items():
                    image = euler_grid(model, source, grid)
                    if label == "uniform_5":
                        torch.testing.assert_close(image, reference, atol=0, rtol=0)
                    row = dict(folder=str(folder), source=name, grid=label, time_points=grid,
                        source_sha256=hashlib.sha256(source.cpu().numpy().tobytes()).hexdigest(),
                        **metrics(image, operator(image), inputs["truth"], inputs["clean"], inputs["observed"]))
                    rows.append(row)
                    print(json.dumps(dict(source=name, grid=label, image_error=row["image_relative_l2"])), flush=True)
                    write_json(args.out, dict(status="running", rows=rows,
                        note="Fixed-source reevaluation, not a reconstruction comparison on changed grids"))
        del model, inputs, sources
    write_json(args.out, dict(status="complete", rows=rows,
        note="Fixed-source reevaluation, not a reconstruction comparison on changed grids"))


if __name__ == "__main__":
    main()
