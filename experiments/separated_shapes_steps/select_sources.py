"""Screen a small source bank using only the noiseless measurement objective."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import time

import torch

from dlr_dflow.api import load_flow_model
from experiments.ct_mri_sweep.operators import CT, MRI
from experiments.ct_mri_sweep.run import metrics, save_tensor, write_json
from experiments.separated_shapes_steps.run import CASES
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
    out = Path("outputs/separated_shapes_steps_20261019/source_selection")
    records = []
    for resolution, problem, sample in CASES:
        name = f"r{resolution}_shapes_{problem}/sample_{sample:03d}"
        case = Path("outputs/separated_shapes_noiseless_20261016/production") / name
        inputs = torch.load(case / "inputs.pt", map_location=device, weights_only=True)
        protocol = inputs["protocol"]
        folder = Path(f"outputs/fm_unet/separated-shapes-{resolution}-noattn-ch16-b8-e200")
        assert hashlib.sha256((folder / "checkpoints/epoch_0200.pt").read_bytes()).hexdigest() == protocol["checkpoint_sha256"]
        model, _ = load_flow_model(resolution, 0, 200, str(device), folder)
        info = protocol["operator"]
        operator = CT(resolution, 10, device) if problem == "ct10" else MRI(resolution, 4, info["mask_seed"], device, info["pattern"])
        if problem == "mri4":
            assert torch.equal(operator.mask, inputs["mask"])
        seeds = [protocol["source_seed"], *range(protocol["source_seed"]+40000000, protocol["source_seed"]+40000031)]
        candidates = []
        best = None
        started = time.monotonic()
        with torch.no_grad():
            for seed in seeds:
                source = torch.randn(inputs["initial"].shape, generator=torch.Generator().manual_seed(seed)).to(device)
                if seed == seeds[0]:
                    assert torch.equal(source, inputs["initial"])
                image = euler_flow(model, source, 5)
                measurement = operator(image)
                loss = float(.5 * (measurement-inputs["observed"]).abs().square().sum(dtype=torch.float64))
                row = dict(seed=seed, objective=loss,
                           **metrics(image, measurement, inputs["truth"], inputs["clean"], inputs["observed"]))
                candidates.append(row)
                if best is None or loss < best["objective"]:
                    best = dict(source=source.clone(), image=image.clone(), **row)
        assert best is not None and best["objective"] <= candidates[0]["objective"]
        record = dict(case=name, steps=5, candidates=candidates, selected_seed=best["seed"],
                      objective_ratio=best["objective"]/candidates[0]["objective"],
                      selected_image_error_percent=100*best["image_relative_l2"],
                      selection_seconds=time.monotonic()-started,
                      note="Selection uses measurement objective only, never image error. Original source included. Shared-GPU timing.")
        (out / name).mkdir(parents=True, exist_ok=True)
        save_tensor(out / name / "selected.pt", dict(**best, protocol=protocol, selection=record))
        records.append(record)
        write_json(out / "summary.json", dict(status="running", cases=records))
        print(name, "objective ratio",round(record["objective_ratio"],4),"selected image error %",round(record["selected_image_error_percent"],2),flush=True)
        del inputs, model, source, image, measurement, best
        torch.cuda.empty_cache()
    write_json(out / "summary.json", dict(status="complete", cases=records))


if __name__ == "__main__":
    main()
