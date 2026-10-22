"""Paired FP32/FP64 DLR reconstructions from the original cached Gaussian source."""
import argparse
import hashlib
import json
from pathlib import Path
import signal

import torch

from dlr_dflow.api import load_flow_model
from dlr_dflow.implementation import enriched, tangent_unet
from dlr_dflow.implementation.graphed_products import GraphedAProducts
from experiments.ct_mri_sweep import run
from experiments.ct_mri_sweep.operators import MRI
from experiments.ct_mri_sweep.validation import ChunkedAutodiffProducts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path(
        "outputs/separated_shapes_117k_noiseless_20261021/production"))
    parser.add_argument("--case", default="r250_shapes_mri4/sample_000")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda:1")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.15, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    interrupted = [False]

    def stop(signum: int, frame: object) -> None:
        interrupted[0] = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    inputs = torch.load(args.baseline/args.case/"inputs.pt", weights_only=True, map_location=device)
    protocol = inputs["protocol"]
    assert protocol["job"]["problem"] == "mri"
    assert torch.equal(inputs["clean"], inputs["observed"])
    settings = json.loads((args.baseline/"manifest.json").read_text())["settings"]
    assert settings["flow_steps"] == 5 and settings["dlr"]["rank"] == 32
    assert settings["dlr"]["trace_probes"] == settings["dlr"]["enrichment_probes"] == 0
    settings["execution"].update(dlr_fused_norm="native", dlr_fused_primal="native", memory_fraction=.15)
    resolution = protocol["job"]["resolution"]
    folder = Path(settings["run_template"].format(resolution=resolution))
    assert hashlib.sha256((folder/"checkpoints/epoch_0500.pt").read_bytes()).hexdigest() == protocol["checkpoint_sha256"]
    enriched.SPECIALIZE_ZERO_ENRICHMENT = enriched.SKIP_UNTRUNCATED_SVD = True
    tangent_unet.GN_STATS_MODE = tangent_unet.GN_ACTION_BACKEND = tangent_unet.GN_PRIMAL_BACKEND = "native"
    info = protocol["operator"]
    operator = MRI(resolution, 4, info["mask_seed"], device, info["pattern"])
    assert torch.equal(operator.mask, inputs["mask"])
    for dtype in (torch.float32, torch.float64):
        if interrupted[0]:
            break
        label = str(dtype).split(".")[-1]
        path = args.out/label
        path.mkdir(parents=True, exist_ok=True)
        model, _ = load_flow_model(resolution, 0, 500, str(device), folder)
        model.to(dtype=dtype)
        if dtype == torch.float64:
            model.time_proj.register_forward_pre_hook(lambda module, arguments: (arguments[0].double(),))
        initial = inputs["initial"].to(dtype)
        assert torch.equal(initial.float(), inputs["initial"])
        observed = inputs["observed"].to(torch.complex128 if dtype == torch.float64 else torch.complex64)
        products = GraphedAProducts(model, initial, share_linearization=True, full_step_graphs=True,
            max_jvp_columns=20, max_vjp_columns=16, max_step_jvp_columns=6, max_step_vjp_columns=4)
        evaluator = run.Evaluator(model, operator, observed, settings, protocol["dlr_seed"], products)
        native = run.Evaluator(model, operator, observed, settings, protocol["dlr_seed"], ChunkedAutodiffProducts(model, columns=4))
        reference = native.gradient(initial, "DLR D-Flow")
        estimate = evaluator.gradient(initial, "DLR D-Flow")
        discrepancy = float((reference-estimate).norm()/reference.norm())
        assert discrepancy < (1e-4 if dtype == torch.float32 else 1e-9)
        torch.testing.assert_close(evaluator.image, native.image, rtol=0, atol=0)
        del native
        evaluator.previous_v = None
        experiment = dict(settings=settings, precision=label, initial_source_sha256=
            hashlib.sha256(inputs["initial"].cpu().numpy().tobytes()).hexdigest(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            note="Original source values promoted exactly; identical Rademacher seed, rank32, five Euler steps and LBFGS/geometric Armijo. Native normalization and products calibrated independently. Fresh optimization, not a restart from failure.")
        run.write_json(path/"initial_validation.json", dict(relative_gradient_backend_error=discrepancy,
            initial_phi=evaluator.value, source_hash=experiment["initial_source_sha256"]))
        paired = {**protocol, "experiment": experiment}
        result = run.solve(evaluator, "DLR D-Flow", initial, inputs["truth"].to(dtype), observed,
            path/"dlr_d_flow.pt", paired, interrupted)
        print(label, result["status"], result["history"][-1], flush=True)
        del model, products, evaluator, result
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
