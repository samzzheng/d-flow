"""Sequential paired fixed-rank reconstructions using the existing optimizer."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import signal

import torch

from dlr_dflow.api import load_flow_model
from dlr_dflow.implementation import enriched, tangent_unet
from experiments.ct_mri_sweep import run
from experiments.ct_mri_sweep.operators import CT, MRI
from experiments.separated_shapes_117k.bounded_products import BoundedGraphedProducts
from experiments.separated_shapes_117k.spectral_weighting import SpectralEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path(
        "outputs/separated_shapes_117k_noiseless_20261021/production"))
    parser.add_argument("--case", default="r250_shapes_mri4/sample_001")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ranks", nargs="+", type=int, default=[32, 1024, 2048])
    parser.add_argument("--memory-fraction", type=float, default=.30)
    parser.add_argument("--cudnn-deterministic", action="store_true")
    parser.add_argument("--repeat-initial-gradient", type=int, default=0)
    parser.add_argument("--diagnostics-only", action="store_true")
    parser.add_argument("--terminal-weighting", choices=("unchanged", "median_cap", "norm_matched_cap"),
                        default="unchanged")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda:1")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = args.cudnn_deterministic
    interrupted = [False]

    def stop(signum: int, frame: object) -> None:
        interrupted[0] = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    inputs = torch.load(args.baseline/args.case/"inputs.pt", weights_only=True, map_location=device)
    protocol = inputs["protocol"]
    settings = json.loads((args.baseline/"manifest.json").read_text())["settings"]
    resolution = protocol["job"]["resolution"]
    folder = Path(settings["run_template"].format(resolution=resolution))
    assert hashlib.sha256((folder/"checkpoints/epoch_0500.pt").read_bytes()).hexdigest() == protocol["checkpoint_sha256"]
    assert settings["flow_steps"] == 5 and torch.equal(inputs["clean"], inputs["observed"])
    assert settings["dlr"]["trace_probes"] == settings["dlr"]["enrichment_probes"] == 0
    model, _ = load_flow_model(resolution, 0, 500, str(device), folder)
    info = protocol["operator"]
    operator = (CT(resolution, protocol["job"]["level"], device) if protocol["job"]["problem"] == "ct"
                else MRI(resolution, 4, info["mask_seed"], device, info["pattern"]))
    if isinstance(operator, MRI):
        assert torch.equal(inputs["mask"], operator.mask)
    enriched.SPECIALIZE_ZERO_ENRICHMENT = enriched.SKIP_UNTRUNCATED_SVD = True
    tangent_unet.GN_STATS_MODE = tangent_unet.GN_ACTION_BACKEND = tangent_unet.GN_PRIMAL_BACKEND = "native"
    initial = inputs["initial"]
    sketch = torch.randint(0, 2, (initial.numel(), max(args.ranks)+8), device=device,
        generator=torch.Generator(device=device).manual_seed(protocol["dlr_seed"]), dtype=torch.int8).float()
    sketch.mul_(2).sub_(1)
    for rank in args.ranks:
        if interrupted[0]:
            break
        configuration = deepcopy(settings)
        configuration["dlr"].update(rank=rank, adaptive_rank=False)
        configuration["terminal_weighting"] = args.terminal_weighting
        configuration["execution"].update(memory_fraction=args.memory_fraction,
            cudnn_deterministic=args.cudnn_deterministic, dlr_full_step_graphs=False,
            dlr_jvp_columns=20, dlr_vjp_columns=16, dlr_fused_norm="native", dlr_fused_primal="native")
        products = BoundedGraphedProducts(model, initial, cache_size=4, share_linearization=True,
            full_step_graphs=False, max_jvp_columns=20, max_vjp_columns=16)
        experiment = dict(settings=configuration,
            calibration_gradient_evaluations=args.repeat_initial_gradient,
            source_sha256=hashlib.sha256(initial.cpu().numpy().tobytes()).hexdigest(),
            sketch_sha256=hashlib.sha256(sketch.cpu().numpy().tobytes()).hexdigest(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            weighting_script_sha256=hashlib.sha256(Path(__file__).with_name("spectral_weighting.py").read_bytes()).hexdigest(),
            note="Paired original sources and nested random initial sketches; sequential ranks on one sample. Subsequent previous-V reuse and LBFGS/geometric Armijo unchanged; max1000 attempts with original safeguards. FP32/native products, TF32 off.")
        path = args.out/args.case/f"rank{rank}"
        path.mkdir(parents=True, exist_ok=True)
        run.write_json(path/"config.json", experiment)
        if args.repeat_initial_gradient:
            diagnostic = SpectralEvaluator(model, operator, inputs["observed"], configuration, protocol["dlr_seed"], products)
            repeats = []
            for repeat in range(args.repeat_initial_gradient):
                diagnostic.previous_v = sketch[:, :rank+8]
                gradient = diagnostic.gradient(initial, "DLR D-Flow").detach().clone()
                if repeat == 0:
                    first_gradient = gradient
                    first_image = diagnostic.image.clone()
                    run.save_tensor(path/"first_gradient.pt", dict(source=initial, gradient=gradient, image=first_image))
                repeats.append(dict(repeat=repeat,
                    gradient_bitwise_equal=torch.equal(gradient, first_gradient),
                    endpoint_bitwise_equal=torch.equal(diagnostic.image, first_image),
                    relative_gradient_difference=float((gradient.double()-first_gradient.double()).norm()/first_gradient.double().norm())))
            run.write_json(path/"gradient_repeatability.json", dict(rows=repeats,
                source_sha256=experiment["source_sha256"], sketch_sha256=experiment["sketch_sha256"],
                cudnn_deterministic=args.cudnn_deterministic))
            print("Initial gradient repeatability", repeats, flush=True)
            del diagnostic, gradient, first_gradient, first_image
        if args.diagnostics_only:
            del products
            continue
        evaluator = SpectralEvaluator(model, operator, inputs["observed"], configuration, protocol["dlr_seed"], products)
        evaluator.previous_v = sketch[:, :rank+8]
        result = run.solve(evaluator, "DLR D-Flow", initial, inputs["truth"], inputs["observed"],
            path/"dlr_d_flow.pt", {**protocol, "experiment": experiment}, interrupted)
        run.write_json(path/"terminal_weighting.json", dict(variant=args.terminal_weighting,
            rows=evaluator.weighting_history))
        print(rank, result["status"], result["history"][-1], flush=True)
        del evaluator, products, result
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
