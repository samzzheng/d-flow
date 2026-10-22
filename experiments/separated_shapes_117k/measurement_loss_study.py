"""Paired low-rank measurement-loss pilot, one sequential worker per sample."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import signal
from typing import Any

import matplotlib.pyplot as plt
import torch

from dlr_dflow.api import load_flow_model
from dlr_dflow.implementation import enriched, tangent_unet
from experiments.ct_mri_sweep import run
from experiments.ct_mri_sweep.operators import CT, MRI
from experiments.separated_shapes_117k.bounded_products import BoundedGraphedProducts
from experiments.separated_shapes_117k.measurement_losses import LossEvaluator
from scripts.reconstruction_figures import RESIDUAL_VMAX, plot_reconstruction_comparison

VARIANTS = ("l2", "coarse_to_fine", "intensity", "l2_resets")
METHODS = {"dlr_d_flow": "DLR D-Flow", "d_flow": "D-Flow", "p_flow": "P-Flow"}


def report(folder: Path, inputs: dict[str, Any], problem: str) -> None:
    """Refresh completed-arm metrics and shared panels, verifying saved arrays."""
    rows = []
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, (slug, method) in zip(axes, METHODS.items()):
        for variant in VARIANTS:
            summary_path = folder / variant / slug / "summary.json"
            if not summary_path.exists():
                continue
            summary = json.loads(summary_path.read_text())
            saved = torch.load(folder / summary["final_checkpoint"], map_location="cpu", weights_only=True)
            image_error = float((saved["image"].double() - inputs["truth"].cpu().double()).norm()
                                / inputs["truth"].cpu().double().norm())
            observed = inputs["observed"].cpu().to(torch.complex128 if problem == "mri" else torch.float64)
            residual = saved["measurement"].to(observed.dtype) - observed
            residual_error = float(residual.norm() / observed.norm())
            assert abs(image_error - summary["history"][-1]["image_relative_l2"]) < 1e-6
            assert abs(residual_error - summary["history"][-1]["measurement_relative_l2_noisy"]) < 1e-6
            rows.append(dict(variant=variant, method=method, attempts=summary["attempts"],
                image_error_percent=100*image_error, residual_percent=100*residual_error,
                original_l2=float(.5*residual.abs().square().sum()), status=summary["status"]))
            axis.plot([row["step"] for row in summary["history"]],
                      [100*row["image_relative_l2"] for row in summary["history"]], label=variant)
        axis.set(title=method, xlabel="Optimization attempts", ylabel="Relative image error (%)")
        if axis.lines:
            axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(folder / "convergence.png", dpi=120)
    plt.close(figure)
    for variant in VARIANTS:
        reconstructions = {}
        for slug, method in METHODS.items():
            summary_path = folder / variant / slug / "summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                saved = torch.load(folder / summary["final_checkpoint"], map_location="cpu", weights_only=True)
                reconstructions[method] = (saved["image"], saved["measurement"])
        if reconstructions:
            figure = plot_reconstruction_comparison(inputs["truth"], inputs["clean"], reconstructions,
                observation=inputs["observed"], include_errors=True, measurement_log=problem == "mri",
                residual_vmax=RESIDUAL_VMAX[problem], title=f"{folder.name}: {variant}, noiseless")
            figure.savefig(folder / variant / "comparison.png", dpi=100, bbox_inches="tight")
            plt.close(figure)
    run.write_json(folder / "results.json", rows)
    lines = ["# Measurement loss pilot", "", "Completed arms only. All errors use original arrays; residuals use ordinary L2.",
             "Weighted objectives are not comparable across losses or stages. Selected cases, not a population study.", "",
             "| Loss | Method | Attempts | Image error (%) | Measurement residual (%) | End status |",
             "| --- | --- | ---: | ---: | ---: | --- |"]
    lines.extend(f"| {row['variant']} | {row['method']} | {row['attempts']} | {row['image_error_percent']:.4f} | {row['residual_percent']:.4f} | {row['status']} |" for row in rows)
    (folder / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-updates", type=int, default=1000)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    args = parser.parse_args()
    if args.max_updates < 4:
        parser.error("At least four attempts are needed for four loss stages")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda:1")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.15, device)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    interrupted = [False]

    def stop(signum: int, frame: object) -> None:
        interrupted[0] = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    baseline = Path("outputs/separated_shapes_117k_noiseless_20261021/production")
    inputs = torch.load(baseline / args.case / "inputs.pt", map_location=device, weights_only=True)
    protocol = inputs["protocol"]
    settings = json.loads((baseline / "manifest.json").read_text())["settings"]
    assert settings["flow_steps"] == 5 and torch.equal(inputs["observed"], inputs["clean"])
    settings["dlr"].update(rank=32, adaptive_rank=False)
    settings["execution"].update(memory_fraction=.15, cudnn_deterministic=True,
        dlr_full_step_graphs=False, dlr_jvp_columns=20, dlr_vjp_columns=16,
        dlr_fused_norm="native", dlr_fused_primal="native")
    settings["stopping"]["max_updates"] = args.max_updates
    resolution, problem = protocol["job"]["resolution"], protocol["job"]["problem"]
    model_folder = Path(settings["run_template"].format(resolution=resolution))
    assert hashlib.sha256((model_folder / "checkpoints/epoch_0500.pt").read_bytes()).hexdigest() == protocol["checkpoint_sha256"]
    model, _ = load_flow_model(resolution, 0, 500, str(device), model_folder)
    info = protocol["operator"]
    operator = (CT(resolution, info["angle_count"], device) if problem == "ct" else
                MRI(resolution, 4, info["mask_seed"], device, info["pattern"]))
    if problem == "mri":
        assert torch.equal(operator.mask, inputs["mask"])
    torch.testing.assert_close(operator(inputs["truth"]), inputs["clean"], rtol=1e-6, atol=1e-7)
    enriched.SPECIALIZE_ZERO_ENRICHMENT = enriched.SKIP_UNTRUNCATED_SVD = True
    tangent_unet.GN_STATS_MODE = tangent_unet.GN_ACTION_BACKEND = tangent_unet.GN_PRIMAL_BACKEND = "native"
    sketch = torch.randint(0, 2, (inputs["initial"].numel(), 40), device=device,
        generator=torch.Generator(device=device).manual_seed(protocol["dlr_seed"]), dtype=torch.int8).float()
    sketch.mul_(2).sub_(1)
    source_files = {**{name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                      for name in json.loads((baseline / "manifest.json").read_text())["source_sha256"]},
                    **{str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
                        Path(__file__), Path(__file__).with_name("measurement_losses.py"),
                        Path(__file__).with_name("bounded_products.py"))}}
    config = dict(case=args.case, settings=settings, source_sha256=hashlib.sha256(inputs["initial"].cpu().numpy().tobytes()).hexdigest(),
        sketch_sha256=hashlib.sha256(sketch.cpu().numpy().tobytes()).hexdigest(), source_files=source_files,
        methods=args.methods, variants=args.variants, beta=.1, intensity_weight=9.,
        sigma_fractions=[.04, .02, .01, 0.], stage_limit=min(100, args.max_updates // 4),
        note="Final stage receives all remaining attempts. Two failed searches advance to next stage. L2-resets follows identical stage limits. Fresh LBFGS and objective cache at each stage; retain previous V. No truth-based scheduling.")
    if (args.out / "config.json").exists():
        assert json.loads((args.out / "config.json").read_text()) == config, "Resume configuration changed"
    run.write_json(args.out / "config.json", config)
    status = dict(status="running", pid=os.getpid(), case=args.case)
    run.write_json(args.out / "status.json", status)
    products = BoundedGraphedProducts(model, inputs["initial"], cache_size=4, share_linearization=True,
        full_step_graphs=False, max_jvp_columns=20, max_vjp_columns=16)
    for slug in args.methods:
        for variant in args.variants:
            folder = args.out / variant / slug
            folder.mkdir(parents=True, exist_ok=True)
            source, previous_v = inputs["initial"].clone(), sketch.clone()
            combined, used = [], 0
            scales = config["sigma_fractions"] if variant in ("coarse_to_fine", "l2_resets") else [0.]
            for stage, fraction in enumerate(scales):
                stage_settings = deepcopy(settings)
                stage_settings["measurement_loss"] = dict(kind="coarse" if variant == "coarse_to_fine" else
                    "intensity" if variant == "intensity" else "l2", sigma=fraction * resolution,
                    beta=config["beta"], intensity_weight=config["intensity_weight"])
                stage_settings["stopping"]["max_updates"] = (args.max_updates - used if stage == len(scales) - 1
                                                              else config["stage_limit"])
                evaluator = LossEvaluator(model, operator, inputs["observed"], stage_settings,
                                          protocol["dlr_seed"], products)
                evaluator.previous_v = previous_v
                path = folder / f"stage_{stage}.pt"
                status.update(method=slug, variant=variant, stage=stage, attempts_before_stage=used)
                run.write_json(args.out / "status.json", status)
                saved = run.solve(evaluator, METHODS[slug], source, inputs["truth"], inputs["clean"], path,
                    {**protocol, "experiment": dict(settings=stage_settings, variant=variant, stage=stage,
                        starting_source_sha256=hashlib.sha256(source.cpu().numpy().tobytes()).hexdigest(),
                        source_files=source_files)}, interrupted)
                for row in saved["history"]:
                    combined.append({**row, "step": used + row["step"], "stage": stage,
                        "original_l2": .5 * row["measurement_relative_l2_noisy"]**2 * float(inputs["observed"].norm())**2})
                used += saved["history"][-1]["step"]
                assert used <= args.max_updates
                if interrupted[0]:
                    status["status"] = "paused"
                    run.write_json(args.out / "status.json", status)
                    return
                source = saved["source"].to(device)
                previous_v = saved["previous_v"].to(device) if saved["previous_v"] is not None else None
                del evaluator
            run.write_json(folder / "summary.json", dict(status=saved["status"], attempts=used,
                history=combined, final_checkpoint=str(path.relative_to(args.out))))
            report(args.out, inputs, problem)
    status["status"] = "complete"
    run.write_json(args.out / "status.json", status)


if __name__ == "__main__":
    main()
