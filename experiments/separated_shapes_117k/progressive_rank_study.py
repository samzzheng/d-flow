"""Paired high-start adaptive DLR controls on cached separated-shapes samples."""
from contextlib import nullcontext
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import signal

import torch

from dlr_dflow.api import load_flow_model
from dlr_dflow.implementation import tangent_unet
from experiments.separated_shapes_117k.bounded_products import BoundedGraphedProducts
from experiments.ct_mri_sweep import run
from experiments.ct_mri_sweep.operators import CT, MRI
from experiments.ct_mri_sweep.queue import source_hashes
from experiments.separated_shapes_117k.diagnose import RankRecordingEvaluator
from experiments.separated_shapes_117k.progressive_rank import progressive_rank
from scripts.reconstruction_figures import RESIDUAL_VMAX, plot_reconstruction_comparison


class ProgressiveEvaluator(RankRecordingEvaluator):
    retain_fraction: float | None = None

    def gradient(self, source: torch.Tensor, method: str) -> torch.Tensor:
        with (nullcontext() if self.retain_fraction is None else
              progressive_rank(self.settings["dlr"]["rank_min"], self.retain_fraction)):
            return super().gradient(source, method)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path,
        default=Path("outputs/separated_shapes_117k_noiseless_20261021/production"))
    parser.add_argument("--case", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--diagnostics-only", action="store_true")
    parser.add_argument("--resume-reviewed-script-sha256",
        help="Previous script hash after review of the graph-cache-only change; preserve saved protocol and optimizer")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda:1")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.2, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    interrupted = [False]

    def stop(signum: int, frame: object) -> None:
        interrupted[0] = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    inputs = torch.load(args.baseline/args.case/"inputs.pt", map_location=device, weights_only=True)
    protocol = inputs["protocol"]
    settings = json.loads((args.baseline/"manifest.json").read_text())["settings"]
    settings["dlr"].update(rank=256, adaptive_rank=True, rank_min=32, rank_max=256, rank_tol=1e-3)
    resolution = protocol["job"]["resolution"]
    folder = Path(settings["run_template"].format(resolution=resolution))
    assert hashlib.sha256((folder/"checkpoints/epoch_0500.pt").read_bytes()).hexdigest() == protocol["checkpoint_sha256"]
    assert torch.equal(inputs["observed"], inputs["clean"])
    model, _ = load_flow_model(resolution, 0, 500, str(device), folder)
    info = protocol["operator"]
    operator = (CT(resolution, protocol["job"]["level"], device) if protocol["job"]["problem"] == "ct"
                else MRI(resolution, 4, info["mask_seed"], device, info["pattern"]))
    if isinstance(operator, MRI):
        assert torch.equal(operator.mask, inputs["mask"])
    tangent_unet.GN_STATS_MODE = "native"
    tangent_unet.GN_ACTION_BACKEND = "native"
    tangent_unet.GN_PRIMAL_BACKEND = "native"
    products = BoundedGraphedProducts(model, inputs["initial"], share_linearization=True,
        full_step_graphs=False, max_jvp_columns=20, max_vjp_columns=16)
    directory = args.out/args.case
    directory.mkdir(parents=True, exist_ok=True)
    experiment = dict(settings=settings, source_hashes=source_hashes(),
        initial_source_sha256=hashlib.sha256(inputs["initial"].cpu().numpy().tobytes()).hexdigest(),
        policy_sha256=hashlib.sha256(Path(progressive_rank.__wrapped__.__code__.co_filename).read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        normalization="native statistics, primal and derivative actions", precision="float32; TF32 disabled",
        note="Rank adaptation within each 5-step flow; fresh high-rank initialization each gradient, previous V reuse unchanged. No growth within a pass, no enrichment, analytical alpha. Three paired controls; no production default changes.")
    config = directory/"config.json"
    if config.exists():
        previous = json.loads(config.read_text())
        if args.resume_reviewed_script_sha256:
            assert previous["script_sha256"] == args.resume_reviewed_script_sha256
            assert {key: value for key, value in previous.items() if key != "script_sha256"} == {
                key: value for key, value in experiment.items() if key != "script_sha256"}
            run.write_json(directory/"resume_cache_review.json", dict(
                previous_script_sha256=previous["script_sha256"], current_script_sha256=experiment["script_sha256"],
                bounded_products_sha256=hashlib.sha256(Path(BoundedGraphedProducts.__init__.__code__.co_filename).read_bytes()).hexdigest(),
                graph_cache_size=products.cache_size,
                note="Resume after confirmed per-process graph-cache OOM. Only graph eviction changes; widths, factor equations, optimizer/source/history/basis and budgets remain unchanged."))
            experiment = previous
        else:
            assert previous == experiment
    run.write_json(config, experiment)
    variable = inputs["initial"].detach().requires_grad_(True)
    exact_evaluator = run.Evaluator(model, operator, inputs["observed"], settings, protocol["dlr_seed"], products)
    exact_evaluator.method = "D-Flow"
    exact = exact_evaluator.gradient(variable, "D-Flow")
    variants = {"unrestricted256": None, "retain_initial256": 0., "gradual256": .5}
    diagnostics = {}
    for name, fraction in variants.items():
        path = directory/name
        path.mkdir(exist_ok=True)
        evaluator = ProgressiveEvaluator(model, operator, inputs["observed"], deepcopy(settings), protocol["dlr_seed"], products)
        evaluator.retain_fraction = fraction
        evaluator.rank_history_path = path/"initial_ranks.jsonl"
        gradient = evaluator.gradient(variable, "DLR D-Flow")
        diagnostics[name] = dict(relative_gradient_error=float((gradient-exact).norm()/exact.norm()),
            gradient_cosine=float(torch.nn.functional.cosine_similarity(gradient.flatten(), exact.flatten(), dim=0)),
            relative_endpoint_difference=float((evaluator.image-exact_evaluator.image).norm()/exact_evaluator.image.norm()))
        print(name, diagnostics[name], flush=True)
    run.write_json(directory/"initial_gradients.json", diagnostics)
    if args.diagnostics_only or interrupted[0]:
        return
    results = {}
    for name, fraction in variants.items():
        path = directory/name
        evaluator = ProgressiveEvaluator(model, operator, inputs["observed"], deepcopy(settings), protocol["dlr_seed"], products)
        evaluator.retain_fraction = fraction
        evaluator.rank_history_path = path/"rank_history.jsonl"
        paired = {**protocol, "experiment": experiment, "rank_policy": name, "retain_fraction": fraction}
        result = run.solve(evaluator, "DLR D-Flow", inputs["initial"], inputs["truth"], inputs["clean"],
                           path/"dlr_d_flow.pt", paired, interrupted)
        results[name] = dict(status=result["status"], final=result["history"][-1])
        run.write_json(directory/"summary.json", results)
        if interrupted[0]:
            break
        fig = plot_reconstruction_comparison(inputs["truth"], inputs["clean"],
            {"DLR D-Flow": (result["image"], result["measurement"])}, observation=inputs["observed"],
            include_errors=True, measurement_log=isinstance(operator, MRI),
            residual_vmax=RESIDUAL_VMAX[protocol["job"]["problem"]], title=f"{args.case}: {name}, noiseless")
        fig.savefig(path/"comparison.png", dpi=120)
        run.plt.close(fig)


if __name__ == "__main__":
    main()
