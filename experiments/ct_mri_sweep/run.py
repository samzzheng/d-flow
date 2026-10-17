"""Resumable worker for one (model, acquisition) and all paired test samples."""
from __future__ import annotations

import argparse
from contextlib import suppress
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dlr_dflow.api import approx_source_gradient, load_flow_model, sensitivity_factors
from dlr_dflow.implementation import enriched, tangent_unet
from dlr_dflow.implementation.graphed_products import GraphedAProducts
from p_flow import endpoint_gradient
from p_flow.armijo import ArmijoLBFGS
from sampler.ode_samplers import euler_flow
from scripts.reconstruction_figures import METHODS, RESIDUAL_VMAX, plot_reconstruction_comparison
from .operators import CT, MRI, noisy_measurement
from .validation import ChunkedEagerProducts

ROOT = Path(__file__).resolve().parents[2]
Settings = dict[str, Any]
TERMINAL = {"objective_plateau", "step_limit", "stalled"}


@dataclass(frozen=True)
class Job:
    resolution: int
    circles: int
    problem: str
    level: int

    @property
    def name(self) -> str:
        if self.circles == 0:  # Mixed-count separated-shapes dataset, not zero circles.
            return f"r{self.resolution}_shapes_{self.problem}{self.level}"
        return f"r{self.resolution}_n{self.circles}_{self.problem}{self.level}"


def jobs(settings: Settings) -> list[Job]:
    return [Job(r, n, problem, level) for r in reversed(settings["resolutions"])
            for n in settings["circles"] for problem, levels in (
                ("ct", settings["ct_angles"]), ("mri", settings["mri_accelerations"]))
            for level in reversed(levels)]


def model_directory(job: Job, settings: Settings) -> Path:
    return ROOT / settings["run_template"].format(**asdict(job))


def model_ready(job: Job, settings: Settings) -> bool:
    run = model_directory(job, settings)
    summary = run / "run_summary.json"
    if not summary.exists():
        return False
    record = {"status": "pending", "epochs_completed": 0}
    with suppress(json.JSONDecodeError):
        record = json.loads(summary.read_text())
    return (record["status"] == "complete" and record["epochs_completed"] == settings["epoch"]
            and (run / f"checkpoints/epoch_{settings['epoch']:04d}.pt").is_file())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def reviewed_source_hashes(manifest_path: Path, manifest: Settings) -> dict[str, str]:
    """Apply an explicit source review without changing checkpoint protocol IDs."""
    expected = dict(manifest["source_sha256"])
    review_path = manifest_path.with_name("source_review.json")
    if review_path.exists():
        review = json.loads(review_path.read_text())
        if review["manifest_sha256"] != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
            raise ValueError("Source review belongs to a different manifest")
        for name, change in review["changes"].items():
            if expected.get(name) != change["before"]:
                raise ValueError(f"Source review baseline mismatch: {name}")
            expected[name] = change["after"]
    return expected


def tensor_tree(value: Any, device: str | torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device)
    if isinstance(value, dict):
        return {key: tensor_tree(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(tensor_tree(item, device) for item in value)
    return value


def save_tensor(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".pt.tmp")
    torch.save(tensor_tree(value, "cpu"), temporary)
    temporary.replace(path)


def metrics(image: torch.Tensor, predicted: torch.Tensor, truth: torch.Tensor,
            clean: torch.Tensor, observed: torch.Tensor) -> Settings:
    noise_norm = float((observed - clean).norm())
    return {"image_relative_l2": float((image - truth).norm() / truth.norm()),
            "measurement_relative_l2_clean": float((predicted - clean).norm() / clean.norm()),
            "measurement_relative_l2_noisy": float((predicted - observed).norm() / observed.norm()),
            "residual_over_noise": float((predicted - observed).norm()) / noise_norm if noise_norm else None}


class Evaluator:
    """One canonical Euler objective; the accepted trial is reused for proxy gx."""

    def __init__(self, model: torch.nn.Module, operator: CT | MRI, observed: torch.Tensor,
                 settings: Settings, seed: int, products: GraphedAProducts | None = None) -> None:
        self.model, self.operator, self.observed = model, operator, observed
        self.settings, self.seed, self.products = settings, seed, products
        self.source: torch.Tensor | None = None
        self.image: torch.Tensor | None = None
        self.value = 0.0
        self.counts = {"forward_objectives": 0, "gradients": 0}
        self.previous_v: torch.Tensor | None = None

    def objective(self, source: torch.Tensor) -> float:
        with torch.no_grad():
            self.image = euler_flow(self.model, source, self.settings["flow_steps"])
            residual = self.operator(self.image) - self.observed
            self.value = float(0.5 * residual.abs().square().sum(dtype=torch.float64))
        self.source = source.detach().clone()
        self.counts["forward_objectives"] += 1
        return self.value

    def gradient(self, source: torch.Tensor, method: str) -> torch.Tensor:
        if self.source is None or not torch.equal(self.source, source.detach()):
            self.objective(source.detach())
        self.counts["gradients"] += 1
        if method == "D-Flow":
            image = euler_flow(self.model, source, self.settings["flow_steps"])
            loss = 0.5 * (self.operator(image) - self.observed).abs().square().sum(dtype=torch.float64)
            gradient, = torch.autograd.grad(loss, source)
        else:
            _, gradient = endpoint_gradient(self.image, self.operator, self.observed)
            if method == "DLR D-Flow":
                _, factors, alpha = sensitivity_factors(
                    self.model, source.detach(), self.settings["flow_steps"],
                    **self.settings["dlr"], seed=self.seed, products=self.products,
                    seed_basis=self.previous_v)
                if alpha != 0.0:
                    raise FloatingPointError("Terminal analytical alpha must be exactly zero")
                gradient = approx_source_gradient(factors, gradient, alpha)
                if self.settings.get("dlr_reuse_previous_v", False):
                    # Graph outputs may be overwritten on the next evaluation.
                    self.previous_v = factors[2].detach().clone()
        if not torch.isfinite(gradient).all() or not math.isfinite(self.value):
            raise FloatingPointError("Nonfinite objective or gradient")
        return gradient.detach()


def solve(evaluator: Evaluator, method: str, initial: torch.Tensor, truth: torch.Tensor,
          clean: torch.Tensor, path: Path, protocol: Settings, interrupted: list[bool]) -> Settings:
    settings = evaluator.settings
    source = initial.detach().clone().requires_grad_(True)
    optimizer = ArmijoLBFGS(source, evaluator.objective, **{
        **settings["optimizer"], "step_cap": settings["optimizer"]["step_cap"] * math.sqrt(source.numel()),
        "project": settings["source_projection"]})
    history: list[Settings] = []
    prior_seconds = 0.0
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["protocol"] != protocol:
            raise ValueError(f"Resume protocol mismatch: {path}")
        if saved["status"] in TERMINAL:
            write_json(path.with_suffix(".json"), {key: saved[key] for key in
                       ("protocol", "method", "status", "history", "evaluations", "elapsed_seconds")})
            return saved
        with torch.no_grad():
            source.copy_(saved["source"].to(source.device))
        for name, value in saved["optimizer"].items():
            setattr(optimizer, name, tensor_tree(value, source.device))
        history, prior_seconds = saved["history"], saved["elapsed_seconds"]
        evaluator.counts = saved["evaluations"]
        evaluator.previous_v = tensor_tree(saved.get("previous_v"), source.device)
    started = time.monotonic()

    def closure() -> torch.Tensor:
        source.grad = evaluator.gradient(source, method)
        return source.new_tensor(evaluator.value, dtype=torch.float64)

    evaluator.objective(source.detach())
    if not history:
        history.append({"step": 0, "phi": evaluator.value, "elapsed_seconds": prior_seconds,
                        **metrics(evaluator.image, evaluator.operator(evaluator.image), truth, clean, evaluator.observed)})
    status = "running"
    while status == "running":
        if interrupted[0]:
            status = "paused"
        else:
            before = source.detach().clone()
            optimizer.step(closure)
            # A failed search leaves the objective cache at a rejected candidate.
            if not torch.equal(evaluator.source, source.detach()):
                evaluator.objective(source.detach())
            displacement = float((source.detach() - before).norm())
            if displacement > optimizer.step_cap * (1 + 1e-4):
                raise AssertionError("Proposed displacement cap violated")
            previous = history[-1]["phi"]
            if displacement > 0 and evaluator.value >= previous:
                raise AssertionError("Accepted step did not strictly lower the canonical objective")
            history.append({"step": optimizer.step_attempts, "phi": evaluator.value,
                            "source_step_norm": displacement,
                            "line_search_trials": optimizer.last_function_evaluations,
                            "elapsed_seconds": prior_seconds + time.monotonic() - started,
                            **metrics(evaluator.image, evaluator.operator(evaluator.image), truth, clean, evaluator.observed)})
            stopping = settings["stopping"]
            if optimizer.consecutive_failures >= stopping["failed_searches"]:
                status = "stalled"
            elif optimizer.step_attempts >= stopping["max_updates"]:
                status = "step_limit"
            elif optimizer.step_attempts >= stopping["min_updates"] and optimizer.step_attempts % stopping["window"] == 0:
                window = stopping["window"]
                values = [history[index]["phi"] for index in (-1 - 2 * window, -1 - window, -1)]
                changes = [(left - right) / max(abs(left), 1e-30) for left, right in zip(values, values[1:])]
                if max(changes) < stopping["plateau_relative"]:
                    status = "objective_plateau"
        if status != "running" or optimizer.step_attempts % settings["execution"]["checkpoint_every"] == 0:
            saved = {"protocol": protocol, "method": method, "status": status,
                     "source": source.detach(), "image": evaluator.image,
                     "measurement": evaluator.operator(evaluator.image).detach(),
                     "previous_v": evaluator.previous_v,
                     "optimizer": {key: value for key, value in vars(optimizer).items() if key not in {"source", "objective"}},
                     "history": history, "evaluations": evaluator.counts,
                     "elapsed_seconds": prior_seconds + time.monotonic() - started}
            save_tensor(path, saved)
            summary = {key: saved[key] for key in ("protocol", "method", "status", "history", "evaluations", "elapsed_seconds")}
            write_json(path.with_suffix(".json"), summary)
            print(f"{path.parent.name} {method} step={optimizer.step_attempts} {status} image={100*history[-1]['image_relative_l2']:.3f}%", flush=True)
    return tensor_tree(saved, "cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--sample-stop", type=int)
    parser.add_argument("--memory-fraction", type=float)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    settings = manifest["settings"]
    sample_stop = settings["samples"] if args.sample_stop is None else args.sample_stop
    if not 0 <= args.sample_start < sample_stop <= settings["samples"]:
        parser.error("Sample interval must lie inside the configured test subset")
    for name, expected in reviewed_source_hashes(args.manifest, manifest).items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source changed after queuing: {name}")
    job = next(job for job in jobs(settings) if job.name == args.job)
    if not model_ready(job, settings):
        raise RuntimeError("Training is not complete for this model")
    execution = settings["execution"]
    torch.set_num_threads(execution["threads"])
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(
        execution["memory_fraction"] if args.memory_fraction is None else args.memory_fraction, device)
    torch.backends.cudnn.allow_tf32 = execution["cudnn_tf32"]
    torch.backends.cuda.matmul.allow_tf32 = execution["matmul_tf32"]
    torch.backends.cudnn.benchmark = execution["cudnn_benchmark"]
    run = model_directory(job, settings)
    architecture = json.loads((run / "config.json").read_text())
    dataset_template = settings.get("dataset_template", "data/circles-{resolution}-n{circles}.pt")
    data = torch.load(ROOT / dataset_template.format(**asdict(job)),
                      map_location="cpu", weights_only=True, mmap=True)
    if data["metadata"] != architecture["dataset_metadata"]:
        raise ValueError("Local evaluation dataset metadata differs from training dataset")
    test_hash = hashlib.sha256(data["test"][:settings["samples"]].numpy().tobytes()).hexdigest()
    model, _ = load_flow_model(job.resolution, job.circles, settings["epoch"], str(device), run)
    checkpoint_hash = hashlib.sha256((run / f"checkpoints/epoch_{settings['epoch']:04d}.pt").read_bytes()).hexdigest()
    tangent_unet.GN_STATS_MODE = "native" if job.resolution <= 96 else "manual"
    tangent_unet.GN_ACTION_BACKEND = execution["dlr_fused_norm"]
    tangent_unet.GN_PRIMAL_BACKEND = execution["dlr_fused_primal"]
    enriched.SPECIALIZE_ZERO_ENRICHMENT = execution["dlr_specialized_algebra"]
    enriched.SKIP_UNTRUNCATED_SVD = True
    initial_shape = torch.zeros(1, 1, job.resolution, job.resolution, device=device)
    products = GraphedAProducts(model, initial_shape, share_linearization=True,
                               full_step_graphs=execution["dlr_full_step_graphs"],
                               max_jvp_columns=20, max_vjp_columns=16,
                               max_step_jvp_columns=execution["dlr_jvp_columns"],
                               max_step_vjp_columns=execution["dlr_vjp_columns"])
    interrupted = [False]

    def stop(signum: int, frame: Any) -> None:
        interrupted[0] = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    output = args.manifest.parent / job.name
    output.mkdir(exist_ok=True)
    for sample in range(args.sample_start, sample_stop):
        case = output / f"sample_{sample:03d}"
        case.mkdir(exist_ok=True)
        source_seed = settings["seed"] + 10000 * job.resolution + 100 * job.circles + sample
        mask_seed, noise_seed, dlr_seed = source_seed + 10000000, source_seed + 20000000, source_seed + 30000000
        operator = (CT(job.resolution, job.level, device) if job.problem == "ct" else
                    MRI(job.resolution, job.level, mask_seed, device, settings["mri_pattern"]))
        stored = data["test"][sample:sample + 1]
        truth = stored.to(device=device, dtype=torch.float32) / (255 if stored.dtype == torch.uint8 else 1)
        initial = torch.randn(truth.shape, generator=torch.Generator().manual_seed(source_seed)).to(device)
        clean = operator(truth)
        observed = noisy_measurement(clean, settings["noise_relative"], noise_seed, getattr(operator, "mask", None))
        protocol = {"manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                    "checkpoint_sha256": checkpoint_hash, "test_subset_sha256": test_hash,
                    "job": asdict(job), "sample": sample,
                    "operator": operator.metadata, "source_seed": source_seed, "noise_seed": noise_seed,
                    "dlr_seed": dlr_seed, "torch": str(torch.__version__),
                    "gpu": torch.cuda.get_device_name(device)}
        if sample == args.sample_start:
            # Validate the fastest backend on this checkpoint before any reconstructions.
            with torch.no_grad():
                canonical = euler_flow(model, initial, settings["flow_steps"])
            _, gx = endpoint_gradient(canonical, operator, observed)
            fast_image, fast_factors, alpha = sensitivity_factors(
                model, initial, settings["flow_steps"], **settings["dlr"], seed=dlr_seed, products=products)
            reference_products = ChunkedEagerProducts(model) if settings["dlr"]["rank"] > 16 else None
            _, reference_factors, _ = sensitivity_factors(
                model, initial, settings["flow_steps"], **settings["dlr"], seed=dlr_seed, products=reference_products)
            fast_gradient = approx_source_gradient(fast_factors, gx, alpha)
            reference_gradient = approx_source_gradient(reference_factors, gx, alpha)
            error = float((fast_gradient - reference_gradient).norm() / reference_gradient.norm().clamp_min(1e-20))
            endpoint_error = float((fast_image - canonical).norm() / canonical.norm().clamp_min(1e-20))
            passed = math.isfinite(error) and error < 0.01 and math.isfinite(endpoint_error) and endpoint_error < 0.005 and alpha == 0
            seeded_error = None
            if settings.get("dlr_reuse_previous_v", False):
                seed_basis = fast_factors[2].detach().clone()
                _, seeded_fast, seeded_alpha = sensitivity_factors(
                    model, initial, settings["flow_steps"], **settings["dlr"], seed=dlr_seed,
                    products=products, seed_basis=seed_basis)
                _, seeded_reference, _ = sensitivity_factors(
                    model, initial, settings["flow_steps"], **settings["dlr"], seed=dlr_seed,
                    seed_basis=seed_basis, products=reference_products)
                seeded_gradient = approx_source_gradient(seeded_fast, gx, seeded_alpha)
                reference_gradient = approx_source_gradient(seeded_reference, gx, seeded_alpha)
                seeded_error = float((seeded_gradient - reference_gradient).norm() / reference_gradient.norm().clamp_min(1e-20))
                passed = passed and math.isfinite(seeded_error) and seeded_error < 0.01 and seeded_alpha == 0
                del seed_basis, seeded_fast, seeded_reference, seeded_gradient
            write_json(output / f"backend_validation_{args.sample_start:03d}.json", {
                "passed": passed, "gradient_relative_error_graph_vs_eager_dlr": error if math.isfinite(error) else None,
                "seeded_gradient_relative_error_graph_vs_eager_dlr": seeded_error if seeded_error is not None and math.isfinite(seeded_error) else None,
                "endpoint_relative_error_fast_vs_native": endpoint_error if math.isfinite(endpoint_error) else None, "terminal_alpha": float(alpha),
                "checkpoint_sha256": checkpoint_hash, "note": "Execution equivalence, not DLR vs exact gradient accuracy."})
            if not passed:
                raise RuntimeError("Fast backend validation failed; see backend_validation output")
            del fast_image, fast_factors, reference_factors, fast_gradient, reference_gradient, reference_products, canonical, gx
        if not (case / "inputs.pt").exists():
            save_tensor(case / "inputs.pt", {"protocol": protocol, "truth": truth, "clean": clean,
                                            "observed": observed, "initial": initial, "mask": getattr(operator, "mask", None)})
        # Rotate execution order without changing fixed figure columns or seeds.
        order = METHODS[sample % 3:] + METHODS[:sample % 3]
        results = {}
        if settings.get("baseline_out") and not manifest["protocol"]["smoke_only"]:
            baseline_case = Path(settings["baseline_out"]) / "production" / job.name / case.name
            for method, stem in (("D-Flow", "d_flow"), ("P-Flow", "p_flow")):
                control_path = baseline_case / f"{stem}.pt"
                if not control_path.exists():
                    continue
                control = torch.load(control_path, map_location="cpu", weights_only=True)
                if control["status"] not in TERMINAL:
                    continue
                if any(control["protocol"][key] != value for key, value in protocol.items()
                       if key != "manifest_sha256"):
                    raise ValueError(f"Unpaired control: {control_path}")
                results[method] = control
        for method in order:
            evaluator = Evaluator(model, operator, observed, settings, dlr_seed, products)
            path = case / (method.lower().replace("-", "_").replace(" ", "_") + ".pt")
            results[method] = solve(evaluator, method, initial, truth, clean, path, protocol, interrupted)
            fig = plot_reconstruction_comparison(truth, clean, {
                name: (result["image"], result["measurement"]) for name, result in results.items()},
                observation=observed, measurement_log=job.problem == "mri",
                residual_vmax=RESIDUAL_VMAX[job.problem],
                histories={name: result["history"] for name, result in results.items()},
                title=f"Resolution: {job.resolution}$^2$, Inverse Problem: {job.problem.upper()}, "
                      f"Noise Level {100*settings['noise_relative']:g}%, Sample: {sample}")
            for extension in ("png", "pdf"):
                fig.savefig(case / f"comparison.{extension}", dpi=140)
            plt.close(fig)
            if interrupted[0]:
                break
        if interrupted[0]:
            return
    shard = {
        "sample_start": args.sample_start, "sample_stop": sample_stop,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20}
    if not all((output / f"sample_{sample:03d}" / f"{method.lower().replace('-', '_').replace(' ', '_')}.json").exists()
               for sample in range(settings["samples"]) for method in METHODS):
        write_json(output / f"shard_{args.sample_start:03d}_{sample_stop:03d}.json", shard)
        return
    summaries = {}
    for method in METHODS:
        stem = method.lower().replace("-", "_").replace(" ", "_")
        records = [json.loads((output / f"sample_{sample:03d}" / f"{stem}.json").read_text())
                   for sample in range(settings["samples"])]
        if any(record["status"] not in TERMINAL for record in records):
            write_json(output / f"shard_{args.sample_start:03d}_{sample_stop:03d}.json", shard)
            return
        errors = [record["history"][-1]["image_relative_l2"] for record in records]
        summaries[method] = {"mean_image_relative_l2": statistics.mean(errors),
                             "std_image_relative_l2": statistics.stdev(errors) if len(errors) > 1 else 0.0,
                             "median_image_relative_l2": statistics.median(errors),
                             "elapsed_seconds": sum(record["elapsed_seconds"] for record in records),
                             "mean_updates": statistics.mean(len(record["history"])-1 for record in records),
                             "statuses": {status: sum(record["status"] == status for record in records) for status in sorted(TERMINAL)}}
    write_json(output / "complete.json", {"job": asdict(job), "samples": settings["samples"],
                                          "methods": summaries, "checkpoint_sha256": checkpoint_hash})
    write_json(output / f"shard_{args.sample_start:03d}_{sample_stop:03d}.json", shard)


if __name__ == "__main__":
    main()
