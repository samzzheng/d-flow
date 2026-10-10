"""Paired FP32 GPU experiments for literal and secant-corrected DLR transport."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import torch

from dlr_dflow.api import Factors, approx_source_gradient, load_flow_model, sensitivity_factors
from dlr_dflow.implementation import enriched, tangent_unet
from dlr_dflow.implementation.graphed_products import GraphedAProducts
from experiments.ct_mri_sweep.operators import CT, MRI
from experiments.ct_mri_sweep.run import Evaluator, metrics, save_tensor, write_json
from p_flow import endpoint_gradient
from p_flow.armijo import ArmijoLBFGS
from .secant import secant_factors
from .transport import transport_factors

Record = dict[str, Any]


class TransportEvaluator(Evaluator):
    """Approximate source gradients with a fresh solve or source-step transport."""

    def __init__(self, model: torch.nn.Module, operator: CT | MRI, observed: torch.Tensor,
                 settings: Record, seed: int, products: GraphedAProducts,
                 mode: str, refresh_every: int = 5) -> None:
        super().__init__(model, operator, observed, settings, seed, products)
        self.mode, self.refresh_every = mode, refresh_every
        self.factors: Factors | None = None
        self.factor_source: torch.Tensor | None = None
        self.age = 0
        self.gradient_records: list[Record] = []

    def gradient(self, source: torch.Tensor, method: str = "") -> torch.Tensor:
        if self.source is None or not torch.equal(self.source, source.detach()):
            self.objective(source.detach())
        torch.cuda.synchronize(source.device)
        started = time.perf_counter()
        endpoint_error = None
        kind = self.mode
        if self.mode in {"exact", "proxy"}:
            gradient = super().gradient(source, "D-Flow" if self.mode == "exact" else "P-Flow")
        else:
            self.counts["gradients"] += 1
            _, gx = endpoint_gradient(self.image, self.operator, self.observed)
            fresh = self.mode == "fresh" or self.factors is None or self.age >= self.refresh_every
            if fresh:
                endpoint, factors, alpha = sensitivity_factors(
                    self.model, source.detach(), self.settings["flow_steps"], **self.settings["dlr"],
                    seed=self.seed, products=self.products, seed_basis=self.previous_v)
                assert alpha == 0
                self.age, kind = 0, "fresh"
                endpoint_error = float((endpoint - self.image).norm() / self.image.norm().clamp_min(1e-20))
            elif self.mode == "secant":
                endpoint, factors = secant_factors(self.model, source.detach(), self.factors,
                    self.settings["flow_steps"], products=self.products)
                endpoint_error = float((endpoint - self.image).norm() / self.image.norm().clamp_min(1e-20))
            else:
                factors = transport_factors(self.model, self.factor_source,
                    source.detach() - self.factor_source, self.factors, self.settings["flow_steps"],
                    backend="exact" if self.mode == "transport" else "finite_difference", difference_step=1.)
            self.factors = tuple(value.detach().clone() for value in factors)
            self.factor_source = source.detach().clone()
            self.previous_v = self.factors[2]
            self.age += 1
            gradient = approx_source_gradient(self.factors, gx, 0.).detach()
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("Nonfinite source gradient")
        torch.cuda.synchronize(source.device)
        self.gradient_records.append({"kind": kind, "seconds": time.perf_counter() - started,
                                      "endpoint_relative_difference": endpoint_error})
        return gradient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("benchmark", "reconstruct"))
    parser.add_argument("--case", default="r64_n1_ct10/sample_000")
    parser.add_argument("--root", type=Path, default=Path("outputs/ct_mri_sweep_fp32_20261009"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", default=["exact", "fresh", "secant"],
                        choices=("exact", "proxy", "fresh", "secant", "transport", "finite_difference"))
    parser.add_argument("--optimizer", choices=("gd", "lbfgs"), default="lbfgs")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--refresh-every", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--displacement", type=float, default=.001,
                        help="Benchmark source displacement norm divided by sqrt(d)")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.85, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    settings = json.loads((args.root / "manifest.json").read_text())["settings"]
    case = args.root / args.case
    inputs = torch.load(case / "inputs.pt", map_location="cpu", weights_only=True)
    protocol = inputs["protocol"]
    job = protocol["job"]
    model_dir = Path(settings["run_template"].format(**job))
    model, _ = load_flow_model(job["resolution"], job["circles"], settings["epoch"], str(device), model_dir)
    operator = (CT(job["resolution"], job["level"], device) if job["problem"] == "ct" else
                MRI(job["resolution"], job["level"], protocol["operator"]["mask_seed"], device))
    if isinstance(operator, MRI):
        assert torch.equal(operator.mask.cpu(), inputs["mask"])
    initial, truth, clean, observed = (inputs[name].to(device) for name in ("initial", "truth", "clean", "observed"))
    assert torch.equal(clean, observed)
    tangent_unet.GN_STATS_MODE = "native" if job["resolution"] <= 96 else "manual"
    tangent_unet.GN_ACTION_BACKEND = "triton"
    tangent_unet.GN_PRIMAL_BACKEND = "full"
    enriched.SPECIALIZE_ZERO_ENRICHMENT = True
    enriched.SKIP_UNTRUNCATED_SVD = True
    products = GraphedAProducts(model, initial, share_linearization=True, full_step_graphs=True,
        max_jvp_columns=20, max_vjp_columns=16, max_step_jvp_columns=6, max_step_vjp_columns=4)
    args.out.mkdir(parents=True, exist_ok=True)
    record: Record = {"task": args.task, "case": args.case, "device": str(device),
        "settings": settings, "optimizer": args.optimizer, "max_attempts": args.steps,
        "refresh_every": args.refresh_every, "source_protocol": protocol,
        "torch": str(torch.__version__), "tf32": False,
        "inputs_sha256": hashlib.sha256((case / "inputs.pt").read_bytes()).hexdigest(),
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(Path(__file__).parent.glob("*.py"))},
        "methods": {}}
    write_json(args.out / "protocol.json", record)
    # Build derivative graphs before timed optimizer work for all modes.
    warm = TransportEvaluator(model, operator, observed, settings, protocol["dlr_seed"], products, "fresh")
    warm.gradient(initial)
    seed_factors = tuple(value.detach().clone() for value in warm.factors)
    del warm
    if args.task == "benchmark":
        direction = torch.randn(initial.shape, device=device,
            generator=torch.Generator(device=device).manual_seed(protocol["source_seed"] + 41))
        direction *= args.displacement * math.sqrt(initial.numel()) / direction.norm()
        source = (initial + direction).detach().requires_grad_(True)
        reference = TransportEvaluator(model, operator, observed, settings, protocol["dlr_seed"], products, "exact")
        exact = reference.gradient(source)
        record["displacement_norm"] = float(direction.norm())
        for mode in args.modes:
            measurements = []
            for repeat in range(args.repeats + 1):
                evaluator = TransportEvaluator(model, operator, observed, settings, protocol["dlr_seed"],
                    products, mode, args.refresh_every)
                evaluator.factors = tuple(value.clone() for value in seed_factors)
                evaluator.factor_source = initial.clone()
                evaluator.previous_v = seed_factors[2].clone()
                evaluator.age = 1
                gradient = evaluator.gradient(source)
                if repeat:
                    measurements.append({**evaluator.gradient_records[-1],
                        "relative_gradient_error": float((gradient - exact).norm() / exact.norm()),
                        "cosine": float((gradient.flatten() @ exact.flatten()) / (gradient.norm() * exact.norm()))})
                del evaluator
            record["methods"][mode] = measurements
            write_json(args.out / "benchmark.json", record)
            print(json.dumps({"mode": mode, "records": measurements}), flush=True)
        return

    for mode in args.modes:
        destination = args.out / mode
        destination.mkdir(exist_ok=True)
        source = initial.detach().clone().requires_grad_(True)
        evaluator = TransportEvaluator(model, operator, observed, settings, protocol["dlr_seed"],
            products, mode, args.refresh_every)
        options = {**settings["optimizer"], "project": False,
                   "step_cap": settings["optimizer"]["step_cap"] * math.sqrt(source.numel())}
        if args.optimizer == "gd":
            options["history_size"] = 0
        optimizer = ArmijoLBFGS(source, evaluator.objective, **options)

        def closure() -> torch.Tensor:
            source.grad = evaluator.gradient(source)
            return source.new_tensor(evaluator.value, dtype=torch.float64)

        evaluator.objective(source)
        history: list[Record] = [{"step": 0, "phi": evaluator.value,
            **metrics(evaluator.image, operator(evaluator.image), truth, clean, observed)}]
        started = time.perf_counter()
        status = "running"
        for attempt in range(1, args.steps + 1):
            previous = source.detach().clone()
            optimizer.step(closure)
            if not torch.equal(evaluator.source, source.detach()):
                evaluator.objective(source)
            displacement = float((source.detach() - previous).norm())
            assert displacement <= optimizer.step_cap * (1 + 1e-4)
            assert displacement == 0 or evaluator.value < history[-1]["phi"]
            history.append({"step": attempt, "phi": evaluator.value,
                "source_step_norm": displacement, "line_search_trials": optimizer.last_function_evaluations,
                "elapsed_seconds": time.perf_counter() - started,
                **metrics(evaluator.image, operator(evaluator.image), truth, clean, observed)})
            if optimizer.consecutive_failures >= 2:
                status = "stalled"
            elif attempt == args.steps:
                status = "step_limit"
            if attempt % 10 == 0 or status != "running":
                result = {"mode": mode, "case": args.case, "optimizer": args.optimizer,
                    "status": status, "history": history, "gradient_records": evaluator.gradient_records,
                    "evaluations": evaluator.counts, "elapsed_seconds": time.perf_counter() - started}
                write_json(destination / "result.json", result)
                save_tensor(destination / "state.pt", {**result, "source": source.detach(),
                    "image": evaluator.image, "measurement": operator(evaluator.image).detach(),
                    "factors": evaluator.factors, "factor_source": evaluator.factor_source,
                    "factor_age": evaluator.age, "optimizer_state": {key: value for key, value in vars(optimizer).items()
                        if key not in {"source", "objective"}}})
                print(json.dumps({"case": args.case, "mode": mode, "attempt": attempt,
                    "status": status, "image_percent": 100 * history[-1]["image_relative_l2"],
                    "residual_percent": 100 * history[-1]["measurement_relative_l2_clean"]}), flush=True)
            if status != "running":
                break
        record["methods"][mode] = result
        write_json(args.out / "reconstruction.json", record)


if __name__ == "__main__":
    main()
