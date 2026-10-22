"""Separate dynamic-basis error from terminal low-rank approximation error."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from unittest.mock import patch
from functools import partial

import torch

from dlr_dflow.api import approx_source_gradient, load_flow_model, sensitivity_factors
from dlr_dflow.implementation import enriched, tangent_unet
from dlr_dflow.implementation.graphed_products import GraphedAProducts
from experiments.ct_mri_sweep import run
from experiments.ct_mri_sweep.operators import CT, MRI
from experiments.ct_mri_sweep.validation import ChunkedAutodiffProducts
from sampler.ode_samplers import euler_flow
from experiments.separated_shapes_117k.ksl_stability import increment_ksl_step, stable_sl_step, refined_increment_step, pure_ksl_step
from experiments.separated_shapes_117k.exact_suffix import task_terminal_factors


class DiscreteProducts:
    """Native autodiff products of the actual discrete Euler map, in small blocks."""

    def __init__(self, model: torch.nn.Module, source: torch.Tensor, steps: int) -> None:
        self.local = ChunkedAutodiffProducts(model, columns=4)
        self.h = 1.0 / steps
        self.points: list[torch.Tensor] = []
        with torch.no_grad():
            image = source.detach()
            for index in range(steps):
                self.points.append(image)
                image = image + self.h * model(image, image.new_tensor([index * self.h]))
        self.image = image

    @torch.no_grad()
    def apply(self, values: torch.Tensor, transpose: bool = False) -> torch.Tensor:
        indices = range(len(self.points) - 1, -1, -1) if transpose else range(len(self.points))
        result = values
        for index in indices:
            self.local.set_state(index * self.h, self.points[index])
            action = self.local.AT(result) if transpose else self.local.A(result)
            result = result + self.h * action
        return result


def scores(estimate: torch.Tensor, exact: torch.Tensor, basis: torch.Tensor,
           terminal: torch.Tensor) -> dict[str, float]:
    projected = basis @ (basis.T @ exact)
    norm = exact.norm()
    proxy = terminal.reshape_as(exact)
    coefficient = (proxy * (exact - estimate)).sum() / proxy.square().sum()
    return dict(relative_error=float((estimate - exact).norm() / norm),
                subspace_relative_error=float((projected - exact).norm() / norm),
                coefficient_relative_error=float((estimate - projected).norm() / norm),
                cosine=float(torch.nn.functional.cosine_similarity(estimate.flatten(), exact.flatten(), dim=0)),
                norm_ratio=float(estimate.norm() / norm),
                oracle_scalar=float(coefficient),
                oracle_scalar_corrected_error=float((estimate + coefficient * proxy - exact).norm() / norm))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ranks", nargs="+", type=int, default=[32, 128])
    parser.add_argument("--power-iterations", type=int, default=3)
    parser.add_argument("--reference-oversampling", type=int, default=8,
        help="Extra columns for the independent terminal SVD reference, not DLR initialization")
    parser.add_argument("--basis-refreshes", type=int, default=0,
        help="Repeatedly reuse the returned right basis at an unchanged source")
    integrator = parser.add_mutually_exclusive_group()
    integrator.add_argument("--stable-sl", action="store_true")
    integrator.add_argument("--increment-ksl", action="store_true")
    integrator.add_argument("--refined-increment", action="store_true")
    integrator.add_argument("--pure-ksl", action="store_true",
        help="Retain the notes' L-step core instead of projecting the Euler target")
    integrator.add_argument("--stable-ksl", action="store_true",
        help="Stable frozen S/L integration retaining its L-step core")
    parser.add_argument("--factor-arithmetic", choices=("specialized", "eager_specialized", "direct", "canonical"),
        default="specialized", help="Compare equivalent fixed-rank arithmetic without changing the model or rank")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--terminal-seed", action="store_true")
    parser.add_argument("--line-search-audit", action="store_true")
    parser.add_argument("--saved-direction-audit", action="store_true",
        help="Probe both signs of the saved optimizer gradient using exact objective slopes")
    parser.add_argument("--repeat-gradients", type=int, default=0,
        help="Repeat native exact gradients at one unchanged source to measure arithmetic variation")
    parser.add_argument("--suffix-audit", action="store_true",
        help="Measure DLR-prefix errors with exact adjoints through the remaining Euler steps")
    parser.add_argument("--exact-only", action="store_true")
    parser.add_argument("--double", action="store_true")
    parser.add_argument("--manifest", type=Path,
        default=Path("outputs/separated_shapes_117k_noiseless_20261021/production/manifest.json"))
    args = parser.parse_args()
    if args.reference_oversampling < 1:
        parser.error("The reference needs an extra singular value to assess the rank cutoff")
    if args.saved_direction_audit and (args.source is None or args.double):
        parser.error("Saved-direction audit requires a saved FP32 source")
    if args.factor_arithmetic != "specialized" and (args.stable_sl or args.increment_ksl or args.refined_increment or args.pure_ksl or args.stable_ksl):
        parser.error("Factor arithmetic controls require the original integrator")
    if args.basis_refreshes < 0 or (args.basis_refreshes and (args.stable_sl or args.increment_ksl or args.refined_increment or args.pure_ksl or args.stable_ksl or args.terminal_seed)):
        parser.error("Basis refresh diagnostics require a nonnegative count and the original integrator")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda:1")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.2, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    inputs = torch.load(args.inputs, map_location=device, weights_only=True)
    saved = torch.load(args.source, map_location=device, weights_only=True) if args.source else None
    source = inputs["initial"] if saved is None else saved["source"]
    protocol = inputs["protocol"]
    settings = (saved["protocol"].get("continuation_settings", saved["protocol"].get("settings"))
                if saved is not None else None)
    if settings is None:
        settings = json.loads(args.manifest.read_text())["settings"]
    resolution = protocol["job"]["resolution"]
    folder = Path(settings["run_template"].format(resolution=resolution))
    assert hashlib.sha256((folder / "checkpoints/epoch_0500.pt").read_bytes()).hexdigest() == protocol["checkpoint_sha256"]
    if saved is not None:
        assert saved["protocol"]["checkpoint_sha256"] == protocol["checkpoint_sha256"]
        assert saved["protocol"]["operator"] == protocol["operator"]
    model, _ = load_flow_model(resolution, 0, 500, str(device), folder)
    info = protocol["operator"]
    operator = (CT(resolution, protocol["job"]["level"], device) if protocol["job"]["problem"] == "ct"
                else MRI(resolution, 4, info["mask_seed"], device, info["pattern"]))
    if isinstance(operator, MRI):
        assert torch.equal(inputs["mask"], operator.mask)
    if args.double:
        if not args.exact_only:
            parser.error("The double-precision control is restricted to exact-gradient line probes")
        model = model.double()
        # The training embedding intentionally computes FP32 sin/cos; preserve
        # those features and promote them for the double-precision linear layer.
        model.time_proj.register_forward_pre_hook(lambda module, arguments: (arguments[0].double(),))
        source = source.double()
        inputs["truth"] = inputs["truth"].double()
        inputs["observed"] = inputs["observed"].to(torch.complex128 if inputs["observed"].is_complex() else torch.float64)
        if isinstance(operator, CT):
            # Preserve the recorded geometry; only promote its existing values.
            operator.grids = tuple(grid.double() for grid in operator.grids)
    variable = source.detach().requires_grad_(True)
    image = euler_flow(model, variable, settings["flow_steps"])
    loss = .5 * (operator(image) - inputs["observed"]).abs().square().sum(dtype=torch.float64)
    exact, = torch.autograd.grad(loss, variable)
    _, terminal = run.endpoint_gradient(image.detach(), operator, inputs["observed"])
    exact = exact.reshape(-1, 1)
    reference = DiscreteProducts(model, source, settings["flow_steps"])
    torch.testing.assert_close(reference.image, image.detach(), rtol=0, atol=0)
    independent = reference.apply(terminal.reshape(-1, 1), transpose=True)
    error = float((independent - exact).norm() / exact.norm())
    if error > 1e-3:
        raise RuntimeError(f"Full-map adjoint fails exact-gradient consistency: {error}")
    report: dict[str, Any] = dict(inputs=str(args.inputs), source=str(args.source), dtype=str(source.dtype),
        source_sha256=hashlib.sha256(source.cpu().numpy().tobytes()).hexdigest(),
        checkpoint_sha256=protocol["checkpoint_sha256"], objective=float(loss.detach()),
        image_error=float((image.detach()-inputs["truth"]).norm()/inputs["truth"].norm()),
        exact_gradient_norm=float(exact.norm()), adjoint_relative_error=error, rows=[],
        note="Fixed-source diagnostic. Terminal randomized references are not certified truncated SVDs.")
    if args.repeat_gradients:
        repeats = []
        for _ in range(args.repeat_gradients):
            repeated_image = euler_flow(model, variable, settings["flow_steps"])
            repeated_loss = .5*(operator(repeated_image)-inputs["observed"]).abs().square().sum(dtype=torch.float64)
            repeated_gradient, = torch.autograd.grad(repeated_loss, variable)
            repeats.append(dict(image_bitwise_equal=torch.equal(repeated_image.detach(), image.detach()),
                objective_change=float(repeated_loss.detach())-report["objective"],
                gradient_change_over_exact_norm=float((repeated_gradient.reshape_as(exact)-exact).norm()/exact.norm())))
        report["repeated_native_gradients"] = repeats

    @torch.no_grad()
    def line_trials(gradient: torch.Tensor) -> list[dict[str, Any]]:
        direction = -gradient.reshape_as(source) / gradient.norm()
        cap = settings["optimizer"]["step_cap"] * math.sqrt(source.numel())
        records = []
        for index in range(30):
            displacement = cap * 2.0 ** -index
            candidate = source + displacement * direction
            if torch.equal(candidate, source):
                break
            candidate_image = euler_flow(model, candidate, settings["flow_steps"])
            candidate_loss = .5 * (operator(candidate_image)-inputs["observed"]).abs().square().sum(dtype=torch.float64)
            change = float(candidate_loss) - report["objective"]
            records.append(dict(displacement=displacement, change=change,
                linear_prediction=displacement*float((exact.reshape_as(source)*direction).sum()),
                armijo=change < 0 and change <= -settings["optimizer"]["c1"]*displacement*float(gradient.norm()),
                true_slope_armijo=(float((exact.reshape_as(source)*direction).sum()) < 0 and change < 0
                    and change <= settings["optimizer"]["c1"]*displacement*float((exact.reshape_as(source)*direction).sum())),
                image_error=float((candidate_image-inputs["truth"]).norm()/inputs["truth"].norm())))
        return records

    if args.line_search_audit:
        report["exact_gradient_trials"] = line_trials(exact)
    if args.saved_direction_audit:
        stored = saved["optimizer"]["g"].reshape_as(source)
        report["saved_gradient_cosine"] = float(torch.nn.functional.cosine_similarity(
            stored.flatten(), exact.flatten(), dim=0))
        report["saved_gradient_trials"] = line_trials(stored)
        report["reversed_saved_gradient_trials"] = line_trials(-stored)
        report["saved_optimizer_step"] = saved["optimizer"]["step_attempts"]
    if args.exact_only:
        report["status"] = "complete"
        run.write_json(args.out, report)
        return
    tangent_unet.GN_STATS_MODE = "native"
    tangent_unet.GN_ACTION_BACKEND = "triton"
    tangent_unet.GN_PRIMAL_BACKEND = "full"
    enriched.SPECIALIZE_ZERO_ENRICHMENT = args.factor_arithmetic in ("specialized", "eager_specialized")
    products = GraphedAProducts(model, source, share_linearization=True,
        full_step_graphs=not (args.stable_sl or args.stable_ksl) and args.factor_arithmetic == "specialized",
        max_jvp_columns=20, max_vjp_columns=16, max_step_jvp_columns=6, max_step_vjp_columns=4)
    for rank in args.ranks:
        for warm in (False, True) if saved is not None and saved.get("previous_v") is not None else (False,):
            options = {**settings["dlr"], "rank": rank}
            prefix_factors: dict[int, tuple[float, tuple[torch.Tensor, ...]]] = {}
            if args.suffix_audit:
                options["factor_callback"] = lambda step, alpha, factors: prefix_factors.update({
                    step: (float(alpha), tuple(value.detach().clone() for value in factors))})
            if args.factor_arithmetic == "canonical":
                options.update(adaptive_rank=True, rank_min=rank, rank_max=rank)
            seed_basis = saved["previous_v"] if warm else None
            if args.terminal_seed:
                seed_basis = (terminal.reshape(-1, 1) if seed_basis is None else
                              torch.cat([terminal.reshape(-1, 1), seed_basis], dim=1))
            with patch.object(enriched, "_zero_enrichment_step",
                              partial(stable_sl_step, recompress=False) if args.stable_ksl else
                              pure_ksl_step if args.pure_ksl else stable_sl_step if args.stable_sl else
                              refined_increment_step if args.refined_increment else
                              increment_ksl_step if args.increment_ksl else enriched._zero_enrichment_step):
                endpoint, factors, alpha = sensitivity_factors(model, source, settings["flow_steps"],
                    **options, seed=protocol["dlr_seed"], products=products,
                    seed_basis=seed_basis)
            estimate = approx_source_gradient(factors, terminal, alpha).reshape(-1, 1)
            if not warm:
                fresh_estimate = estimate.clone()
            row = dict(method="stable_ksl" if args.stable_ksl else "pure_ksl" if args.pure_ksl else "stable_sl" if args.stable_sl else "refined_increment" if args.refined_increment else "increment_ksl" if args.increment_ksl else "dynamic", rank=rank, warm=warm,
                       factor_arithmetic=args.factor_arithmetic,
                       terminal_seed=args.terminal_seed, **scores(estimate, exact, factors[2], terminal),
                       endpoint_relative_error=float((endpoint-image.detach()).norm()/image.detach().norm()))
            report["rows"].append(row)
            if args.suffix_audit:
                suffix = terminal.reshape(-1, 1)
                suffix_rows = []
                adjoints = {}
                for prefix in range(settings["flow_steps"], -1, -1):
                    adjoints[prefix] = suffix.detach().clone()
                    baseline, (left, core, right) = prefix_factors[prefix]
                    hybrid = baseline*suffix + right @ (core.T @ (left.T @ suffix))
                    suffix_rows.append(dict(prefix_steps=prefix, suffix_steps=settings["flow_steps"]-prefix,
                        relative_error=float((hybrid-exact).norm()/exact.norm()),
                        cosine=float(torch.nn.functional.cosine_similarity(hybrid.flatten(), exact.flatten(), dim=0)),
                        norm_ratio=float(hybrid.norm()/exact.norm())))
                    if prefix:
                        reference.local.set_state((prefix-1)*reference.h, reference.points[prefix-1])
                        suffix = suffix + reference.h*reference.local.AT(suffix)
                row["exact_suffix_diagnostics"] = suffix_rows
                baseline, prefix = prefix_factors[settings["flow_steps"]-1]
                products.set_state((settings["flow_steps"]-1)*reference.h, reference.points[-1])
                left, core, right = task_terminal_factors(prefix, terminal, reference.h, products)
                task_gradient = right @ (core.T @ (left.T @ terminal.reshape(-1, 1)))
                reference.local.set_state((settings["flow_steps"]-1)*reference.h, reference.points[-1])
                pullback = terminal.reshape(-1, 1)
                pullback = pullback + reference.h*reference.local.AT(pullback)
                left_prefix, core_prefix, right_prefix = prefix
                hybrid = baseline*pullback + right_prefix @ (core_prefix.T @ (left_prefix.T @ pullback))
                row["task_terminal_equivalence"] = dict(rank=core.shape[0],
                    relative_error=float((task_gradient-exact).norm()/exact.norm()),
                    difference_from_exact_suffix_over_exact_norm=float((task_gradient-hybrid).norm()/exact.norm()))
                transported = []
                for count in sorted({1, min(2, settings["flow_steps"]-1), settings["flow_steps"]-1}):
                    cut = settings["flow_steps"]-count
                    baseline, task_factors = prefix_factors[cut]
                    left_prefix, core_prefix, right_prefix = task_factors
                    expected = baseline*adjoints[cut] + right_prefix @ (core_prefix.T @ (left_prefix.T @ adjoints[cut]))
                    for index in range(cut, settings["flow_steps"]):
                        products.set_state(index*reference.h, reference.points[index])
                        task_factors = task_terminal_factors(task_factors, adjoints[index+1],
                            reference.h, products, alpha=1.-index*reference.h)
                    left, core, right = task_factors
                    estimate_task = right @ (core.T @ (left.T @ terminal.reshape(-1, 1)))
                    transported.append(dict(goal_steps=count, rank=core.shape[0],
                        relative_error=float((estimate_task-exact).norm()/exact.norm()),
                        difference_from_exact_suffix_over_exact_norm=float((estimate_task-expected).norm()/exact.norm())))
                row["transported_goal_equivalence"] = transported
            if args.line_search_audit:
                row["line_trials"] = line_trials(estimate)
            run.write_json(args.out, report)
            print(json.dumps({key: value for key, value in row.items() if key != "line_trials"}), flush=True)
        for refresh in range(1, args.basis_refreshes + 1):
            previous = estimate
            endpoint, factors, alpha = sensitivity_factors(model, source, settings["flow_steps"],
                **options, seed=protocol["dlr_seed"], products=products, seed_basis=factors[2])
            estimate = approx_source_gradient(factors, terminal, alpha).reshape(-1, 1)
            row = dict(method="fixed_source_basis_refresh", rank=rank, refresh=refresh,
                factor_arithmetic=args.factor_arithmetic, **scores(estimate, exact, factors[2], terminal),
                change_over_exact_norm=float((estimate-previous).norm()/exact.norm()),
                change_over_previous_norm=float((estimate-previous).norm()/previous.norm()),
                endpoint_relative_error=float((endpoint-image.detach()).norm()/image.detach().norm()))
            report["rows"].append(row)
            run.write_json(args.out, report)
            print(json.dumps(row), flush=True)
        if args.basis_refreshes:
            endpoint, factors, alpha = sensitivity_factors(model, source, settings["flow_steps"],
                **options, seed=protocol["dlr_seed"], products=products, seed_basis=None)
            estimate = approx_source_gradient(factors, terminal, alpha).reshape(-1, 1)
            row = dict(method="fixed_source_fresh_repeat", rank=rank,
                change_over_exact_norm=float((estimate-fresh_estimate).norm()/exact.norm()),
                change_over_previous_norm=float((estimate-fresh_estimate).norm()/fresh_estimate.norm()),
                endpoint_relative_error=float((endpoint-image.detach()).norm()/image.detach().norm()))
            report["rows"].append(row)
            run.write_json(args.out, report)
            print(json.dumps(row), flush=True)
    del products, factors, endpoint, estimate
    torch.cuda.empty_cache()
    if args.skip_reference:
        report["status"] = "complete"
        run.write_json(args.out, report)
        return
    generator = torch.Generator(device=device).manual_seed(protocol["dlr_seed"])
    report["reference_oversampling"] = args.reference_oversampling
    omega = torch.randn(source.numel(), max(args.ranks) + args.reference_oversampling,
                        device=device, generator=generator)
    with torch.no_grad():
        basis, _ = torch.linalg.qr(reference.apply(omega), mode="reduced")
        previous_estimates: dict[int, torch.Tensor] = {}
        for power in range(args.power_iterations + 1):
            transpose = reference.apply(basis, transpose=True)
            right, values, coordinates = enriched._thin_svd(transpose)
            left = basis @ coordinates.T
            for rank in args.ranks:
                estimate = right[:, :rank] @ (values[:rank, None] * (left[:, :rank].T @ terminal.reshape(-1, 1)))
                row = dict(method="terminal_randomized", rank=rank, power_iterations=power,
                           **scores(estimate, exact, right[:, :rank], terminal),
                           singular_values=values[:rank].cpu().tolist(),
                           gradient_change_over_exact_norm=(float((estimate-previous_estimates[rank]).norm()/exact.norm())
                                                            if rank in previous_estimates else None))
                previous_estimates[rank] = estimate
                report["rows"].append(row)
                run.write_json(args.out, report)
                print(json.dumps({key: value for key, value in row.items() if key != "singular_values"}), flush=True)
            if power < args.power_iterations:
                right_basis, _ = torch.linalg.qr(transpose, mode="reduced")
                basis, _ = torch.linalg.qr(reference.apply(right_basis), mode="reduced")
        residual = reference.apply(right[:, :max(args.ranks)]) - left[:, :max(args.ranks)] * values[:max(args.ranks)]
        transpose_residual = (reference.apply(left[:, :max(args.ranks)], transpose=True)
                              - right[:, :max(args.ranks)] * values[:max(args.ranks)])
        report["max_triplet_residual_over_leading_value"] = float(residual.norm(dim=0).max()/values[0])
        coefficients = values[:max(args.ranks), None] * (left[:, :max(args.ranks)].T @ terminal.reshape(-1, 1))
        report["reference_gradient_energy_curve"] = (coefficients.square().cumsum(dim=0)
            / exact.square().sum()).flatten().cpu().tolist()
        report["reference_checks"] = []
        for rank in args.ranks:
            triplet_error = float(torch.maximum(residual[:, :rank].norm(dim=0).max(),
                transpose_residual[:, :rank].norm(dim=0).max())/values[0])
            change = next(row["gradient_change_over_exact_norm"] for row in reversed(report["rows"])
                          if row["method"] == "terminal_randomized" and row["rank"] == rank)
            gap = float((values[rank-1]-values[rank])/values[0])
            report["reference_checks"].append(dict(rank=rank,
                max_triplet_residual_over_leading_value=triplet_error,
                gradient_change_over_exact_norm=change,
                cutoff_gap_over_leading_value=gap,
                cutoff_unresolved=gap <= 2*triplet_error,
                numerical_checks_passed=(triplet_error < 1e-3 and change is not None
                                         and change < .01 and gap > 2*triplet_error)))
    report["status"] = "complete"
    run.write_json(args.out, report)


if __name__ == "__main__":
    main()
