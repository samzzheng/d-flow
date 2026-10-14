"""CPU Jacobian spectra and DLR errors along native Euler flow time."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable

import torch

from dlr_dflow.api import Factors, load_flow_model, sensitivity_factors
from dlr_dflow.implementation import enriched
from experiments.ct_mri_sweep.operators import CT, MRI
from experiments.ct_mri_sweep.run import write_json
from p_flow import endpoint_gradient
from sampler.ode_samplers import euler_flow

Tensor = torch.Tensor
Product = Callable[[Tensor], Tensor]
Record = dict[str, Any]
OUT = Path("outputs/noise_regularization_analysis_20261014/flow_time")


def approximation_spectrum(alpha: float, factors: Factors) -> Tensor:
    """Exact spectrum of alpha I + USV^T using its <=2r-dimensional block."""
    u, s, v = (factor.double() for factor in factors)
    dimension = u.shape[0]
    if not s.numel():
        return torch.full((dimension,), abs(alpha), dtype=torch.float64)
    basis = torch.linalg.qr(torch.cat((u, v), dim=1), mode="reduced")[0]
    block = alpha * torch.eye(basis.shape[1], dtype=torch.float64) + (basis.T @ u) @ s @ (v.T @ basis)
    return torch.cat((torch.linalg.svdvals(block),
                      torch.full((dimension - basis.shape[1],), abs(alpha), dtype=torch.float64))).sort(descending=True).values


def prefix_products(model: torch.nn.Module, source: Tensor, completed: int,
                    steps: int) -> tuple[Tensor, Product, Product]:
    """Keep dt=1/steps and the original times; never rescale a prefix to [0,1]."""
    def flow(z: Tensor) -> Tensor:
        dt = 1.0 / steps
        for index in range(completed):
            z = z + dt * model(z, z.new_full((1,), index * dt))
        return z

    endpoint, transpose = torch.func.vjp(flow, source)

    def forward(values: Tensor) -> Tensor:
        return torch.cat([
            torch.func.vmap(lambda q: torch.func.jvp(flow, (source,), (q,))[1])(
                block.T.reshape(-1, *source.shape)).reshape(block.shape[1], -1).T
            for block in values.split(16, dim=1)], dim=1).detach()

    def backward(values: Tensor) -> Tensor:
        return torch.cat([
            torch.func.vmap(lambda q: transpose(q)[0])(
                block.T.reshape(-1, *source.shape)).reshape(block.shape[1], -1).T
            for block in values.split(16, dim=1)], dim=1).detach()

    return endpoint.detach(), forward, backward


def measure(model: torch.nn.Module, source: Tensor, completed: int, steps: int,
            alpha: float, factors: Factors, operator: CT | MRI, inputs: Record,
            width: int, probes: int, refinements: int, seed: int) -> Record:
    started = time.monotonic()
    dimension = source.numel()
    endpoint, j, jt = prefix_products(model, source, completed, steps)
    u, s, v = factors

    def approximate(values: Tensor) -> Tensor:
        return alpha * values + u @ (s @ (v.T @ values))

    def approximate_transpose(values: Tensor) -> Tensor:
        return alpha * values + v @ (s.T @ (u.T @ values))

    generator = torch.Generator().manual_seed(seed)
    omega = torch.randn(dimension, probes, generator=generator)
    heldout = j(omega)
    estimated = approximate(omega)
    adjoint_probe = torch.randn(dimension, 1, generator=generator)
    transpose_probe = jt(adjoint_probe)
    adjoint_error = float(abs((adjoint_probe.double() * heldout[:, :1].double()).sum() -
                              (transpose_probe.double() * omega[:, :1].double()).sum()) /
                          (adjoint_probe.double().norm() * heldout[:, :1].double().norm() +
                           transpose_probe.double().norm() * omega[:, :1].double().norm()))
    assert adjoint_error < 1e-5, adjoint_error
    stage_rows = []
    if completed:
        q = torch.linalg.qr(j(torch.randn(dimension, width, generator=generator)), mode="reduced")[0]
        for refinement in range(refinements):
            b = jt(q)
            reference_v64, singular, coordinates = torch.linalg.svd(b.double(), full_matrices=False)
            reference_v = reference_v64.float()
            reference_u = q @ coordinates.T.float()
            triplet_errors = (j(reference_v) - reference_u * singular.float()).double().norm(dim=0) / singular[0]
            outside = heldout.double() - q.double() @ (q.double().T @ heldout.double())
            energies = outside.square().sum(dim=0)
            tail, tail_se = float(energies.mean()), float(energies.std() / math.sqrt(probes))
            total = float(singular.square().sum()) + tail
            ranks: dict[str, int | None] = {}
            for target in (.99, .999):
                candidates = torch.where(singular.square().cumsum(0) >= target * total)[0]
                ranks[str(target)] = int(candidates[0]) + 1 if len(candidates) else None
            stage_rows.append({"refinement": refinement, "singular_values": singular.tolist(),
                               "triplet_relative_residuals": triplet_errors.tolist(),
                               "tail_energy_estimate": tail, "tail_energy_standard_error": tail_se,
                               "energy_rank_estimates": ranks})
            if refinement + 1 < refinements:
                q = torch.linalg.qr(j(torch.linalg.qr(b, mode="reduced")[0]), mode="reduced")[0]
        rank_status = {}
        for target, rank in ranks.items():
            stable = len(stage_rows) > 1 and stage_rows[-2]["energy_rank_estimates"][target] == rank
            boundary_stable = False
            if rank is not None:
                bounds = []
                for omitted in (max(0., tail - 2 * tail_se), tail + 2 * tail_se):
                    candidates = torch.where(singular.square().cumsum(0) >= float(target) *
                                             (float(singular.square().sum()) + omitted))[0]
                    bounds.append(int(candidates[0]) + 1 if len(candidates) else None)
                boundary_stable = bounds == [rank, rank]
            rank_status[target] = "resolved estimate" if (rank is not None and stable and boundary_stable and
                                                          float(triplet_errors[:rank].max()) < 1e-3) else "unresolved at budget"
        principal_cosines = torch.linalg.svdvals(reference_v[:, :3].double().T @ v.double())
        right_subspace_error = float((reference_v[:, :3].double() - v.double() @
                                     (v.double().T @ reference_v[:, :3].double())).norm() / math.sqrt(3))
    else:
        singular = torch.ones(dimension, dtype=torch.float64)
        total, tail, tail_se = float(dimension), 0., 0.
        ranks = {str(target): math.ceil(target * dimension) for target in (.99, .999)}
        rank_status = dict.fromkeys(ranks, "exact identity")
        principal_cosines, right_subspace_error = torch.tensor([]), None
        triplet_errors = torch.zeros(3)
    approximation_singular = approximation_spectrum(alpha, factors)
    approximation_energy = approximation_singular.square().sum()
    approx_ranks = {str(target): int(torch.searchsorted(approximation_singular.square().cumsum(0),
                                                      target * approximation_energy)) + 1 for target in (.99, .999)}
    sample_error_energy = (estimated.double() - heldout.double()).square().sum(dim=0)
    sample_true_energy = heldout.double().square().sum(dim=0)
    squared_relative_error = float(sample_error_energy.mean() / sample_true_energy.mean())
    error_se = float((sample_error_energy - squared_relative_error * sample_true_energy).std() /
                     math.sqrt(probes) / sample_true_energy.mean())
    measurement_true = torch.stack([operator(column.reshape_as(source)).flatten() for column in heldout.T])
    measurement_approx = torch.stack([operator(column.reshape_as(source)).flatten() for column in estimated.T])
    _, clean_terminal = endpoint_gradient(endpoint, operator, inputs["clean"])
    _, noise_terminal = endpoint_gradient(torch.zeros_like(endpoint), operator, inputs["observed"] - inputs["clean"])
    terminals = torch.stack((clean_terminal.flatten(), noise_terminal.flatten()), dim=1)
    exact_gradients, approximate_gradients = jt(terminals).double(), approximate_transpose(terminals).double()
    exact_clean, exact_noise = exact_gradients.unbind(dim=1)
    approx_clean, approx_noise = approximate_gradients.unbind(dim=1)
    exact_noisy, approx_noisy = exact_clean + exact_noise, approx_clean + approx_noise
    result = {
        "completed_steps": completed, "time": completed / steps, "alpha": alpha,
        "native_sigma_max_estimate": float(singular[0]), "native_frobenius_squared_estimate": total,
        "native_stable_rank_estimate": total / float(singular[0].square()),
        "native_energy_rank_estimates": ranks, "native_energy_rank_status": rank_status,
        "native_top8_energy_fraction_estimate": float(singular[:8].square().sum()) / total,
        "native_top3_triplet_relative_residual": float(triplet_errors[:3].max()),
        "approximation_sigma_max": float(approximation_singular[0]),
        "approximation_stable_rank": float(approximation_energy / approximation_singular[0].square()),
        "approximation_energy_ranks": approx_ranks,
        "approximation_top8_energy_fraction": float(approximation_singular[:8].square().sum() / approximation_energy),
        "full_relative_frobenius_error_estimate": math.sqrt(squared_relative_error),
        "full_relative_frobenius_error_two_se_interval": [math.sqrt(max(0., squared_relative_error - 2 * error_se)),
                                                          math.sqrt(squared_relative_error + 2 * error_se)],
        "measurement_relative_frobenius_error_estimate": float((measurement_approx - measurement_true).norm() /
                                                               measurement_true.norm()),
        "right_top3_containment_error": right_subspace_error,
        "right_top3_principal_cosines": principal_cosines.tolist(),
        "noise_gradient_norm_ratio": float(approx_noise.norm() / exact_noise.norm()),
        "noisy_gradient_relative_error": float((approx_noisy - exact_noisy).norm() / exact_noisy.norm()),
        "noisy_gradient_cosine": float(approx_noisy @ exact_noisy / (approx_noisy.norm() * exact_noisy.norm())),
        "clean_gradient_R": float((approx_noisy - exact_clean).norm() / exact_noise.norm()),
        "gradient_interpretation": "Loss on A(x_t); only t=1 is the actual reconstruction loss",
        "adjoint_scaled_error": adjoint_error, "spectral_stages": stage_rows,
        "seconds": time.monotonic() - started,
    }
    if completed == steps:
        canonical = euler_flow(model, source, steps).detach()
        assert torch.allclose(endpoint, canonical, atol=1e-5, rtol=1e-5)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--sources", nargs="+", default=["initial", "d_flow_final", "dlr_final"])
    parser.add_argument("--case", choices=["ct", "mri"], default="ct")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--probes", type=int, default=24)
    parser.add_argument("--refinements", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--refine-existing", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    enriched.SPECIALIZE_ZERO_ENRICHMENT = True
    # Validate the small-block spectrum against a dense non-symmetric example.
    generator = torch.Generator().manual_seed(401)
    u, v = [torch.linalg.qr(torch.randn(24, 3, generator=generator, dtype=torch.float64))[0] for _ in range(2)]
    s = torch.randn(3, 3, generator=generator, dtype=torch.float64)
    for alpha in (1., .2, 0.):
        expected = torch.linalg.svdvals(alpha * torch.eye(24) + u @ s @ v.T)
        torch.testing.assert_close(approximation_spectrum(alpha, (u, s, v)), expected, atol=1e-7, rtol=1e-7)
    model_path = Path("outputs/fm_unet/circles-64-noattn-ch16-b8-e200/n1")
    model, _ = load_flow_model(64, 1, 200, "cpu", model_path)
    if args.refine_existing:
        for path in sorted(OUT.glob(f"{args.case}_*_steps{args.steps}.json")):
            record = json.loads(path.read_text())
            directory = Path(record["source_directory"])
            saved = torch.load(directory / ("inputs.pt" if record["source_kind"] == "initial" else
                               "d_flow.pt" if record["source_kind"] == "d_flow_final" else "dlr_d_flow.pt"),
                               map_location="cpu", weights_only=True)
            source = saved["initial" if record["source_kind"] == "initial" else "source"]
            for point in record["points"]:
                if not point["completed_steps"] or point.get("leading_norm_refinement", {}).get("relative_triplet_residual", 1.) <= 1e-3:
                    continue
                residual = point["spectral_stages"][-1]["triplet_relative_residuals"][0]
                if residual <= 1e-3:
                    continue
                _, j, jt = prefix_products(model, source, point["completed_steps"], args.steps)
                direction = torch.randn(source.numel(), 1, generator=torch.Generator().manual_seed(851014))
                direction /= direction.norm()
                for iteration in range(256):
                    image = j(direction)
                    sigma = float(image.norm())
                    backward = jt(image / sigma)
                    residual = float((backward - sigma * direction).norm() / sigma)
                    if residual < 1e-4:
                        break
                    direction = backward / backward.norm()
                previous = point["native_sigma_max_estimate"]
                point["leading_norm_refinement"] = {
                    "previous_sigma_max_estimate": previous, "power_sigma_estimate": sigma,
                    "relative_triplet_residual": residual, "iterations": iteration + 1,
                    "leading_only": True, "other_spectral_cutoffs_unchanged": True,
                }
                if sigma >= previous:
                    point["native_sigma_max_estimate"] = sigma
                    point["native_stable_rank_estimate"] = point["native_frobenius_squared_estimate"] / sigma ** 2
                write_json(path, record)
                print(json.dumps({"file": str(path), "time": point["time"], "sigma": sigma,
                                  "residual": residual, "iterations": iteration + 1}), flush=True)
            terminal = record["points"][-1]
            if terminal["time"] == 1 and "terminal_frobenius_refinement" not in terminal:
                _, factors, alpha = sensitivity_factors(
                    model, source, args.steps, **record["dlr_settings"], seed=saved["protocol"]["dlr_seed"],
                    seed_basis=saved.get("previous_v") if record["source_kind"] == "dlr_final" else None)
                assert alpha == 0
                u, s, v = factors
                _, j, jt = prefix_products(model, source, args.steps, args.steps)
                jv, jtu = j(v).double(), jt(u).double()
                us = u.double() @ s.double()
                cross = float((us * jv).sum())
                offset = float(s.double().square().sum()) - 2 * cross
                total = terminal["native_frobenius_squared_estimate"]
                tail_se = terminal["spectral_stages"][-1]["tail_energy_standard_error"]
                terminal["terminal_frobenius_refinement"] = {
                    "formula": "||J-B||F^2 = ||J||F^2 + ||S||F^2 - 2<US,JV>; alpha=0",
                    "previous_probe_estimate": terminal["full_relative_frobenius_error_estimate"],
                    "J_inner_product_B": cross, "B_frobenius_squared": float(s.double().square().sum()),
                    "relative_left_range_loss": math.sqrt(max(0., total - float(jtu.square().sum())) / total),
                    "relative_right_range_loss": math.sqrt(max(0., total - float(jv.square().sum())) / total),
                    "relative_within_right_subspace_error": float((us - jv).norm()) / math.sqrt(total),
                    "uncertainty": "Only the independently probed omitted Frobenius energy remains stochastic; spectral numerical error is separate",
                }
                terminal["full_relative_frobenius_error_estimate"] = math.sqrt(max(0., (total + offset) / total))
                terminal["full_relative_frobenius_error_two_se_interval"] = sorted([
                    math.sqrt(max(0., (energy + offset) / energy)) for energy in (total - 2 * tail_se, total + 2 * tail_se)])
                write_json(path, record)
                print(json.dumps({"file": str(path), "terminal_frobenius_error": terminal["full_relative_frobenius_error_estimate"]}), flush=True)
        return
    directory = Path("outputs/five_step_noise005_n124_20261013/production/r64_n1_ct10/sample_015" if args.case == "ct" else
                     "outputs/five_step_noise020_n124_20261013/production/r64_n1_mri4/sample_014")
    inputs = torch.load(directory / "inputs.pt", map_location="cpu", weights_only=True)
    checkpoint_hash = hashlib.sha256((model_path / "checkpoints/epoch_0200.pt").read_bytes()).hexdigest()
    assert checkpoint_hash == inputs["protocol"]["checkpoint_sha256"]
    settings = json.loads((directory.parent.parent / "manifest.json").read_text())["settings"]
    operator: CT | MRI = CT(64, 10, torch.device("cpu")) if args.case == "ct" else MRI(
        64, 4, inputs["protocol"]["operator"]["mask_seed"], torch.device("cpu"))
    selected = set(range(args.steps + 1)) if args.steps == 5 else {
        round(t * args.steps) for t in (0., .1, .25, .5, .75, .9, .95, .99, 1.)}
    for name in args.sources:
        saved = None if name == "initial" else torch.load(directory / (
            "d_flow.pt" if name == "d_flow_final" else "dlr_d_flow.pt"), map_location="cpu", weights_only=True)
        source = inputs["initial"] if saved is None else saved["source"]
        basis = saved["previous_v"] if name == "dlr_final" else None
        observed: dict[int, tuple[float, Factors]] = {}

        def capture(step: int, alpha: float, factors: Factors) -> None:
            if step in selected:
                assert abs(alpha - (1 - step / args.steps)) < 1e-12
                observed[step] = (0. if step == args.steps else alpha,
                                  tuple(factor.detach().clone() for factor in factors))

        fast_endpoint, _, _ = sensitivity_factors(
            model, source, args.steps, **settings["dlr"], seed=inputs["protocol"]["dlr_seed"],
            seed_basis=basis, factor_callback=capture)
        native_endpoint = euler_flow(model, source, args.steps).detach()
        result: Record = {
            "case": args.case, "source_kind": name, "flow_steps": args.steps, "device": "cpu",
            "threads": args.threads, "width": args.width, "independent_probes": args.probes,
            "refinements": args.refinements, "source_directory": str(directory), "dlr_settings": settings["dlr"],
            "source_sha256": hashlib.sha256(source.numpy().tobytes()).hexdigest(), "checkpoint_sha256": checkpoint_hash,
            "basis_initialization": "saved terminal V used for a new evolution; not original gradient replay" if basis is not None else "fresh sketch",
            "fast_native_terminal_relative_difference": float((fast_endpoint - native_endpoint).norm() / native_endpoint.norm()),
            "limitations": ["Selected64px cases, not a model/dataset survey",
                            "Native partial Euler map derivatives; flow time is not optimization iteration",
                            "Stable and energy ranks are effective ranks, not algebraic rank",
                            "Unresolved cutoffs are reported honestly; stochastic tail uncertainty is not a full certification",
                            "Intermediate gradient diagnostics use hypothetical loss on A(x_t)",
                            "A100step trajectory changes the generator; saved optimized sources were optimized using5steps"],
            "points": [],
        }
        path = OUT / f"{args.case}_{name}_steps{args.steps}.json"
        for step in sorted(selected):
            alpha, factors = observed[step]
            row = measure(model, source, step, args.steps, alpha, factors, operator, inputs,
                          args.width, args.probes, args.refinements, inputs["protocol"]["dlr_seed"] + 85000000)
            result["points"].append(row)
            write_json(path, result)
            print(json.dumps({"file": str(path), **{key: row[key] for key in
                ("time", "native_stable_rank_estimate", "native_energy_rank_estimates",
                 "full_relative_frobenius_error_estimate", "seconds")}}), flush=True)


if __name__ == "__main__":
    main()
