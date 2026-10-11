"""Estimate effective ranks of the native Euler Jacobian, independently of DLR."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable

import torch

from dlr_dflow import dlr_dflow as core
from dlr_dflow.api import load_flow_model, make_terminal_flow_products
from dlr_dflow.implementation.graphed_products import GraphedAProducts
from experiments.ct_mri_sweep.run import write_json

Tensor = torch.Tensor
Product = Callable[[Tensor], Tensor]
Record = dict[str, Any]


class NativeProducts:
    """Differentiate each native Euler step along a saved native trajectory.

    CUDA graphs only replay torch.func AD products; no low-rank approximation,
    manual UNet derivative, or fast DLR primal enters the reference.
    """

    def __init__(self, model: torch.nn.Module, source: Tensor, steps: int,
                 columns: int) -> None:
        self.steps, self.columns = steps, columns
        self.states = [source.detach().clone()]
        with torch.no_grad():
            for index in range(steps):
                t = source.new_full((1,), index / steps)
                self.states.append(self.states[-1] + model(self.states[-1], t) / steps)
        # Isolated rank worker only; hybrid workers are separate processes.
        core.HAND_JVP = core.HAND_VJP = False
        self.products = GraphedAProducts(model, source, share_linearization=False,
                                        full_step_graphs=False)

    def apply(self, values: Tensor, transpose: bool = False) -> Tensor:
        outputs = []
        with torch.no_grad():
            for block in values.split(self.columns, dim=1):
                width = block.shape[1]
                q = torch.nn.functional.pad(block, (0, self.columns - width)).contiguous()
                indices = reversed(range(self.steps)) if transpose else range(self.steps)
                for index in indices:
                    self.products.set_state(index / self.steps, self.states[index])
                    action = self.products.AT if transpose else self.products.A
                    q = q + action(q) / self.steps
                outputs.append(q[:, :width].clone())
        return torch.cat(outputs, dim=1)

    def transpose(self, values: Tensor) -> Tensor:
        return self.apply(values, transpose=True)


def spectrum(J: Product, JT: Product, source: Tensor, output: Path,
             widths: list[int], refinements: int, probes: int, seed: int,
             metadata: Record) -> Record:
    """Randomized subspace iteration with independent omitted-energy checks.

    Energy outside Q is estimated with fresh Gaussian probes. Total energy is
    ||J^T Q||_F^2 plus that estimate, rather than the energy in computed values
    alone. Report an estimate and its sampling uncertainty, not an exact rank.
    """
    dimension = source.numel()
    generator = torch.Generator(device=source.device).manual_seed(seed)
    omega = torch.randn(dimension, probes, generator=generator,
                       device=source.device, dtype=source.dtype)
    heldout = J(omega)
    adjoint_probe = torch.randn(dimension, 1, generator=generator,
                               device=source.device, dtype=source.dtype)
    adjoint = JT(adjoint_probe)
    adjoint_error = float(((adjoint_probe * heldout[:, :1]).sum() -
                           (adjoint * omega[:, :1]).sum()).abs() /
                          (adjoint_probe.norm() * heldout[:, :1].norm() +
                           adjoint.norm() * omega[:, :1].norm()).clamp_min(1e-30))
    if not adjoint_error < 1e-4:
        raise AssertionError(f"Adjoint check failed: {adjoint_error}")
    result = {**metadata, "adjoint_scaled_error": adjoint_error, "stages": [],
              "status": "running", "energy_definition": "sum(sigma_i^2)",
              "independent_tail_probes": probes, "spectral_seed": seed}
    started = time.monotonic()
    previous_ranks = None
    for requested_width in widths:
        width = min(requested_width, dimension)
        sketch = torch.randn(dimension, width, generator=generator,
                             device=source.device, dtype=source.dtype)
        q = torch.linalg.qr(J(sketch), mode="reduced")[0]
        for refinement in range(refinements):
            b = JT(q)
            # Tall, small-width SVD in float64 for the spectral algebra.
            v64, singular, ut64 = torch.linalg.svd(b.double(), full_matrices=False)
            v, u = v64.to(source.dtype), q @ ut64.T.to(source.dtype)
            left = J(v) - u * singular.to(source.dtype)
            residuals = left.double().norm(dim=0) / singular[0].clamp_min(1e-30)
            outside = heldout - q @ (q.T @ heldout)
            energies = outside.double().square().sum(dim=0)
            tail = float(energies.mean())
            tail_se = float(energies.std() / probes ** .5)
            squared = singular.square()
            total = float(squared.sum()) + tail
            # Rank includes the estimated energy beyond the calculated range.
            remaining = squared.sum() - squared.cumsum(0) + tail
            ranks: dict[str, int | None] = {}
            intervals: dict[str, list[int | None]] = {}
            for target in (.99, .999, .9999):
                key = str(target)
                indices = torch.where(remaining <= (1 - target) * total)[0]
                ranks[key] = int(indices[0]) + 1 if len(indices) else None
                bounds = []
                for omitted in (max(0., tail - 2 * tail_se), tail + 2 * tail_se):
                    indices = torch.where(squared.sum() - squared.cumsum(0) + omitted <=
                                          (1 - target) * (float(squared.sum()) + omitted))[0]
                    bounds.append(int(indices[0]) + 1 if len(indices) else None)
                intervals[key] = bounds
            needed = ranks["0.9999"]
            residual = float(residuals[:needed].max()) if needed else float(residuals.max())
            stable = previous_ranks == ranks and needed is not None
            tail_resolved = needed is not None and intervals["0.9999"][1] is not None
            resolved = stable and tail_resolved and residual < 1e-3
            row = {"width": width, "refinement": refinement,
                   "singular_values": singular.tolist(), "left_triplet_relative_residuals": residuals.tolist(),
                   "right_triplet_note": "J^T U = V Sigma by the thin SVD construction, up to roundoff",
                   "omitted_energy_estimate": tail, "omitted_energy_standard_error": tail_se,
                   "omitted_energy_fraction": tail / total if total else None,
                   "effective_ranks": ranks, "tail_two_se_rank_sensitivity": intervals,
                   "singular_threshold_counts": {str(tolerance): int((singular > tolerance * singular[0]).sum())
                                                 for tolerance in (.01, .001, .0001)},
                   "stable_from_previous_stage": stable, "relevant_triplet_residual": residual,
                   "numerically_resolved": resolved,
                   "seconds": time.monotonic() - started}
            result["stages"].append(row)
            result["latest"] = row
            write_json(output, result)
            print(json.dumps({"file": str(output), **{k: row[k] for k in
                  ("width", "refinement", "effective_ranks", "omitted_energy_fraction",
                   "relevant_triplet_residual", "numerically_resolved")}}), flush=True)
            if resolved:
                result["status"] = "resolved_estimate"
                write_json(output, result)
                return result
            previous_ranks = ranks
            if refinement + 1 < refinements:
                right_q = torch.linalg.qr(b, mode="reduced")[0]
                q = torch.linalg.qr(J(right_q), mode="reduced")[0]
    result["status"] = "unresolved_at_budget"
    write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--circles", type=int, required=True)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--widths", nargs="+", type=int, default=[48, 96, 192, 256])
    parser.add_argument("--refinements", type=int, default=3)
    parser.add_argument("--probes", type=int, default=24)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists() and json.loads(args.output.read_text())["status"] != "running":
        return
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(args.device)
    torch.cuda.set_per_process_memory_fraction(.65)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    run = Path(f"outputs/fm_unet/circles-{args.resolution}-noattn-ch16-b8-e200/n{args.circles}")
    checkpoint = run / "checkpoints/epoch_0200.pt"
    model, _ = load_flow_model(args.resolution, args.circles, 200, args.device, run)
    seed = 20261007 + 10000 * args.resolution + 100 * args.circles + args.sample
    source = torch.randn((1, 1, args.resolution, args.resolution),
                         generator=torch.Generator().manual_seed(seed)).to(args.device)
    source_hash = None
    if args.source:
        saved = torch.load(args.source, weights_only=True, map_location="cpu")
        if saved["protocol"]["checkpoint_sha256"] != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
            raise ValueError("Source/model checkpoint mismatch")
        source = saved["source"].to(args.device)
        source_hash = hashlib.sha256(source.cpu().numpy().tobytes()).hexdigest()
    products = NativeProducts(model, source, args.steps, args.columns)
    # Validate graph products against full-map native AD before spectrum work.
    raw_j, raw_jt = make_terminal_flow_products(model, source, args.steps)
    probe = torch.randn(source.numel(), 1, device=args.device)
    with torch.no_grad():
        j, jt = products.apply(probe), products.transpose(probe)
        reference_j, reference_jt = raw_j(probe), raw_jt(probe)
    errors = [float((a - b).norm() / b.norm().clamp_min(1e-30)) for a, b in
              ((j, reference_j), (jt, reference_jt))]
    if max(errors) > 1e-3:
        raise AssertionError(f"Native graph/full-map AD mismatch: {errors}")
    metadata = {"resolution": args.resolution, "circles": args.circles,
                "sample": args.sample, "source_kind": "optimized" if args.source else "initial",
                "source_path": str(args.source) if args.source else None,
                "source_sha256": source_hash, "source_seed": seed,
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "steps": args.steps, "dtype": "float32", "tf32": False,
                "native_graph_vs_full_map_relative_errors": errors}
    del raw_j, raw_jt, reference_j, reference_jt
    torch.cuda.empty_cache()
    spectrum(products.apply, products.transpose, source, args.output, args.widths,
             args.refinements, args.probes, seed + 70000000, metadata)


if __name__ == "__main__":
    main()
