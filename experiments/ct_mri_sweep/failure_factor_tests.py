"""CPU same-source factor diagnostics at four frozen sweep checkpoints."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
from typing import Any

import torch

from dlr_dflow.api import approx_source_gradient, load_flow_model, sensitivity_factors
from dlr_dflow.implementation import enriched, tangent_unet
from p_flow import endpoint_gradient
from sampler.ode_samplers import euler_flow
from .operators import CT, MRI
from .run import save_tensor, write_json

Record = dict[str, Any]
CASES = ("r128_n3_ct10/sample_008", "r128_n3_ct10/sample_018",
         "r128_n2_mri4/sample_003", "r128_n2_mri4/sample_019")
VARIANTS = (("rank8_random", 8, False, 0), ("rank8_saved_v", 8, True, 0),
            ("rank16_random", 16, False, 0), ("rank32_random", 32, False, 0),
            ("rank8_enrichment2", 8, False, 2))


def gradient_metrics(approximate: torch.Tensor, exact: torch.Tensor,
                     basis: torch.Tensor) -> Record:
    approximate, exact, basis = approximate.flatten().double(), exact.flatten().double(), basis.double()
    projected = basis @ (basis.T @ exact)
    error, subspace, coefficient = approximate - exact, projected - exact, approximate - projected
    denominator = float(exact.norm())
    error_squared = float(error.square().sum())
    decomposition_gap = abs(error_squared - float(subspace.square().sum() + coefficient.square().sum()))
    cosine = None
    if denominator and float(approximate.norm()):
        cosine = float(approximate @ exact / (approximate.norm() * exact.norm()))
    assert decomposition_gap <= 1e-4 * max(error_squared, denominator ** 2, 1e-30)
    return {"relative_error": float(error.norm()) / denominator if denominator else None,
            "subspace_relative_error": float(subspace.norm()) / denominator if denominator else None,
            "coefficient_relative_error": float(coefficient.norm()) / denominator if denominator else None,
            "exact_norm": denominator, "approximate_norm": float(approximate.norm()), "cosine": cosine,
            "norm_ratio": float(approximate.norm()) / denominator if denominator else None,
            "negative_gradient_true_slope": -float(exact @ approximate),
            "negative_unit_gradient_true_slope": -float(exact @ approximate) / float(approximate.norm()) if float(approximate.norm()) else None,
            "orthogonal_decomposition_relative_gap": decomposition_gap / max(error_squared, denominator ** 2, 1e-30)}


def run_case(snapshot_path: Path, output: Path, case: str) -> Record:
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    enriched.SPECIALIZE_ZERO_ENRICHMENT = True
    tangent_unet.GN_STATS_MODE = "native"
    tangent_unet.GN_ACTION_BACKEND = "native"
    tangent_unet.GN_PRIMAL_BACKEND = "native"
    snapshot = json.loads(snapshot_path.read_text())
    settings, root = snapshot["settings"], Path(snapshot["root"])
    original = next(record for record in snapshot["records"] if record["case"] == str(root / case))
    state_path = root / case / "dlr_d_flow.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    inputs = torch.load(root / case / "inputs.pt", map_location="cpu", weights_only=True)
    assert state["history"] == original["methods"]["DLR D-Flow"]["history"]
    protocol, source = state["protocol"], state["source"].detach().clone().requires_grad_(True)
    job = protocol["job"]
    model_dir = Path(settings["run_template"].format(**job))
    checkpoint = model_dir / f"checkpoints/epoch_{settings['epoch']:04d}.pt"
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == protocol["checkpoint_sha256"]
    model, _ = load_flow_model(job["resolution"], job["circles"], settings["epoch"], "cpu", model_dir)
    operator = (CT(job["resolution"], job["level"], torch.device("cpu")) if job["problem"] == "ct" else
                MRI(job["resolution"], job["level"], protocol["operator"]["mask_seed"], torch.device("cpu")))
    if isinstance(operator, MRI):
        assert torch.equal(operator.mask, inputs["mask"])
    assert torch.equal(inputs["observed"], inputs["clean"])
    destination = output / case
    destination.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    image = euler_flow(model, source, settings["flow_steps"])
    loss = .5 * (operator(image) - inputs["observed"]).abs().square().sum(dtype=torch.float64)
    exact, = torch.autograd.grad(loss, source)
    image = image.detach()
    _, gx = endpoint_gradient(image, operator, inputs["observed"])
    result: Record = {"case": case, "device": "cpu", "threads": 1, "flow_steps": settings["flow_steps"],
        "snapshot": str(snapshot_path), "checkpoint_sha256": protocol["checkpoint_sha256"],
        "saved_state_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "saved_method_errors": {name: record["history"][-1]["image_relative_l2"] for name, record in original["methods"].items()},
        "source_attempt": state["history"][-1]["step"], "saved_status": state["status"],
        "exact_gradient_norm": float(exact.norm()), "native_cpu_phi": float(loss.detach()),
        "saved_cuda_phi": state["history"][-1]["phi"],
        "cpu_cuda_endpoint_relative_difference": float((image-state["image"]).norm()/state["image"].norm()),
        "cpu_image_relative_error": float((image-inputs["truth"]).norm()/inputs["truth"].norm()),
        "saved_cuda_gradient": gradient_metrics(state["optimizer"]["g"], exact, state["previous_v"]),
        "variants": {},
        "reuse_note": "Saved terminal V seeds a fresh factor solve at this same saved source; it is not the previous source's unavailable seed basis."}
    del loss
    save_tensor(destination / "reference.pt", {"source": source.detach(), "image": image, "exact": exact, "gx": gx})
    write_json(destination / "result.json", result)
    print(json.dumps({"case": case, "exact_ready": True, "cpu_cuda_endpoint_drift": result["cpu_cuda_endpoint_relative_difference"]}), flush=True)
    for name, rank, reuse, enrichment in VARIANTS:
        variant_start = time.monotonic()
        options = {**settings["dlr"], "rank": rank, "enrichment_probes": enrichment,
                   "trace_probes": 0, "return_alpha_path": True}
        endpoint, factors, alpha, alpha_path = sensitivity_factors(
            model, source.detach(), settings["flow_steps"], **options,
            seed=protocol["dlr_seed"], seed_basis=state["previous_v"] if reuse else None)
        assert alpha == 0
        assert max(abs(value - (1-index/settings["flow_steps"])) for index, value in enumerate(alpha_path)) < 1e-12
        gradient = approx_source_gradient(factors, gx, alpha).detach()
        assert torch.isfinite(gradient).all()
        record = gradient_metrics(gradient, exact, factors[2])
        record.update(rank=rank, enrichment=enrichment, seeded=reuse, terminal_alpha=alpha,
                      dlr_native_endpoint_relative_difference=float((endpoint-image).norm()/image.norm()),
                      factor_seconds=time.monotonic()-variant_start,
                      singular_values=torch.diag(factors[1]).tolist())
        result["variants"][name] = record
        result["elapsed_seconds"] = time.monotonic()-started
        save_tensor(destination / f"{name}.pt", {"gradient": gradient, "v": factors[2], "core": factors[1]})
        write_json(destination / "result.json", result)
        print(json.dumps({"case": case, "variant": name, **record}), flush=True)
    assert not torch.cuda.is_initialized()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("outputs/ct_mri_geometric_v_audit_20261008_1645/snapshot.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/dlr_failure_tests_20261008/factors"))
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = []
        futures = [pool.submit(run_case, args.snapshot, args.output, case) for case in CASES]
        for future in as_completed(futures):
            results.append(future.result())
            write_json(args.output / "results.json", results)


if __name__ == "__main__":
    main()
