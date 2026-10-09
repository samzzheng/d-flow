"""Short, interleaved TF32/FP32 benchmark on GPU1 without optimizer steps."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Any

import torch

from dlr_dflow.api import approx_source_gradient, load_flow_model, sensitivity_factors
from dlr_dflow.implementation import enriched, tangent_unet
from dlr_dflow.implementation.graphed_products import GraphedAProducts
from p_flow import endpoint_gradient
from sampler.ode_samplers import euler_flow
from .operators import CT
from .run import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--sample", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--skip-exact", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda:1")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.08, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    enriched.SPECIALIZE_ZERO_ENRICHMENT = True
    enriched.SKIP_UNTRUNCATED_SVD = True
    tangent_unet.GN_STATS_MODE = "manual"
    tangent_unet.GN_ACTION_BACKEND = "triton"
    tangent_unet.GN_PRIMAL_BACKEND = "full"
    root = Path("outputs/ct_mri_sweep_geometric_v_20261008")
    settings = json.loads((root / "manifest.json").read_text())["settings"]
    case = root / f"r{args.resolution}_n2_ct10/sample_{args.sample:03d}"
    inputs = torch.load(case / "inputs.pt", map_location="cpu", weights_only=True)
    saved = torch.load(case / "dlr_d_flow.pt", map_location="cpu", weights_only=True)
    model_dir = Path(settings["run_template"].format(resolution=args.resolution, circles=2))
    model, _ = load_flow_model(args.resolution, 2, settings["epoch"], str(device), model_dir)
    source = inputs["initial"].to(device).requires_grad_(True)
    observed = inputs["observed"].to(device)
    operator = CT(args.resolution, 10, device)
    # Fix the same seed basis for every timing sample, rather than evolving V.
    seed_basis = saved["previous_v"].to(device)
    products: dict[bool, GraphedAProducts] = {}
    records: list[dict[str, Any]] = []
    methods = ["objective", "P-Flow", "DLR D-Flow"]
    if not args.skip_exact:
        methods.insert(2, "D-Flow")

    def evaluate(method: str, provider: GraphedAProducts) -> torch.Tensor:
        if method == "D-Flow":
            image = euler_flow(model, source, 100)
            loss = .5 * (operator(image) - observed).abs().square().sum(dtype=torch.float64)
            gradient, = torch.autograd.grad(loss, source)
            return gradient.detach()
        with torch.no_grad():
            image = euler_flow(model, source, 100)
            if method == "objective":
                return .5 * (operator(image) - observed).abs().square().sum(dtype=torch.float64)
        _, gradient = endpoint_gradient(image, operator, observed)
        if method == "DLR D-Flow":
            _, factors, alpha = sensitivity_factors(model, source.detach(), 100,
                **settings["dlr"], seed=saved["protocol"]["dlr_seed"], seed_basis=seed_basis, products=provider)
            assert alpha == 0
            gradient = approx_source_gradient(factors, gradient, alpha)
        return gradient

    result: dict[str, Any] = {
        "resolution": args.resolution, "circles": 2, "flow_steps": 100, "input_case": str(case),
        "device": str(device), "gpu": torch.cuda.get_device_name(device),
        "torch": str(torch.__version__), "dlr": settings["dlr"],
        "inputs_sha256": hashlib.sha256((case / "inputs.pt").read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256((model_dir / "checkpoints/epoch_0200.pt").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "block_order": [True, False, False, True], "repeats_per_block": args.repeats,
        "measurement": "synchronized wall seconds; each gradient includes its forward evaluation; no optimizer or line search",
        "contention": "Existing GPU1 sweep workers remain active; balanced order reduces but cannot remove contention effects.",
        "records": records}
    for block, flag in enumerate(result["block_order"]):
        torch.backends.cudnn.allow_tf32 = flag
        if flag not in products:
            products[flag] = GraphedAProducts(model, source, share_linearization=True, full_step_graphs=True,
                max_jvp_columns=20, max_vjp_columns=16, max_step_jvp_columns=6, max_step_vjp_columns=4)
        provider = products[flag]
        for method in (methods if block % 2 == 0 else list(reversed(methods))):
            # Exclude graph capture, kernel selection, and per-block warmup.
            warm = evaluate(method, provider)
            assert torch.isfinite(warm).all()
            del warm
            torch.cuda.synchronize(device)
            for repeat in range(args.repeats):
                started = time.perf_counter()
                output = evaluate(method, provider)
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                assert torch.isfinite(output).all()
                del output
                records.append({"block": block, "tf32": flag, "method": method,
                                "repeat": repeat, "seconds": elapsed})
            write_json(args.output, result)
            print(json.dumps({"block": block, "tf32": flag, "method": method,
                              "seconds": [r["seconds"] for r in records[-args.repeats:]]}), flush=True)
    result["summary"] = {}
    for method in methods:
        summary: dict[str, Any] = {}
        for flag, label in ((True, "tf32"), (False, "fp32")):
            samples = [r["seconds"] for r in records if r["method"] == method and r["tf32"] == flag]
            summary[label] = {"median": statistics.median(samples), "min": min(samples), "max": max(samples), "n": len(samples)}
        summary["fp32_over_tf32"] = summary["fp32"]["median"] / summary["tf32"]["median"]
        result["summary"][method] = summary
    result["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
    result["complete"] = True
    write_json(args.output, result)
    print(json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()
