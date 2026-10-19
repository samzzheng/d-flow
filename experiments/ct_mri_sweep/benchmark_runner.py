"""Paired GPU benchmark of the experiment harness against a committed runner.

Run only on an idle or explicitly reserved device. Numerical settings are copied
from a manifest; only the attempt budget is shortened. Outputs are isolated.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import ModuleType

import torch

from . import run


class CountedOperator:
    def __init__(self, operator: run.CT | run.MRI) -> None:
        self.operator = operator
        self.calls = 0

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.operator(image)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resolutions", nargs="+", type=int, default=[64, 250])
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--methods", nargs="+", default=list(run.METHODS))
    parser.add_argument("--problems", nargs="+", choices=("ct", "mri"), default=["ct", "mri"])
    parser.add_argument("--deterministic", action="store_true",
                        help="Controlled equivalence benchmark only; requires CUBLAS_WORKSPACE_CONFIG on CUDA")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    baseline_source = subprocess.check_output(
        ["git", "show", f"{args.baseline_ref}:experiments/ct_mri_sweep/run.py"], cwd=run.ROOT, text=True)
    baseline = ModuleType("experiments.ct_mri_sweep._benchmark_baseline")
    baseline.__file__ = run.__file__
    baseline.__package__ = "experiments.ct_mri_sweep"
    sys.modules[baseline.__name__] = baseline
    exec(compile(baseline_source, baseline.__file__, "exec"), baseline.__dict__)
    settings = deepcopy(json.loads(args.manifest.read_text())["settings"])
    settings["dlr"]["rank"] = args.rank
    settings["stopping"].update(min_updates=args.updates, max_updates=args.updates)
    execution = settings["execution"]
    torch.set_num_threads(execution["threads"])
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cudnn.allow_tf32 = execution["cudnn_tf32"]
    torch.backends.cuda.matmul.allow_tf32 = execution["matmul_tf32"]
    torch.backends.cudnn.benchmark = execution["cudnn_benchmark"]
    torch.use_deterministic_algorithms(args.deterministic)
    torch.backends.cudnn.deterministic = args.deterministic
    rows = []
    for resolution in args.resolutions:
        circles = settings["circles"][0]
        job = run.Job(resolution, circles, "ct", settings["ct_angles"][0])
        directory = run.model_directory(job, settings)
        model, _ = run.load_flow_model(resolution, circles, settings["epoch"], str(device), directory)
        dataset = settings.get("dataset_template", "data/circles-{resolution}-n{circles}.pt")
        data = torch.load(run.ROOT / dataset.format(resolution=resolution, circles=circles),
                          map_location="cpu", weights_only=True, mmap=True)
        stored = data["test"][:1]
        truth = stored.to(device=device, dtype=torch.float32) / (255 if stored.dtype == torch.uint8 else 1)
        seed = settings["seed"] + 10000 * resolution + 100 * circles
        source = torch.randn(truth.shape, generator=torch.Generator().manual_seed(seed)).to(device)
        run.tangent_unet.GN_STATS_MODE = "native" if resolution <= 96 else "manual"
        run.tangent_unet.GN_ACTION_BACKEND = execution["dlr_fused_norm"]
        run.tangent_unet.GN_PRIMAL_BACKEND = execution["dlr_fused_primal"]
        run.enriched.SPECIALIZE_ZERO_ENRICHMENT = execution["dlr_specialized_algebra"]
        run.enriched.SKIP_UNTRUNCATED_SVD = True
        products = run.GraphedAProducts(model, source, share_linearization=True,
            full_step_graphs=execution["dlr_full_step_graphs"], max_jvp_columns=20, max_vjp_columns=16,
            max_step_jvp_columns=execution["dlr_jvp_columns"], max_step_vjp_columns=execution["dlr_vjp_columns"])
        for problem in args.problems:
            operator = (run.CT(resolution, settings["ct_angles"][0], device) if problem == "ct" else
                        run.MRI(resolution, settings["mri_accelerations"][0], seed + 10000000, device,
                                settings["mri_pattern"]))
            clean = operator(truth)
            observed = run.noisy_measurement(clean, settings["noise_relative"], seed + 20000000,
                                              getattr(operator, "mask", None))
            for method in args.methods:
                warm_settings = deepcopy(settings)
                warm_settings["stopping"].update(min_updates=2, max_updates=2)
                warm_path = args.out / "warm.pt"
                warm_path.unlink(missing_ok=True)
                evaluator = run.Evaluator(model, operator, observed, warm_settings, seed + 30000000, products)
                run.solve(evaluator, method, source, truth, clean, warm_path, {}, [False])
                for repeat in range(args.repeats):
                    results = {}
                    variants = [("baseline", baseline), ("optimized", run), ("baseline_repeat", baseline)]
                    for label, module in variants[::1 if repeat % 2 == 0 else -1]:
                        counted = CountedOperator(operator)
                        evaluator = module.Evaluator(model, counted, observed, settings, seed + 30000000, products)
                        destination = args.out / f"r{resolution}_{problem}_{method.replace(' ', '_')}_{repeat}_{label}.pt"
                        destination.unlink(missing_ok=True)
                        torch.cuda.synchronize(device)
                        started = time.perf_counter()
                        result = module.solve(evaluator, method, source, truth, clean, destination, {}, [False])
                        torch.cuda.synchronize(device)
                        seconds = time.perf_counter() - started
                        results[label] = result
                        rows.append({"resolution": resolution, "problem": problem, "rank": args.rank,
                            "method": method, "repeat": repeat, "variant": label, "seconds": seconds,
                            "operator_calls": counted.calls, "attempts": result["history"][-1]["step"],
                            "history_length": len(result["optimizer"]["S"]),
                            "evaluations": result["evaluations"], "status": result["status"],
                            "image_relative_l2": result["history"][-1]["image_relative_l2"]})
                    old, new = results["baseline"], results["optimized"]
                    differences = {}
                    for key in ("source", "image", "measurement"):
                        denominator = old[key].norm().clamp_min(1e-20)
                        difference = float((new[key] - old[key]).norm() / denominator)
                        control = float((results["baseline_repeat"][key] - old[key]).norm() / denominator)
                        differences[key] = {"optimized_relative_difference": difference,
                                            "baseline_repeat_relative_difference": control,
                                            "within_repeatability_envelope": difference <= max(1e-5, 5 * control)}
                        # CUDA grid_sample backward is nondeterministic. Compare norms,
                        # record an unchanged-run control, and check discrete decisions.
                    rows[-1]["equivalence"] = {"arrays": differences,
                        "same_evaluations": old["evaluations"] == new["evaluations"],
                        "same_status": old["status"] == new["status"],
                        "same_line_search_trials": [h.get("line_search_trials") for h in old["history"]] ==
                                                   [h.get("line_search_trials") for h in new["history"]]}
                    run.write_json(args.out / "timings.json", {"baseline_ref": args.baseline_ref,
                        "runner_sha256": hashlib.sha256(Path(run.__file__).read_bytes()).hexdigest(),
                        "deterministic": args.deterministic,
                        "settings": settings, "gpu": torch.cuda.get_device_name(device), "rows": rows})
        del evaluator, products, model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
