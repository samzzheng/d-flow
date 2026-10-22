"""Measurement-selected source restarts with one shared optimizer-attempt budget."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import torch

from dlr_dflow.api import load_flow_model
from dlr_dflow.implementation import enriched, tangent_unet
from dlr_dflow.implementation.graphed_products import GraphedAProducts
from experiments.ct_mri_sweep import run
from experiments.ct_mri_sweep.operators import CT, MRI
from experiments.ct_mri_sweep.queue import source_hashes
from scripts.reconstruction_figures import RESIDUAL_VMAX, plot_reconstruction_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path,
        default=Path("outputs/separated_shapes_117k_noiseless_20261021/production"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", choices=run.METHODS, default=list(run.METHODS))
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--probe-updates", type=int, default=50)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--no-reuse-v", action="store_true")
    parser.add_argument("--source-pool-root", type=Path,
        help="Reuse a previously cached candidate pool exactly across controls")
    args = parser.parse_args()
    if not 0 < 4 * args.probe_updates < args.budget:
        parser.error("Four positive probe budgets must leave a continuation budget")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda:1")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.2, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    settings = json.loads((args.baseline / "manifest.json").read_text())["settings"]
    settings["dlr"]["rank"] = args.rank
    if args.no_reuse_v:
        settings["dlr_reuse_previous_v"] = False
    settings["stopping"]["max_updates"] = args.probe_updates
    evidence = dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_hashes=source_hashes(), settings=settings, budget=args.budget,
        probe_updates=args.probe_updates, cases=args.cases, methods=args.methods,
        selection="Minimum final probe measurement objective; no ground-truth selection")
    if args.source_pool_root:
        evidence["source_pool_root"] = str(args.source_pool_root.resolve())
    run.write_json(args.out / "config.json", evidence)
    tangent_unet.GN_STATS_MODE = "native"
    tangent_unet.GN_ACTION_BACKEND = "triton"
    tangent_unet.GN_PRIMAL_BACKEND = "full"
    enriched.SPECIALIZE_ZERO_ENRICHMENT = True
    enriched.SKIP_UNTRUNCATED_SVD = True
    for case in args.cases:
        original = torch.load(args.baseline / case / "inputs.pt", map_location=device, weights_only=True)
        protocol = original["protocol"]
        resolution = protocol["job"]["resolution"]
        folder = Path(settings["run_template"].format(resolution=resolution))
        assert hashlib.sha256((folder / "checkpoints/epoch_0500.pt").read_bytes()).hexdigest() == protocol["checkpoint_sha256"]
        model, _ = load_flow_model(resolution, 0, 500, str(device), folder)
        info = protocol["operator"]
        operator = (CT(resolution, protocol["job"]["level"], device) if protocol["job"]["problem"] == "ct"
                    else MRI(resolution, 4, info["mask_seed"], device, info["pattern"]))
        if isinstance(operator, MRI):
            assert torch.equal(operator.mask, original["mask"])
        directory = args.out / case
        directory.mkdir(parents=True, exist_ok=True)
        pool_path = directory / "pool.pt"
        if pool_path.exists():
            pool = torch.load(pool_path, map_location=device, weights_only=True)
            assert pool["protocol"] == protocol and pool["experiment"] == evidence
            for key in ("truth", "clean", "observed"):
                assert torch.equal(pool[key], original[key])
        elif args.source_pool_root:
            parent = args.source_pool_root / case / "pool.pt"
            cached = torch.load(parent, map_location=device, weights_only=True)
            assert cached["protocol"] == protocol
            for key in ("truth", "clean", "observed"):
                assert torch.equal(cached[key], original[key])
            for name, value in cached["sources"].items():
                assert hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest() == cached["source_hashes"][name]
            pool = {**cached, "experiment": evidence,
                    "source_pool_parent_sha256": hashlib.sha256(parent.read_bytes()).hexdigest()}
            run.save_tensor(pool_path, pool)
        else:
            source = original["initial"]
            variable = torch.zeros_like(source, requires_grad=True)
            pairing = (operator(variable).conj() * original["observed"]).real.sum()
            direction, = torch.autograd.grad(pairing, variable)
            direction -= direction.mean()
            if not torch.isfinite(direction).all() or direction.norm() == 0:
                raise FloatingPointError("Measurement adjoint cannot define an initialization")
            direction *= math.sqrt(source.numel()) / direction.norm()
            additional_seed = protocol["source_seed"] + 50000000
            candidates = {
                "original": source,
                "measurement_half": math.sqrt(.75) * source + .5 * direction,
                "measurement_full": direction,
                "independent": torch.randn(source.shape,
                    generator=torch.Generator().manual_seed(additional_seed)).to(source),
            }
            pool = {**original, "experiment": evidence, "sources": candidates,
                "additional_source_seed": additional_seed,
                "source_hashes": {name: hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest()
                                  for name, value in candidates.items()}}
            run.save_tensor(pool_path, pool)
        products = GraphedAProducts(model, pool["initial"], share_linearization=True,
            full_step_graphs=True, max_jvp_columns=20, max_vjp_columns=16,
            max_step_jvp_columns=6, max_step_vjp_columns=4)
        results = {}
        summaries = {}
        for method in args.methods:
            stem = method.lower().replace("-", "_").replace(" ", "_")
            probes = {}
            for name, source in pool["sources"].items():
                (directory / name).mkdir(exist_ok=True)
                paired = {**protocol, "experiment": evidence, "candidate": name,
                    "initial_source_sha256": pool["source_hashes"][name]}
                evaluator = run.Evaluator(model, operator, pool["observed"], settings,
                                          protocol["dlr_seed"], products)
                probes[name] = run.solve(evaluator, method, source, pool["truth"], pool["clean"],
                    directory / name / f"{stem}.pt", paired, [False])
            winner = min(probes, key=lambda name: probes[name]["history"][-1]["phi"])
            selected = probes[winner]
            spent = sum(result["history"][-1]["step"] for result in probes.values())
            selected_steps = selected["history"][-1]["step"]
            continuation = deepcopy(settings)
            continuation["stopping"]["max_updates"] = selected_steps + args.budget - spent
            paired = {**selected["protocol"], "stage": "continuation", "selected_candidate": winner,
                "probe_attempts_total": spent, "continuation_settings": continuation}
            final_path = directory / f"{stem}.pt"
            if not final_path.exists():
                run.save_tensor(final_path, {**selected, "protocol": paired,
                    "status": "paused" if selected["status"] == "step_limit" else selected["status"]})
            evaluator = run.Evaluator(model, operator, pool["observed"], continuation,
                                      protocol["dlr_seed"], products)
            final = run.solve(evaluator, method, pool["sources"][winner], pool["truth"], pool["clean"],
                              final_path, paired, [False])
            total = spent + final["history"][-1]["step"] - selected_steps
            assert total <= args.budget
            summaries[method] = dict(selected=winner, status=final["status"], total_attempts=total,
                final_metrics=final["history"][-1],
                probes={name: dict(objective=result["history"][-1]["phi"],
                                  attempts=result["history"][-1]["step"],
                                  image_error=result["history"][-1]["image_relative_l2"])
                        for name, result in probes.items()})
            run.write_json(directory / "summary.json", summaries)
            results[method] = final
        figure = plot_reconstruction_comparison(pool["truth"], pool["clean"],
            {method: (result["image"], result["measurement"]) for method, result in results.items()},
            observation=pool["observed"], include_errors=True, measurement_log=isinstance(operator, MRI),
            residual_vmax=RESIDUAL_VMAX[protocol["job"]["problem"]],
            title=f"{case}: measurement-selected starts; total budget {args.budget}")
        figure.savefig(directory / "comparison.png", dpi=120)
        run.plt.close(figure)
        del evaluator, products, model, original, pool, probes, results
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
