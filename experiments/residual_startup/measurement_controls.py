"""Held-out shape/count/source controls for measurement-informed initialization."""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dataset.separated_shapes import rasterize, sample_geometry
from dlr_dflow.api import approx_source_gradient, load_flow_model, sensitivity_factors
from dlr_dflow.implementation import enriched, tangent_unet
from dlr_dflow.implementation.graphed_products import GraphedAProducts
from experiments.ct_mri_sweep.operators import CT, MRI
from experiments.ct_mri_sweep.queue import source_hashes
from experiments.ct_mri_sweep.run import Evaluator, TERMINAL, solve, write_json
from p_flow import endpoint_gradient
from scripts.reconstruction_figures import plot_reconstruction_comparison

ARMS = ("original", "random_mix", "measurement_mix")
LABELS = ("Original source", "Random-direction mix", "Measurement-adjoint mix")


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def prepare(out: Path) -> None:
    """Cache sources once on CPU, before looking at reconstruction results."""
    if (out / "plan.json").exists():
        raise FileExistsError("Refusing to replace the frozen initialization study")
    torch.set_num_threads(2)
    template = Path("outputs/separated_shapes_steps_20261019/measurement_init_original_0.5_paired/production/manifest.json")
    settings = deepcopy(json.loads(template.read_text())["settings"])
    settings.pop("initialization_cache", None)
    settings.pop("initialization", None)
    settings["execution"]["checkpoint_every"] = 50
    settings["dlr"]["residual_sketch_steps"] = None
    dataset = torch.load("data/separated-shapes-64.pt", weights_only=True, mmap=True)
    tasks = []
    for sample, count in ((20, 2), (22, 2), (9, 4), (14, 4)):
        geometry_seed = int(dataset["sample_seeds"]["test"][sample])
        geometry = sample_geometry(geometry_seed)
        assert len(geometry) == count
        stored = dataset["test"][sample:sample + 1]
        regenerated = torch.from_numpy((rasterize(geometry, 64) * 255).round()).to(torch.uint8)
        assert torch.equal(stored.squeeze(), regenerated)
        truth = stored.float() / 255
        for problem in ("ct", "mri"):
            operator = CT(64, 10, torch.device("cpu")) if problem == "ct" else MRI(
                64, 4, 20261020 + sample, torch.device("cpu"))
            observed = operator(truth).detach()
            variable = torch.zeros_like(truth, requires_grad=True)
            pairing = (operator(variable).conj() * observed).real.sum()
            backprojection, = torch.autograd.grad(pairing, variable)
            centered = backprojection - backprojection.mean()
            if not torch.isfinite(centered).all() or centered.norm() == 0:
                raise FloatingPointError("Invalid centered measurement adjoint")
            direction = centered * (math.sqrt(truth.numel()) / centered.norm())
            for seed in (202610201, 202610202):
                initial = torch.randn(truth.shape, generator=torch.Generator().manual_seed(seed))
                random = torch.randn(truth.shape, generator=torch.Generator().manual_seed(seed + 1000000))
                random = random - random.mean()
                random = random * (direction.norm() / random.norm())
                sources = dict(original=initial, random_mix=math.sqrt(.75) * initial + .5 * random,
                               measurement_mix=math.sqrt(.75) * initial + .5 * direction)
                torch.testing.assert_close(random.norm(), direction.norm(), rtol=1e-6, atol=0)
                assert abs(float(random.mean())) < 1e-6 and abs(float(direction.mean())) < 1e-6
                lhs = (operator(initial).conj() * observed).real.sum()
                rhs = (initial * backprojection).sum()
                torch.testing.assert_close(lhs, rhs, rtol=2e-5, atol=1e-5)
                case = out / f"sample_{sample:03d}" / f"{problem}_seed{seed}"
                case.mkdir(parents=True, exist_ok=True)
                bundle = dict(truth=truth, observed=observed, sources=sources,
                              direction=direction, random_direction=random, operator=operator.metadata)
                torch.save(bundle, case / "inputs.pt")
                tasks.append(dict(sample=sample, count=count, geometry_seed=geometry_seed,
                                  problem=problem, seed=seed, dlr_seed=seed + 2000000,
                                  case=str(case.resolve()), operator=operator.metadata,
                                  hashes={**{name: tensor_hash(value) for name, value in sources.items()},
                                          "truth": tensor_hash(truth), "observed": tensor_hash(observed)},
                                  source_norms={name: float(value.norm()) for name, value in sources.items()},
                                  direction_norm=float(direction.norm()), random_direction_norm=float(random.norm())))
    run = Path(settings["run_template"].format(resolution=64))
    checkpoint = run / f"checkpoints/epoch_{settings['epoch']:04d}.pt"
    hashes = source_hashes()
    hashes[str(Path(__file__).relative_to(Path.cwd()))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_json(out / "plan.json", dict(settings=settings, checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                                     source_sha256=hashes, tasks=tasks, rho=.5,
                                     methods=["DLR D-Flow"], arms=ARMS, expected_solves=len(tasks) * len(ARMS),
                                     selection="First two previously untested indices of each requested count among test0:30; excludes prior samples1 and7",
                                     hypothesis="Measurement-adjoint mix improves paired image error against both original and same-norm random-direction starts"))


def worker(out: Path, sample: int) -> None:
    plan = json.loads((out / "plan.json").read_text())
    for name, expected in plan["source_sha256"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    settings = plan["settings"]
    torch.set_num_threads(2)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.25, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    tangent_unet.GN_STATS_MODE = "native"
    tangent_unet.GN_ACTION_BACKEND = settings["execution"]["dlr_fused_norm"]
    tangent_unet.GN_PRIMAL_BACKEND = settings["execution"]["dlr_fused_primal"]
    enriched.SPECIALIZE_ZERO_ENRICHMENT = True
    enriched.SKIP_UNTRUNCATED_SVD = True
    run = Path(settings["run_template"].format(resolution=64))
    assert hashlib.sha256((run / f"checkpoints/epoch_{settings['epoch']:04d}.pt").read_bytes()).hexdigest() == plan["checkpoint_sha256"]
    model, _ = load_flow_model(run_dir=run, epoch=settings["epoch"], device=str(device))
    products = GraphedAProducts(model, torch.zeros(1, 1, 64, 64, device=device),
                               share_linearization=True, full_step_graphs=True,
                               max_jvp_columns=20, max_vjp_columns=16,
                               max_step_jvp_columns=6, max_step_vjp_columns=4)
    for task in filter(lambda item: item["sample"] == sample, plan["tasks"]):
        case = Path(task["case"])
        inputs = torch.load(case / "inputs.pt", weights_only=True)
        for key in ("truth", "observed"):
            assert tensor_hash(inputs[key]) == task["hashes"][key]
        truth, observed = inputs["truth"].to(device), inputs["observed"].to(device)
        operator = CT(64, 10, device) if task["problem"] == "ct" else MRI(
            64, 4, task["operator"]["mask_seed"], device)
        assert operator.metadata == inputs["operator"]
        torch.testing.assert_close(operator(truth), observed, rtol=2e-6, atol=2e-6)
        for arm, label in zip(ARMS, LABELS):
            initial = inputs["sources"][arm].to(device)
            assert tensor_hash(initial) == task["hashes"][arm]
            protocol = dict(settings=settings, task=task, arm=arm, rho=plan["rho"],
                            checkpoint_sha256=plan["checkpoint_sha256"], sources=plan["source_sha256"],
                            initial_source_sha256=tensor_hash(initial), gpu=torch.cuda.get_device_name(device))
            evaluator = Evaluator(model, operator, observed, settings, task["dlr_seed"], products)
            evaluator.method = "DLR D-Flow"
            graph_gradient = evaluator.gradient(initial, "DLR D-Flow")
            _, factors, alpha = sensitivity_factors(model, initial, 5, **settings["dlr"], seed=task["dlr_seed"])
            _, gx = endpoint_gradient(evaluator.image, operator, observed)
            eager_gradient = approx_source_gradient(factors, gx, alpha)
            error = float((graph_gradient - eager_gradient).norm() / eager_gradient.norm())
            if not math.isfinite(error) or error >= .01:
                raise AssertionError(f"Graph/eager calibration failed: {error}")
            write_json(case / f"{arm}_calibration.json", dict(gradient_relative_error=error,
                                                            initial_source_sha256=tensor_hash(initial)))
            evaluator = Evaluator(model, operator, observed, settings, task["dlr_seed"], products)
            result = solve(evaluator, "DLR D-Flow", initial, truth, observed,
                           case / f"{arm}.pt", protocol, [False])
            panel = plot_reconstruction_comparison(
                inputs["truth"], inputs["observed"], {"DLR D-Flow": (result["image"], result["measurement"])},
                observation=inputs["observed"], measurement_log=task["problem"] == "mri",
                title=f"Resolution: 64², Inverse Problem: {task['problem'].upper()}, Noise Level 0%, Sample: {sample}\n"
                      f"{task['count']} objects; {label}; source seed {task['seed']}")
            panel.savefig(case / f"{arm}.png", dpi=100)
            plt.close(panel)
    write_json(out / f"sample_{sample:03d}/complete.json", dict(status="complete", solves=12))


def report(out: Path) -> None:
    plan = json.loads((out / "plan.json").read_text())
    rows, groups, pairs = [], [], []
    for task in plan["tasks"]:
        records = {}
        for arm in ARMS:
            path = Path(task["case"]) / f"{arm}.json"
            if not path.exists():
                continue
            record = json.loads(path.read_text())
            if record["status"] not in TERMINAL:
                continue
            assert record["protocol"]["initial_source_sha256"] == task["hashes"][arm]
            final = record["history"][-1]
            row = dict(sample=task["sample"], count=task["count"], problem=task["problem"], seed=task["seed"],
                       arm=arm, image_percent=100 * final["image_relative_l2"],
                       measurement_percent=100 * final["measurement_relative_l2_clean"],
                       attempts=final["step"], status=record["status"], elapsed_seconds=record["elapsed_seconds"])
            rows.append(row)
            records[arm] = record
        if len(records) == len(ARMS):
            common = min(record["history"][-1]["step"] for record in records.values())
            pairs.append(dict(sample=task["sample"], count=task["count"], problem=task["problem"],
                              seed=task["seed"], common_attempt=common,
                              final={arm: 100 * record["history"][-1]["image_relative_l2"] for arm, record in records.items()},
                              common={arm: 100 * record["history"][common]["image_relative_l2"] for arm, record in records.items()}))
    figure, axes = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
    for ax, problem in zip(axes, ("ct", "mri")):
        for pair in filter(lambda item: item["problem"] == problem, pairs):
            ax.plot(range(3), list(pair["final"].values()), marker="o", alpha=.65,
                    color="tab:blue" if pair["count"] == 2 else "tab:orange")
        ax.set_xticks(range(3), LABELS, rotation=15)
        ax.set_title(f"{problem.upper()}: blue=2 objects, orange=4 objects")
        ax.set_ylabel("Final relative image error (%)")
        ax.grid(alpha=.2)
        for count in (2, 4):
            selected = [p for p in pairs if p["problem"] == problem and p["count"] == count]
            if selected:
                groups.append(dict(problem=problem, count=count, paired_cases=len(selected),
                                   median={arm: statistics.median(p["final"][arm] for p in selected) for arm in ARMS},
                                   measurement_better_than_original=sum(p["final"]["measurement_mix"] < p["final"]["original"] for p in selected),
                                   measurement_better_than_random=sum(p["final"]["measurement_mix"] < p["final"]["random_mix"] for p in selected)))
    figure.savefig(out / "paired_errors.png", dpi=150)
    plt.close(figure)
    write_json(out / "summary.json", dict(completed=len(rows), expected=plan["expected_solves"],
                                        groups=groups, pairs=pairs, runs=rows))
    if rows:
        with (out / "summary.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "worker", "report"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample", type=int)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.out)
    elif args.action == "worker":
        if args.sample is None:
            parser.error("worker requires --sample")
        worker(args.out, args.sample)
    else:
        report(args.out)


if __name__ == "__main__":
    main()
