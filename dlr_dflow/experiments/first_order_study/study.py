"""Matched CPU pilot. Run with python -m dlr_dflow.experiments.first_order_study.study."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import signal
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from dlr_dflow.api import approx_source_gradient, load_flow_model, sensitivity_factors, terminal_gradient
from sampler.ode_samplers import euler_flow

Tensor = torch.Tensor
Record = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


class BudgetExpired(Exception):
    """End a phase without losing completed updates."""


def timeout_handler(signum: int, frame: Any) -> None:
    raise BudgetExpired("CPU phase deadline reached")


def project(z: Tensor) -> Tensor:
    return z * (math.sqrt(z.numel()) / z.norm())


def tangent(g: Tensor, z: Tensor, projected: bool) -> Tensor:
    return g - z * (g * z).sum() / z.square().sum() if projected else g


def candidate(z: Tensor, direction: Tensor, rate: float, projected: bool) -> Tensor:
    trial = z + rate * direction
    return project(trial) if projected else trial


class Radon:
    """15 evenly spaced rotations in [0, pi), followed by a height mean."""

    def __init__(self, size: int = 64, angles: int = 15, dtype: torch.dtype = torch.float32) -> None:
        theta = torch.arange(angles, dtype=dtype) * (math.pi / angles)
        affine = torch.zeros(angles, 2, 3, dtype=dtype)
        affine[:, 0, 0], affine[:, 0, 1] = theta.cos(), -theta.sin()
        affine[:, 1, 0], affine[:, 1, 1] = theta.sin(), theta.cos()
        self.grid = F.affine_grid(affine, (angles, 1, size, size), align_corners=False)

    def __call__(self, x: Tensor) -> Tensor:
        rotated = F.grid_sample(x.expand(self.grid.shape[0], -1, -1, -1), self.grid,
                                mode="bilinear", padding_mode="zeros", align_corners=False)
        return rotated.mean(dim=-2).transpose(0, 1).unsqueeze(0)


@dataclass
class Problem:
    model: torch.nn.Module
    truth: Tensor
    source: Tensor
    operator: Radon
    clean: Tensor
    noisy: Tensor
    seed: int
    steps: int = 100
    counts: dict[str, int] = field(default_factory=lambda: {"exact_gradients": 0, "dlr_gradients": 0, "trial_flows": 0})

    def evaluate(self, z: Tensor, estimator: str, rank: int = 8) -> tuple[Tensor, float, Tensor]:
        if estimator == "exact":
            source = z.detach().requires_grad_(True)
            x = euler_flow(self.model, source, self.steps)
            loss = .5 * (self.operator(x) - self.noisy).square().sum()
            gradient, = torch.autograd.grad(loss, source)
            self.counts["exact_gradients"] += 1
        else:
            x, factors, alpha = sensitivity_factors(
                self.model, z, self.steps, rank=rank, speed_profile="fast",
                trace_probes=0, enrichment_probes=0, initialization_oversampling=8,
                zero_terminal_baseline=True, seed=self.seed)
            gradient = approx_source_gradient(factors, terminal_gradient(x, self.operator, self.noisy), alpha)
            loss = .5 * (self.operator(x) - self.noisy).square().sum()
            self.counts["dlr_gradients"] += 1
        if not torch.isfinite(gradient).all() or not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite objective or gradient")
        return x.detach(), float(loss.detach()), gradient.detach()

    def trial(self, z: Tensor) -> tuple[Tensor, float]:
        with torch.no_grad():
            x = euler_flow(self.model, z, self.steps)
            loss = .5 * (self.operator(x) - self.noisy).square().sum()
        self.counts["trial_flows"] += 1
        return x, float(loss)

    def metrics(self, x: Tensor) -> Record:
        with torch.no_grad():
            predicted = self.operator(x)
            residual = float((predicted - self.noisy).norm())
            return {"relative_image_error": float((x - self.truth).norm() / self.truth.norm()),
                    "measurement_residual": residual,
                    "clean_residual": float((predicted - self.clean).norm()),
                    "residual_over_noise": residual / float((self.noisy - self.clean).norm())}


def make_problem(model: torch.nn.Module, data: dict[str, Tensor], split: str, index: int,
                 seed: int, operator: Radon) -> Problem:
    truth = data[split][index:index + 1].float() / 255
    local_seed = seed + (0 if split == "val" else 1000) + index
    source = torch.randn(truth.shape, generator=torch.Generator().manual_seed(local_seed))
    clean = operator(truth)
    noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(local_seed + 10000))
    noisy = clean + noise * (.01 * clean.norm() / noise.norm())
    return Problem(model, truth, source, operator, clean, noisy, local_seed + 20000)


@dataclass
class Run:
    label: str
    estimator: str
    optimizer: str
    projected: bool
    lr: float
    z: Tensor
    rank: int = 8
    sample: int = 0
    attempts: int = 0
    status: str = "ready"
    history: list[Record] = field(default_factory=list)
    sources: list[Tensor] = field(default_factory=list)
    state: Record = field(default_factory=dict)

    def public(self) -> Record:
        return {"label": self.label, "estimator": self.estimator, "optimizer": self.optimizer,
                "projected": self.projected, "lr": self.lr, "rank": self.rank, "sample": self.sample,
                "attempts": self.attempts, "status": self.status, "history": self.history}

    def advance(self, problem: Problem) -> None:
        started = time.perf_counter()
        counts = problem.counts.copy()
        x, loss, gradient = problem.evaluate(self.z, self.estimator, self.rank)
        gradient_seconds = time.perf_counter() - started
        bookkeeping_started = time.perf_counter()
        if not self.history:
            self.history.append({"iteration": 0, "phi": loss, **problem.metrics(x)})
            self.sources.append(self.z.clone())
        bookkeeping_seconds = time.perf_counter() - bookkeeping_started
        g = tangent(gradient, self.z, self.projected)
        # Build trial state separately: a deadline cannot leave a half-applied update.
        state = self.state.copy()
        if self.optimizer == "adam":
            state["m"] = .9 * state.get("m", torch.zeros_like(g)) + .1 * g
            state["v"] = .999 * state.get("v", torch.zeros_like(g)) + .001 * g.square()
            direction = -(state["m"] / (1 - .9 ** (self.attempts + 1))) / (
                (state["v"] / (1 - .999 ** (self.attempts + 1))).sqrt() + 1e-8)
        elif self.optimizer == "lbfgs":
            pairs = list(state.get("pairs", []))
            if "previous_z" in state:
                s, y = (self.z - state["previous_z"]).flatten(), (g - state["previous_g"]).flatten()
                if float(s @ y) > 1e-10:
                    pairs = (pairs + [(s, y)])[-100:]
            direction = -g.flatten().clone()
            coefficients = []
            for s, y in reversed(pairs):
                rho = 1 / (s @ y)
                coefficient = rho * (s @ direction)
                direction -= coefficient * y
                coefficients.append((rho, coefficient))
            if pairs:
                s, y = pairs[-1]
                direction *= (s @ y) / (y @ y)
            for (s, y), (rho, coefficient) in zip(pairs, reversed(coefficients)):
                direction += s * (coefficient - rho * (y @ direction))
            direction = direction.reshape_as(g)
            state.update(pairs=pairs, previous_z=self.z.clone(), previous_g=g.clone())
        else:
            direction = -g
        direction = tangent(direction, self.z, self.projected)
        slope = float((gradient * direction).sum())
        if self.optimizer != "adam" and (not math.isfinite(slope) or slope >= 0):
            direction, slope = -g, -float(g.square().sum())
            state["pairs"] = []
        # This normalization is lifetime-first-attempt only, including after failures.
        scale = min(1., 1. / max(float(gradient.abs().sum()), 1e-30)) if self.attempts == 0 else 1.
        initial_rate = self.lr * scale
        cap = .01 * math.sqrt(self.z.numel())
        rate = min(initial_rate, cap / max(float(direction.norm()), 1e-30))
        capped_rate = rate
        trial_seconds, trials = 0., 0
        accepted = False
        for _ in range(1 if self.optimizer == "adam" else 12):
            if not torch.isfinite(direction).all() or float(direction.norm()) == 0:
                break
            trial = candidate(self.z, direction, rate, self.projected)
            if torch.equal(trial, self.z):
                break
            tick = time.perf_counter()
            trial_x, value = problem.trial(trial)
            trial_seconds += time.perf_counter() - tick
            trials += 1
            accepted = math.isfinite(value) and (self.optimizer == "adam" or
                       (value < loss and value - loss <= 1e-4 * rate * slope))
            if accepted:
                break
            denominator = 2 * (value - loss - slope * rate)
            interpolated = -slope * rate * rate / denominator if math.isfinite(value) and denominator > 0 else .1 * rate
            rate = min(max(interpolated, .1 * rate), .5 * rate)
        seconds = time.perf_counter() - started - bookkeeping_seconds
        displacement = float((trial - self.z).norm()) if accepted else 0.
        if accepted:
            self.z, x, loss = trial.detach(), trial_x.detach(), value
        elif self.optimizer == "lbfgs":
            state = {}
        self.state = state
        self.attempts += 1
        self.status = "running" if accepted else "stalled"
        self.sources.append(self.z.clone())
        self.history.append({"iteration": self.attempts, "accepted": accepted, "phi": loss,
            "gradient_seconds": gradient_seconds, "seconds": seconds,
            "seconds_excluding_trials": seconds - trial_seconds, "trial_seconds": trial_seconds,
            "trials": trials, "initial_scale": scale, "uncapped_rate": initial_rate,
            "capped_initial_rate": capped_rate, "accepted_rate": rate if accepted else 0.,
            "source_step_norm": displacement, "source_norm": float(self.z.norm()),
            "ambient_gradient_norm": float(gradient.norm()), "tangent_gradient_norm": float(g.norm()),
            "predicted_slope": slope, "evaluations": {k: problem.counts[k] - counts[k] for k in counts},
            **problem.metrics(x)})


def save(out: Path, report: Record, runs: list[Run]) -> None:
    report["updated_at"] = datetime.now().astimezone().isoformat()
    temporary = out / "results.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
    temporary.replace(out / "results.json")
    torch.save({run.label: {"source": run.z, "sources": run.sources, "optimizer_state": run.state,
                           "attempts": run.attempts} for run in runs}, out / "states.tmp")
    (out / "states.tmp").replace(out / "states.pt")


def rounds(runs: list[Run], problems: list[Problem], updates: int, deadline: float,
           out: Path, report: Record, all_runs: list[Run], section: str) -> None:
    try:
        signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - time.monotonic()))
        for iteration in range(updates):
            for run in runs:
                if run.status in {"stalled", "failed"}:
                    continue
                try:
                    run.advance(problems[run.sample])
                except (FloatingPointError, RuntimeError) as error:
                    run.status = "failed"
                    report.setdefault("failures", []).append({"label": run.label, "error": str(error)})
                report[section] = [r.public() for r in runs]
                save(out, report, all_runs)
            print(f"{section}: completed round {iteration + 1}/{updates}", flush=True)
            if all(run.status in {"stalled", "failed"} for run in runs):
                break
    except BudgetExpired:
        report.setdefault("incomplete", []).append(section + ": phase deadline")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for run in runs:
            if run.status not in {"stalled", "failed"}:
                run.status = "complete" if run.attempts >= updates else "budget_limited"
        report[section] = [r.public() for r in runs]
        save(out, report, all_runs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "results" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--hours", type=float, default=4.)
    parser.add_argument("--threads", type=int, default=2, choices=(1, 2, 3, 4))
    parser.add_argument("--seed", type=int, default=3009)
    args = parser.parse_args()
    if args.hours <= 0 or args.out.exists():
        parser.error("Use a positive budget and a new output directory")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    signal.signal(signal.SIGALRM, timeout_handler)
    start = time.monotonic()
    scale = args.hours / 4.
    deadlines = [start + minutes * 60 * scale for minutes in (10, 40, 145, 190, 235, 240)]
    run_dir = ROOT / "outputs/fm_unet/circles-64-noattn-ch16-b8/n1"
    checkpoint = run_dir / "checkpoints/epoch_0100.pt"
    data_path = ROOT / "data/circles-64-n1.pt"
    architecture = json.loads((run_dir / "config.json").read_text())["arch"]
    if architecture.get("mid_attn", True) or any(architecture["att_channels"]):
        raise ValueError("Non-attention checkpoint required")
    model, _ = load_flow_model(64, epoch=100, run_dir=run_dir, device="cpu")
    data = torch.load(data_path, map_location="cpu", weights_only=True, mmap=True)
    if data["test"].shape[-2:] != (64, 64) or data["val"].shape[-2:] != (64, 64):
        raise ValueError("Expected 64x64 data")
    operator = Radon()
    validation = make_problem(model, data, "val", 0, args.seed, operator)
    problems = [make_problem(model, data, "test", i, args.seed, operator) for i in range(3)]
    args.out.mkdir(parents=True)
    report: Record = {"created_at": datetime.now().astimezone().isoformat(), "settings": {
        "hours": args.hours, "threads": args.threads, "seed": args.seed, "steps": 100,
        "angles": 15, "relative_noise": .01, "alpha": "1-t", "trace_probes": 0,
        "enrichment_probes": 0, "initialization_oversampling": 8, "factor_refresh": "every gradient",
        "initial_step_policy": "first_attempt_only", "step_cap": .64, "device": "cpu",
        "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "torch_version": str(torch.__version__), "dlr_seed": "source seed + 20000",
        "source_seed": "base + (0 validation, 1000 test) + index", "noise_seed": "source seed + 10000",
        "timing": "CPU wall time; excluding_trials subtracts all trial forwards, including Adam's single trial"},
        "incomplete": [], "failures": []}
    # Fingerprint the actual dirty-worktree dependencies without committing unrelated work.
    dependencies = list((ROOT / "dlr_dflow").glob("*.py")) + list((ROOT / "dlr_dflow/implementation").glob("*.py"))
    dependencies += list((ROOT / "models").rglob("*.py")) + list((ROOT / "sampler").glob("*.py")) + list(HERE.glob("*.py"))
    report["source_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in dependencies}
    torch.save({"validation": {"truth": validation.truth, "source": validation.source, "clean": validation.clean, "noisy": validation.noisy},
                "test": [{"truth": p.truth, "source": p.source, "clean": p.clean, "noisy": p.noisy} for p in problems]}, args.out / "inputs.pt")
    all_runs: list[Run] = []
    from .diagnostics import preflight, diagnose
    try:
        signal.setitimer(signal.ITIMER_REAL, max(.001, deadlines[0] - time.monotonic()))
        report["preflight"] = preflight(validation)
    except BudgetExpired:
        report["incomplete"].append("preflight deadline: study not started")
        save(args.out, report, [])
        return
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    print("preflight: " + json.dumps(report["preflight"]), flush=True)
    tuning = [Run(f"tune_{est}_{opt}_{proj}_{lr}", est, opt, proj, lr,
                  project(validation.source) if proj else validation.source.clone())
              for est in ("exact", "dlr") for proj in (False, True)
              for opt, rates in (("gd", (.01, .1, 1.)), ("adam", (.001, .01, .1))) for lr in rates]
    all_runs.extend(tuning)
    rounds(tuning, [validation], 8, deadlines[1], args.out, report, all_runs, "tuning")
    selected: dict[tuple[str, str, bool], float] = {}
    selections = []
    for est in ("exact", "dlr"):
        for proj in (False, True):
            matched = [r for r in tuning if (r.estimator, r.projected) == (est, proj)]
            common = min(r.attempts for r in matched)
            if common == 0:
                raise RuntimeError("Tuning did not complete one comparable update per learning rate")
            for opt in ("gd", "adam"):
                group = [r for r in tuning if (r.estimator, r.optimizer, r.projected) == (est, opt, proj)]
                # Compare every rate at the same completed budget, including stalled runs.
                winner = min(group, key=lambda r: r.history[common]["relative_image_error"])
                selected[est, opt, proj] = winner.lr
                selections.append({"estimator": est, "optimizer": opt, "projected": proj,
                                   "lr": winner.lr, "common_updates": common,
                                   "validation_error": winner.history[common]["relative_image_error"]})
    report["selected"] = selections
    comparisons = [Run(f"test{i}_{est}_{opt}_{proj}", est, opt, proj,
                       selected.get((est, opt, proj), 1.), project(p.source) if proj else p.source.clone(), sample=i)
                   for i, p in enumerate(problems) for est in ("exact", "dlr")
                   for proj in (False, True) for opt in ("gd", "adam", "lbfgs")]
    all_runs.extend(comparisons)
    rounds(comparisons, problems, 30, deadlines[2], args.out, report, all_runs, "comparisons")
    # For each constraint, select family/rate from DLR validation and freeze them for all ranks AND exact.
    rank_runs = []
    for proj in (False, True):
        choices = [row for row in selections if row["estimator"] == "dlr" and row["projected"] == proj]
        winner = min(choices, key=lambda row: row["validation_error"])
        for rank in (1, 2, 4, 8, 16, 32, 0):
            rank_runs.append(Run(f"rank{rank}_{proj}", "dlr" if rank else "exact", winner["optimizer"], proj,
                                 winner["lr"], project(problems[0].source) if proj else problems[0].source.clone(), rank=rank))
    all_runs.extend(rank_runs)
    rounds(rank_runs, problems, 10, deadlines[3], args.out, report, all_runs, "ranks")
    try:
        signal.setitimer(signal.ITIMER_REAL, max(.001, deadlines[4] - time.monotonic()))
        report["diagnostics"] = []
        for projected in (False, True):
            reference = next(r for r in comparisons if r.sample == 0 and r.estimator == "exact" and r.optimizer == "gd" and r.projected == projected)
            for index in sorted({0, reference.attempts // 2, reference.attempts}):
                if not reference.sources:
                    continue
                entry: Record = {"projected": projected, "iteration": index, "status": "running"}
                report["diagnostics"].append(entry)
                diagnose(problems[0], reference.sources[index], entry, deadlines[4], projected)
                entry["status"] = "complete"
                save(args.out, report, all_runs)
                print(f"diagnostics: projected={projected}, iteration={index}", flush=True)
    except BudgetExpired:
        report["incomplete"].append("diagnostics: phase deadline")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    report["elapsed_seconds"] = time.monotonic() - start
    save(args.out, report, all_runs)
    from .report import render
    render(args.out)
    print(f"Finished in {report['elapsed_seconds'] / 60:.1f} minutes: {args.out}", flush=True)


if __name__ == "__main__":
    main()
