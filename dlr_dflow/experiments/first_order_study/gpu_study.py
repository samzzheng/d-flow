"""Resumable GPU convergence study, independent of the four-hour CPU pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any

import torch

from dlr_dflow.api import approx_source_gradient, load_flow_model, sensitivity_factors, terminal_gradient
from sampler.ode_samplers import euler_flow
from .diagnostics import diagnose, preflight
from .study import HERE, ROOT, Problem, Radon, Record, Run, Tensor, project, tangent

TERMINAL = {"objective_plateau", "oscillatory_plateau", "step_limit", "stalled", "failed", "complete"}


class CanonicalProblem(Problem):
    """Use the same native Euler endpoint for the objective and both gradients.

    The fast sensitivity implementation has a small endpoint roundoff difference.
    An extra forward solve avoids that difference contaminating late Armijo tests.
    """

    def evaluate(self, z: Tensor, estimator: str, rank: int = 8) -> tuple[Tensor, float, Tensor]:
        if estimator == "exact":
            return super().evaluate(z, estimator, rank)
        _, factors, alpha = sensitivity_factors(
            self.model, z, self.steps, rank=rank, speed_profile="fast", trace_probes=0,
            enrichment_probes=0, initialization_oversampling=8,
            zero_terminal_baseline=True, seed=self.seed)
        with torch.no_grad():
            x = euler_flow(self.model, z, self.steps)
            loss = .5 * (self.operator(x) - self.noisy).square().sum()
        gradient = approx_source_gradient(factors, terminal_gradient(x, self.operator, self.noisy), alpha)
        self.counts["dlr_gradients"] += 1
        if alpha != 0 or not torch.isfinite(gradient).all() or not torch.isfinite(loss):
            raise FloatingPointError("Invalid DLR gradient, objective, or terminal alpha")
        return x.detach(), float(loss), gradient.detach()


def tensors_to(value: Any, device: str) -> Any:
    if isinstance(value, Tensor):
        return value.detach().to(device)
    if isinstance(value, dict):
        return {key: tensors_to(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [tensors_to(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(tensors_to(item, device) for item in value)
    return value


def stopping(history: list[Record], minimum: int, maximum: int, window: int,
             tolerance: float) -> tuple[str, Record]:
    """Two consecutive windows without progress in the best objective.

    No image ground truth or noise-floor crossing determines termination.
    A plateau is deliberately not labeled a stationary point.
    """
    steps = len(history) - 1
    evidence: Record = {}
    if steps >= minimum and steps >= 2 * window and steps % window == 0:
        best = [min(row["phi"] for row in history[:end + 1])
                for end in (steps - 2 * window, steps - window, steps)]
        changes = [(a - b) / max(abs(a), 1e-30) for a, b in zip(best, best[1:])]
        recent = [row["phi"] for row in history[-window:]]
        spread = (max(recent) - min(recent)) / max(abs(min(recent)), 1e-30)
        evidence = {"window": window, "relative_best_objective_improvements": changes,
                    "relative_recent_objective_range": spread}
        if max(changes) < tolerance:
            return ("oscillatory_plateau" if spread > .05 else "objective_plateau"), evidence
    return ("step_limit" if steps >= maximum else "running"), evidence


def checkpoint(out: Path, report: Record, run: Run | None = None,
               snapshots: dict[int, Tensor] | None = None) -> None:
    if run is not None:
        path = out / "states" / f"{run.label}.pt"
        torch.save({"run": run.public(), "source": run.z, "optimizer_state": run.state,
                    "snapshots": snapshots}, path.with_suffix(".tmp"))
        path.with_suffix(".tmp").replace(path)
    report["updated_at"] = datetime.now().astimezone().isoformat()
    path = out / "results.json"
    path.with_suffix(".tmp").write_text(json.dumps(report, indent=2, allow_nan=False))
    path.with_suffix(".tmp").replace(path)


def schedule(runs: list[Run], problems: list[CanonicalProblem], section: str,
             report: Record, args: argparse.Namespace, interrupted: list[bool]) -> None:
    snapshots = {}
    for run in runs:
        path = args.out / "states" / f"{run.label}.pt"
        snapshots[run.label] = {0: run.z.clone()}
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=True)
            for key, value in saved["run"].items():
                setattr(run, key, value)
            run.z, run.state = saved["source"], saved["optimizer_state"]
            snapshots[run.label] = saved["snapshots"]
    while any(run.status not in TERMINAL for run in runs):
        for run in runs:
            if run.status in TERMINAL:
                continue
            problem = problems[run.sample]
            run.z, run.state = run.z.to(args.device), tensors_to(run.state, args.device)
            for _ in range(10):
                if interrupted[0]:
                    break
                try:
                    run.advance(problem)
                    row = run.history[-1]
                    if row["source_step_norm"] > .64001 or (run.projected and abs(row["source_norm"] - 64) > 2e-5):
                        raise AssertionError("Source cap or sphere invariant failed")
                    if run.attempts > 1 and row["initial_scale"] != 1:
                        raise AssertionError("Initial scaling repeated")
                    if row["evaluations"]["dlr_gradients" if run.estimator == "dlr" else "exact_gradients"] != 1:
                        raise AssertionError("Gradient was not freshly recomputed")
                    if run.optimizer != "adam" and row["accepted"] and row["phi"] >= run.history[-2]["phi"]:
                        raise AssertionError("Accepted Armijo update did not decrease objective")
                    # Sources and optimizer histories stay on the host between configurations.
                    run.sources.clear()
                    if run.attempts % 25 == 0:
                        snapshots[run.label][run.attempts] = run.z.cpu().clone()
                    if run.status == "stalled" and run.optimizer == "lbfgs":
                        # Run.advance already resets its history. Allow one retry after the reset.
                        previous_failed = len(run.history) > 2 and not run.history[-2]["accepted"]
                        if not previous_failed:
                            run.status = "running"
                            row["lbfgs_reset_retry"] = True
                    if run.status not in TERMINAL:
                        if section == "tuning":
                            run.status = "complete" if run.attempts >= args.tuning_updates else "running"
                        else:
                            run.status, evidence = stopping(run.history, args.min_updates, args.max_updates,
                                                            args.window, args.plateau_relative)
                            if evidence:
                                report.setdefault("stopping_evidence", {})[run.label] = evidence
                    if run.status in TERMINAL:
                        break
                except (RuntimeError, FloatingPointError) as error:
                    run.status = "failed"
                    report.setdefault("failures", []).append({"label": run.label, "error": str(error)})
                    break
            run.z, run.state = run.z.cpu(), tensors_to(run.state, "cpu")
            report[section] = [r.public() for r in runs]
            checkpoint(args.out, report, run, snapshots[run.label])
            print(f"{section}: {run.label} update={run.attempts} {run.status} "
                  f"error={run.history[-1]['relative_image_error'] if run.history else None}", flush=True)
            if interrupted[0]:
                raise KeyboardInterrupt
        if getattr(args, "render_reports", True):
            from .gpu_report import render
            render(args.out)
        if getattr(args, "one_chunk", False):
            break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "results/gpu_convergence_20260930")
    parser.add_argument("--pilot", type=Path, default=HERE / "results/pilot_20260930")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tuning-updates", type=int, default=200)
    parser.add_argument("--min-updates", type=int, default=500)
    parser.add_argument("--max-updates", type=int, default=2000)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--plateau-relative", type=float, default=.001)
    parser.add_argument("--memory-fraction", type=float, default=.15)
    parser.add_argument("--gpus", type=int, nargs="+", help="Use a process pool on these visible GPU indices")
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    args = parser.parse_args()
    if args.out.exists() != args.resume:
        parser.error("Use a new directory, or --resume with an existing directory")
    if not 0 < 2 * args.window <= args.min_updates <= args.max_updates or args.tuning_updates < 1:
        parser.error("Require 0 < 2*window <= min-updates <= max-updates and positive tuning budget")
    if args.gpus and (len(set(args.gpus)) != len(args.gpus) or not 0 < len(args.gpus) * args.workers_per_gpu <= 4):
        parser.error("Use unique GPUs and one to four total workers")
    torch.set_num_threads(1 if args.gpus else 2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    interrupted = [False]

    def stop(signum: int, frame: Any) -> None:
        interrupted[0] = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    pilot = json.loads((args.pilot / "results.json").read_text())
    settings = pilot["settings"].copy()
    settings.pop("hours", None)
    settings.pop("timing", None)
    settings.update(device=args.device, threads=2, tuning_updates=args.tuning_updates,
                    min_updates=args.min_updates, max_updates=args.max_updates, window=args.window,
                    plateau_relative=args.plateau_relative, memory_fraction=args.memory_fraction,
                    tf32=False, canonical_dlr_endpoint=True, source_checkpoint_interval=25,
                    dlr_initialization="Device-local seeded sketches, fresh factors every gradient",
                    terminal_policy="Two objective-plateau windows after minimum; stalls separate; hard step limit",
                    git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    model_path = Path(settings["checkpoint"])
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != settings["checkpoint_sha256"]:
        raise ValueError("Checkpoint no longer matches paired pilot")
    model, _ = load_flow_model(64, epoch=100, run_dir=model_path.parents[1], device=args.device)
    inputs = torch.load(args.pilot / "inputs.pt", map_location="cpu", weights_only=True)
    op = Radon()
    op.grid = op.grid.to(args.device)
    validation = CanonicalProblem(model, inputs["validation"]["truth"].to(args.device),
        inputs["validation"]["source"].to(args.device), op, inputs["validation"]["clean"].to(args.device),
        inputs["validation"]["noisy"].to(args.device), settings["seed"] + 20000)
    problems = [CanonicalProblem(model, row["truth"].to(args.device), row["source"].to(args.device),
        op, row["clean"].to(args.device), row["noisy"].to(args.device), settings["seed"] + 21000 + i)
        for i, row in enumerate(inputs["test"])]
    if args.resume:
        report = json.loads((args.out / "results.json").read_text())
        for key in ("tuning_updates", "min_updates", "max_updates", "window", "plateau_relative", "checkpoint_sha256"):
            if report["settings"][key] != settings[key]:
                raise ValueError(f"Resume protocol mismatch: {key}")
        report["status"] = "running"
        report.setdefault("resumes", []).append({"at": datetime.now().astimezone().isoformat(),
            "git_head": settings["git_head"],
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    else:
        (args.out / "states").mkdir(parents=True)
        torch.save(inputs, args.out / "inputs.pt")
        dependencies = list((ROOT / "dlr_dflow").glob("*.py")) + list((ROOT / "dlr_dflow/implementation").glob("*.py"))
        dependencies += list((ROOT / "models").rglob("*.py")) + list((ROOT / "sampler").glob("*.py")) + list(HERE.glob("*.py"))
        report = {"created_at": datetime.now().astimezone().isoformat(), "settings": settings,
                  "failures": [], "status": "running", "source_sha256": {
                      str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in dependencies}}
        checkpoint(args.out, report)

    def record_exception(kind: type[BaseException], error: BaseException, trace: TracebackType | None) -> None:
        report["status"] = "interrupted" if issubclass(kind, KeyboardInterrupt) else "driver_failed"
        report["driver_message"] = str(error)
        checkpoint(args.out, report)
        sys.__excepthook__(kind, error, trace)

    sys.excepthook = record_exception
    if "preflight" not in report:
        report["preflight"] = preflight(validation)
        # Timing is deliberately omitted from this accuracy-focused report.
        report["preflight"] = {key: value for key, value in report["preflight"].items() if not key.endswith("seconds")}
        checkpoint(args.out, report)
        print("GPU preflight: " + json.dumps(report["preflight"]), flush=True)
    torch.manual_seed(settings["seed"])
    tuning = [Run(f"tune_{est}_{opt}_{proj}_{lr}", est, opt, proj, lr,
                  (project(validation.source) if proj else validation.source).cpu().clone())
              for est in ("exact", "dlr") for proj in (False, True)
              for opt, rates in (("gd", (.01, .1, 1.)), ("adam", (.001, .01, .1))) for lr in rates]
    parallel = None
    if args.gpus:
        import atexit
        from .gpu_parallel import ParallelStudy
        parallel = ParallelStudy(args, report)
        atexit.register(parallel.close)
        parallel.run({"tuning": tuning}, report, interrupted)
    else:
        schedule(tuning, [validation], "tuning", report, args, interrupted)
    selections = []
    for est in ("exact", "dlr"):
        for proj in (False, True):
            for opt in ("gd", "adam"):
                group = [r for r in tuning if (r.estimator, r.projected, r.optimizer) == (est, proj, opt) and r.status != "failed"]
                if not group:
                    raise RuntimeError(f"All validation rates failed: {est}, {proj}, {opt}")
                winner = min(group, key=lambda r: r.history[-1]["relative_image_error"])
                selections.append({"estimator": est, "projected": proj, "optimizer": opt, "lr": winner.lr,
                    "validation_error": winner.history[-1]["relative_image_error"], "attempts": winner.attempts,
                    "status": winner.status, "selection_budget": args.tuning_updates,
                    "stalled_policy": "Hold terminal iterate through the fixed validation budget"})
    report["selected"] = selections
    selected = {(r["estimator"], r["optimizer"], r["projected"]): r["lr"] for r in selections}
    comparisons = [Run(f"test{i}_{est}_{opt}_{proj}", est, opt, proj,
        selected.get((est, opt, proj), 1.), (project(p.source) if proj else p.source).cpu().clone(), sample=i)
        for i, p in enumerate(problems) for est in ("exact", "dlr")
        for proj in (False, True) for opt in ("gd", "adam", "lbfgs")]
    ranks = []
    for proj in (False, True):
        winner = min((row for row in selections if row["estimator"] == "dlr" and row["projected"] == proj),
                     key=lambda row: row["validation_error"])
        for rank in (1, 2, 4, 8, 16, 32, 0):
            ranks.append(Run(f"rank{rank}_{proj}", "dlr" if rank else "exact", winner["optimizer"], proj,
                winner["lr"], (project(problems[0].source) if proj else problems[0].source).cpu().clone(), rank=rank))
    if parallel:
        parallel.run({"comparisons": comparisons, "ranks": ranks}, report, interrupted)
    else:
        schedule(comparisons, problems, "comparisons", report, args, interrupted)
        schedule(ranks, problems, "ranks", report, args, interrupted)
    # Check actual stationarity, including runs that stopped by line-search failure.
    for run in comparisons + ranks:
        if not run.history:
            continue
        if run.label in report.get("terminal_checks", {}):
            continue
        z = run.z.to(args.device)
        x, loss, exact = problems[run.sample].evaluate(z, "exact")
        norm = float(tangent(exact, z, run.projected).norm())
        initial_norm = run.history[1]["tangent_gradient_norm"]
        report.setdefault("terminal_checks", {})[run.label] = {
            "exact_tangent_gradient_norm": norm,
            "ratio_to_initial_estimator_gradient_norm": norm / initial_norm if initial_norm else None,
            "native_objective": loss, **problems[run.sample].metrics(x)}
        checkpoint(args.out, report)
    diagnostic_jobs = []
    for projected in (False, True):
        run = next(r for r in comparisons if r.sample == 0 and r.estimator == "exact" and r.optimizer == "gd" and r.projected == projected)
        saved = torch.load(args.out / "states" / f"{run.label}.pt", map_location="cpu", weights_only=True)
        sources = {**saved["snapshots"], run.attempts: saved["source"]}
        midpoint = min(sources, key=lambda index: abs(index - run.attempts / 2))
        for iteration in sorted({0, midpoint, run.attempts}):
            if any(e["projected"] == projected and e["iteration"] == iteration and e["status"] == "complete"
                   for e in report.get("diagnostics", [])):
                continue
            entry = {"projected": projected, "iteration": iteration, "status": "running", "source_run": run.label}
            if parallel:
                diagnostic_jobs.append(entry)
                continue
            report.setdefault("diagnostics", []).append(entry)
            diagnose(problems[0], sources[iteration].to(args.device), entry, math.inf, projected,
                     product_batch_size=17, spectral_seconds=1800, max_refinements=12, refine_cutoff=True)
            entry.pop("seconds", None)
            entry["status"] = "complete"
            checkpoint(args.out, report)
            print(f"Diagnostics: {run.label}, {iteration}, {entry['spectral']['status']}", flush=True)
            if interrupted[0]:
                raise KeyboardInterrupt
    if parallel:
        parallel.diagnostics(diagnostic_jobs, report, interrupted)
        parallel.close()
    report["status"] = "complete"
    checkpoint(args.out, report)
    from .gpu_report import render
    render(args.out)
    now = datetime.now().astimezone()
    dated = ROOT / "experiments" / f"experiments-{now:%Y-%m-%d}.md"
    if not dated.exists():
        dated.write_text(f"# Experiments — {now:%Y-%m-%d}\n\nTimes are local America/New_York; durations in HH:MM.\n\n"
                         "| Time | Experiment | Motivation | What Tried / Evidence | Correctness | Timing | Merge decision |\n"
                         "| --- | --- | --- | --- | --- | --- | --- |\n")
    elapsed_minutes = int((now - datetime.fromisoformat(report["created_at"])).total_seconds() / 60)
    counts = {status: sum(r.status == status for r in comparisons + ranks)
              for status in sorted({r.status for r in comparisons + ranks})}
    with dated.open("a") as stream:
        stream.write(f"| {now:%H:%M} | Longer GPU convergence study completed | Short pilot trajectories did not establish convergence. "
            f"| Retuned for {args.tuning_updates} updates; reconstruction/rank statuses: {counts}. "
            f"[Accuracy report](../{args.out.relative_to(ROOT)}/report.md); shared-source gradient and clean-noise diagnostics repeated. "
            "| GPU preflight passed; source caps, projection, fresh gradients, initial scaling, and accepted Armijo decrease checked during updates. "
            f"Stalls and step limits remain distinct from objective plateaus. | {elapsed_minutes // 60:02d}:{elapsed_minutes % 60:02d} "
            "wall elapsed; no speed comparison. | Keep isolated evidence; inspect terminal statuses before drawing convergence conclusions. |\n")
    index = ROOT / "EXPERIMENTS.md"
    text = index.read_text()
    row = f"| {now:%Y-%m-%d} | [Experiments](experiments/experiments-{now:%Y-%m-%d}.md) |"
    if row not in text:
        index.write_text(text.replace("\n\nDated log header:", f"\n{row}\n\nDated log header:"))
    print(f"Completed GPU convergence study: {args.out}", flush=True)


if __name__ == "__main__":
    main()
