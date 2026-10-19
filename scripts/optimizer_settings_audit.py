"""CPU investigation of empty-history scaling in capped Armijo L-BFGS.

Uses the 250px non-attention checkpoint and the existing matched test problems.
The default probe reuses cached exact gradients solely as trial directions;
every reported trial objective is recomputed on CPU. The solve mode computes
fresh exact gradients and retains the original cap. The default scaling and
numerical-progress guards match production. --policies legacy reproduces the
historical repeated-scaling rule; --legacy-progress independently selects the
historical rounded Armijo acceptance rule for comparisons.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_dflow_dlr_sparse_ct_circles_gpu as bench


class AuditedArmijo(bench.ArmijoLBFGS):
    """Identical recurrence, with an explicit empty-history trial-step policy."""

    def __init__(self, *args, scaling="once", progress="strict", **kwargs):
        super().__init__(*args, **kwargs)
        self.scaling = scaling
        self.initial_step_policy = {
            "once": bench.ArmijoLBFGS.initial_step_policy,
            "legacy": "legacy_empty_history",
            "cap": "no_l1_scaling",
        }[scaling]
        if progress not in ("strict", "legacy"):
            raise ValueError("progress must be 'strict' or explicit 'legacy'")
        self.progress_policy = ("legacy_rounded_rhs" if progress == "legacy"
                                else bench.ArmijoLBFGS.progress_policy)
        self.last_diagnostics = {}

    def step(self, closure):
        if self.state == {}:
            self.S, self.Y = [], []
            self.f, self.g = None, None
            self.consecutive_failures = 0
        if self.f is None:
            self.f, self.g = self._evaluate_gradient(closure)
            self.state["initialised"] = True
        f, g = self.f, self.g
        fresh = not self.S
        d = self._direction(g)
        gtd = g.dot(d).item()
        if not math.isfinite(gtd) or gtd >= 0:
            self.S, self.Y = [], []
            d, gtd, fresh = -g, -g.dot(g).item(), True
        scaled = ((self.scaling == "legacy" and fresh)
                  or (self.scaling == "once" and self.step_attempts == 0))
        strict_progress = self.progress_policy != "legacy_rounded_rhs"
        valid_direction = (not strict_progress
                           or (math.isfinite(f) and math.isfinite(gtd) and gtd < 0))
        t = 0.0
        if valid_direction:
            t = (min(1.0, 1.0 / g.abs().sum().item()) if scaled else 1.0) * self.lr
        d_norm = d.norm().item()
        capped = valid_direction and self.step_cap > 0 and t * d_norm > self.step_cap
        if capped:
            t = self.step_cap / d_norm
            self.capped_steps += 1
        self.last_diagnostics = {
            "history_before": len(self.S), "fresh": fresh, "l1_scaled": scaled,
            "initial_step_norm": t * d_norm, "capped": capped,
            "gradient_norm": g.norm().item(), "g_dot_direction": gtd,
        }
        x0 = self.param.detach().flatten().clone()
        evaluations, accepted = 0, False
        self.step_attempts += 1
        for _ in range(self.max_ls):
            if strict_progress and (not valid_direction or not math.isfinite(t) or t <= 0):
                break
            candidate = x0 + t * d
            if strict_progress and torch.equal(candidate, x0):
                break
            f_new = self.objective(candidate.view_as(self.param))
            evaluations += 1
            sufficient = (self._sufficient_decrease(f_new, f, t, gtd) if strict_progress
                          else math.isfinite(f_new) and f_new <= f + self.c1 * t * gtd)
            if sufficient:
                accepted = True
                break
            denominator = 2.0 * (f_new - f - gtd * t)
            t_q = -gtd * t * t / denominator if math.isfinite(f_new) and denominator > 0 else .1 * t
            t = min(max(t_q, .1 * t), .5 * t)
        self.last_function_evaluations = evaluations
        self.total_function_evaluations += evaluations
        self.last_diagnostics.update(accepted=accepted, function_evaluations=evaluations,
                                     accepted_step_norm=t * d_norm if accepted else 0.0)
        if not accepted:
            self.failed_searches += 1
            self.consecutive_failures += 1
            self.S, self.Y = [], []
            return f
        self.consecutive_failures = 0
        with torch.no_grad():
            self.param.copy_((x0 + t * d).view_as(self.param))
        f_new, g_new = self._evaluate_gradient(closure)
        s_vec, y_vec = t * d, g_new - g
        curvature = y_vec.dot(s_vec).item()
        if curvature > 1e-10:
            self.S.append(s_vec)
            self.Y.append(y_vec)
            if len(self.S) > self.history:
                self.S.pop(0)
                self.Y.pop(0)
        self.last_diagnostics.update(curvature=curvature, history_after=len(self.S))
        self.f, self.g = f_new, g_new
        return f_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["probe", "solve"], default="probe")
    parser.add_argument("--samples", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--policies", nargs="+", choices=["legacy", "once", "cap"], default=["once"],
                        help="Current policy by default; legacy explicitly reproduces historical repeated scaling")
    parser.add_argument("--legacy-progress", action="store_true",
                        help="Explicitly reproduce rounded Armijo acceptance independently of initial-step scaling")
    parser.add_argument("--outer-steps", type=int, default=10)
    parser.add_argument("--flow-steps", type=int, default=100)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--history", type=int, default=100)
    parser.add_argument("--step-cap", type=float, default=.01)
    parser.add_argument("--checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--probe-multipliers", type=float, nargs="+", default=[1., 10., 100., 267.])
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/optimizer_settings_2026_09_27/first_iteration_only/strict_progress/audit.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = {"settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "protocol": {"device": "cpu", "resolution": 250, "noise": .01,
                           "integrator": "Euler", "model": "non-attention ch16 epoch100",
                           "optimizer": "Armijo L-BFGS", "c1": 1e-4, "lr": 1.,
                           "armijo_progress_policy": ("legacy_rounded_rhs" if args.legacy_progress
                               else bench.ArmijoLBFGS.progress_policy),
                           "source_prior": "none", "gradient": "exact D-Flow",
                           "base_seed": 20260807, "measurement_seed": "base_seed + 250000 + sample",
                           "source_seed": "base_seed + 10000000 + 250000 + sample",
                           "checkpoint_sha256": hashlib.sha256((ROOT / "outputs/fm_unet/circles-250-noattn-ch16-b8/n1/checkpoints/epoch_0100.pt").read_bytes()).hexdigest()},
              "runs": []}

    def emit(event, **row):
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"event": event, **row}), flush=True)

    device = torch.device("cpu")
    model = bench.load_model(250, 100, device, "outputs/fm_unet/circles-{res}-noattn-ch16-b8/n1")
    data = torch.load(ROOT / "data/circles-250-n1.pt", map_location="cpu", weights_only=True, mmap=True)["test"]
    angles = torch.linspace(0, math.pi, 16)[:-1]
    for sample in args.samples:
        truth = bench.normalize_images(data[sample:sample + 1])
        clean = bench.sparse_radon(truth, angles)
        y, _, _ = bench.add_relative_noise(clean, .01, torch.Generator().manual_seed(20260807 + 250000 + sample))
        z0 = torch.randn(truth.shape, generator=torch.Generator().manual_seed(20260807 + 10000000 + 250000 + sample))
        accepted = {}

        def evaluate(z, grad=False):
            start = time.perf_counter()
            with torch.set_grad_enabled(grad):
                x = bench.solve_flow(model, z, args.flow_steps, "forward_euler", activation_checkpointing=grad and args.checkpoint)
                value, relative_sq = bench.data_objective(bench.sparse_radon(x, angles), y, 250, "relative_scaled")
                gradient, = torch.autograd.grad(value, z) if grad else (None,)
            state = {"objective": float(value.detach()), "residual": math.sqrt(float(relative_sq.detach())),
                     "image_relative_l2": float(bench.relative_l2(x.detach().clamp(0, 1), truth)),
                     "raw_image_relative_l2": float(bench.relative_l2(x.detach(), truth)),
                     "source_norm_over_sqrt_d": float(z.detach().norm()) / 250,
                     "evaluation_seconds": time.perf_counter() - start}
            return state, gradient

        if args.mode == "probe":
            cache = torch.load(ROOT / f"outputs/dlr_speed_2026_09_27/initial/points/res250_sample{sample}.pt",
                               map_location="cpu", weights_only=False)
            if args.flow_steps != cache["metadata"]["steps"] or not torch.equal(z0, cache["points"][0]):
                raise ValueError("Cached exact gradient does not match the requested source/protocol")
            g = cache["references"][0]["gradient"].view_as(z0)
            base, _ = evaluate(z0)
            run = {"sample": sample, "base": base, "cached_gradient_source": str(ROOT / f"outputs/dlr_speed_2026_09_27/initial/points/res250_sample{sample}.pt"),
                   "gradient_norm": float(g.norm()), "gradient_l1": float(g.abs().sum()), "trials": []}
            result["runs"].append(run)
            emit("base", sample=sample, **base)
            t0 = min(1., 1. / float(g.abs().sum()))
            for multiplier in args.probe_multipliers:
                t = min(multiplier * t0, args.step_cap * 250 / float(g.norm()))
                state, _ = evaluate(z0 - t * g)
                row = {"multiplier": multiplier, "step_norm": t * float(g.norm()),
                       "armijo_rhs": base["objective"] - 1e-4 * t * float(g.square().sum()), **state}
                row["armijo_accepted"] = (state["objective"] <= row["armijo_rhs"] if args.legacy_progress else
                    not torch.equal(z0 - t * g, z0)
                    and state["objective"] < base["objective"]
                    and state["objective"] - base["objective"] <= -1e-4 * t * float(g.square().sum()))
                row["relative_objective_drop"] = (base["objective"] - state["objective"]) / base["objective"]
                run["trials"].append(row)
                emit("trial", sample=sample, **row)
            continue

        for policy in args.policies:
            source = z0.clone().requires_grad_(True)
            accepted = {}
            counts = {"gradients": 0, "forward_trials": 0}
            run = {"sample": sample, "policy": policy, "steps": []}
            result["runs"].append(run)
            started = time.perf_counter()

            def objective(candidate):
                counts["forward_trials"] += 1
                state, _ = evaluate(candidate)
                return state["objective"]

            optimizer = AuditedArmijo(source, objective, history=args.history,
                                     step_cap=args.step_cap * 250, scaling=policy,
                                     progress="legacy" if args.legacy_progress else "strict")
            run["armijo_initial_step_policy"] = optimizer.initial_step_policy
            run["armijo_progress_policy"] = optimizer.progress_policy

            def closure():
                nonlocal accepted
                optimizer.zero_grad()
                accepted, gradient = evaluate(source, grad=True)
                source.grad = gradient
                counts["gradients"] += 1
                if counts["gradients"] == 1:
                    run["initial"] = dict(accepted)
                return accepted["objective"]

            for step in range(1, args.outer_steps + 1):
                optimizer.step(closure)
                row = {"step": step, **accepted, **optimizer.last_diagnostics, **counts,
                       "elapsed_seconds": time.perf_counter() - started}
                run["steps"].append(row)
                emit("step", sample=sample, policy=policy, **row)
                if optimizer.stalled:
                    break
            source_path = args.out.with_name(f"{args.out.stem}_s{sample}_{policy}_final.pt")
            torch.save({"source": source.detach(), "protocol": result["protocol"],
                        "sample": sample, "policy": policy, "final": run["steps"][-1]}, source_path)
            run["final_source"] = str(source_path)
    emit("complete", output=str(args.out))


if __name__ == "__main__":
    main()
