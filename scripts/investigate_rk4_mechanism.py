"""Why does RK4 not beat forward Euler in noisy sparse-CT reconstruction?

Mechanism tests, all on the attention-free ch16 models and the same problem
definition as the 250px Euler-vs-RK4 benchmark (15 views, 1% noise, capped
Armijo L-BFGS, no source prior, first-iteration-only L1 scaling):

  solve      one (sample, integrator, flow steps) reconstruction with every
             accepted source/image saved, so semi-convergence and error
             decomposition can be studied post hoc. Four generators separate
             "integrator family" from "ODE accuracy": Euler-100, Euler-400,
             RK4-25 (same 100 velocity evaluations as Euler-100), RK4-100.
  radon      build the explicit 15-view Radon matrix A (m x d) once.
  decompose  per accepted step: split the image error into the part A can see
             (range of A^T) and the part A cannot see (null space of A), and
             split the visible part into the noise-fitting component and the
             rest. Works on solve outputs and on the noise-floor pilot states.
  cross      push saved sources through other generators (Euler-100, RK4-100,
             RK4-800 reference): is Euler's solution a good source for the
             true flow, or only for the Euler-discretised generator?
  spectra    top singular values of the generator Jacobian dG/dz per
             integrator at given sources (randomized SVD with forward-mode
             JVPs and batched reverse-mode VJPs).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_dflow_dlr_sparse_ct_circles_gpu as bench  # noqa: E402

TEMPLATE = "outputs/fm_unet/circles-{res}-noattn-ch16-b8/n1"
BASE_SEED = 20260807
OUT = ROOT / "outputs/rk4_mechanism_2026_09_27"
OPTIMIZER_ATTRIBUTES = ("S", "Y", "f", "g", "state", "step_attempts", "capped_steps",
                        "consecutive_failures", "last_function_evaluations",
                        "total_function_evaluations", "failed_searches")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_save(value, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def write_json(value, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=1) + "\n")
    tmp.replace(path)


def problem(res: int, sample: int, device: torch.device) -> dict:
    """Same seeds, noise definition and initial source as the benchmark."""
    dataset = torch.load(ROOT / f"data/circles-{res}-n1.pt", map_location="cpu",
                         weights_only=True, mmap=True)["test"]
    truth = bench.normalize_images(dataset[sample:sample + 1])
    angles = torch.linspace(0, math.pi, 16)[:-1]
    clean = bench.sparse_radon(truth, angles)
    observed, noise, _ = bench.add_relative_noise(
        clean, 0.01, torch.Generator().manual_seed(BASE_SEED + res * 1000 + sample))
    source0 = torch.randn(truth.shape, generator=torch.Generator().manual_seed(
        BASE_SEED + 10_000_000 + res * 1000 + sample))
    return {k: v.to(device) for k, v in dict(truth=truth, angles=angles, clean=clean,
                                             observed=observed, noise=noise,
                                             source0=source0).items()}


def load_model(res: int, device: torch.device, epoch: int = 100, template: str = None):
    model = bench.load_model(res, epoch, device, template or TEMPLATE)
    assert not model.mid_attn and not any(model.att_channels)
    return model


def generator_label(method: str, steps: int) -> str:
    return f"{'euler' if method == 'forward_euler' else 'rk4'}{steps}"


def parse_generator(label: str) -> tuple[str, int]:
    if label.startswith("euler"):
        return "forward_euler", int(label[5:])
    if label.startswith("rk4"):
        return "rk4", int(label[3:])
    raise ValueError(label)


# ---------------------------------------------------------------------------
# solve
# ---------------------------------------------------------------------------
def cmd_solve(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if args.no_tf32:
            # exact fp32 convolutions: with TF32 the objective is non-monotone at
            # the 1e-3 level and the strict Armijo rule stalls near the noise floor
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(args.threads)
    res = args.resolution
    method, steps = parse_generator(args.generator)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "accepted").mkdir(exist_ok=True)
    checkpoint_path = out / "resume.pt"
    model = load_model(res, device, epoch=args.model_epoch, template=args.model_template)
    P = problem(res, args.sample, device)
    noise_norm = P["noise"].norm().item()
    truth_norm = P["truth"].norm().item()
    if args.init_source:
        init = torch.load(args.init_source, map_location="cpu", weights_only=False)["source"].to(device)
        source = init.clone().requires_grad_(True)
    else:
        source = P["source0"].clone().requires_grad_(True)
    cap = args.cap * math.sqrt(source.numel())
    protocol = dict(resolution=res, sample=args.sample, method=method, flow_steps=steps,
                    generator=args.generator, init_source=args.init_source, target=args.target,
                    tf32=(not args.no_tf32) if device.type == "cuda" else None,
                    velocity_evaluations_per_flow=steps * (4 if method == "rk4" else 1),
                    device=str(device), activation_checkpointing=args.checkpointing,
                    noise_relative_clean=0.01, noise_norm=noise_norm,
                    optimizer="ArmijoLBFGS", initial_step_policy=bench.ArmijoLBFGS.initial_step_policy,
                    progress_policy=bench.ArmijoLBFGS.progress_policy, history=100, lr=1.0,
                    step_cap=cap, cap_per_sqrt_d=args.cap, c1=1e-4, max_ls=30,
                    max_steps=args.max_steps, stop_ratio=args.stop_ratio,
                    objective="relative_scaled", source_prior="none", torch=torch.__version__)
    counts = {"gradients": 0, "forward_trials": 0}
    accepted: dict = {}
    report = {"protocol": protocol, "started_utc": now(), "history": [], "attempts": [],
              "first_crossings": {}, "stop_reason": None}
    elapsed_before, started = 0.0, time.perf_counter()

    def loss(x: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
        if args.target == "image":
            # oracle fit: how close can this generator's range get to the truth?
            return (x - P["truth"]).square().sum() / P["truth"].square().sum() * float(res * res)
        value, _ = bench.data_objective(predicted, P["observed"], res, "relative_scaled")
        return value

    def objective(candidate: torch.Tensor) -> float:
        with torch.no_grad():
            x = bench.solve_flow(model, candidate, steps, method, False)
            value = loss(x, bench.sparse_radon(x, P["angles"]))
        counts["forward_trials"] += 1
        return float(value)

    optimizer = bench.ArmijoLBFGS(source, objective, lr=1.0, history=100, max_ls=30,
                                  c1=1e-4, step_cap=cap)

    products = None
    if args.gradient == "dlr":
        if method != "forward_euler":
            raise ValueError("--gradient dlr uses the Euler flow")
        from dlr_dflow import dlr_dflow as core, enriched, tangent_unet as tangent
        from dlr_dflow.graphed_products import GraphedAProducts
        bench.configure_dlr_kernels(res)
        tangent.GN_ACTION_BACKEND = "triton"; tangent.GN_PRIMAL_BACKEND = "full"
        enriched.SPECIALIZE_ZERO_ENRICHMENT = True; enriched.SKIP_UNTRUNCATED_SVD = True
        core.SHARE_LINEARIZATION = True
        products = GraphedAProducts(model, source.detach(), share_linearization=True,
            full_step_graphs=True, max_jvp_columns=20, max_vjp_columns=16,
            max_step_jvp_columns=6, max_step_vjp_columns=4)
        protocol.update(gradient="dlr", dlr_rank=args.dlr_rank, dlr_trace_probes=args.dlr_trace_probes,
                        dlr_config="optimized fast profile: fused norm/primal, specialized algebra, 6/4 chunks")

    def closure():
        optimizer.zero_grad()
        if products is not None:
            x, factors, _ = core.sensitivity_factors(model, source.detach(), steps, rank=args.dlr_rank,
                trace_probes=args.dlr_trace_probes, enrichment_probes=0, initialization_oversampling=8,
                seed=0, factor_stride=1, zero_terminal_baseline=True, products=products)
            xt = x.detach().reshape_as(source).requires_grad_(True)
            predicted = bench.sparse_radon(xt, P["angles"])
            value = loss(xt, predicted)
            gx, = torch.autograd.grad(value, xt)
            source.grad = core.approx_source_gradient(factors, gx.flatten(), 0.).reshape_as(source).detach()
            counts["gradients"] += 1
            accepted.update(x=xt.detach(), predicted=predicted.detach(), objective=float(value))
            return value.detach()
        x = bench.solve_flow(model, source, steps, method, args.checkpointing)
        predicted = bench.sparse_radon(x, P["angles"])
        value = loss(x, predicted)
        grad, = torch.autograd.grad(value, source)
        source.grad = grad.detach()
        counts["gradients"] += 1
        accepted.update(x=x.detach(), predicted=predicted.detach(), objective=float(value))
        return value.detach()

    previous_source = {"z": source.detach().clone()}

    def metrics(attempt: int) -> dict:
        raw = accepted["x"]
        clamped = raw.clamp(0, 1)
        pred_raw, pred_cl = accepted["predicted"], bench.sparse_radon(clamped, P["angles"])
        r_raw = (pred_raw - P["observed"]).norm().item()
        ae = pred_raw - P["clean"]                      # A (x - x_true), raw image
        noise_cos = (ae.flatten() @ P["noise"].flatten() / (ae.norm() * noise_norm + 1e-30)).item()
        noise_fit = (ae.flatten() @ P["noise"].flatten() / noise_norm ** 2).item()
        z = source.detach()
        step_norm = (z - previous_source["z"]).norm().item()
        previous_source["z"] = z.clone()
        mse = (clamped - P["truth"]).square().mean().item()
        return dict(attempt=attempt, accepted_index=counts["gradients"] - 1,
                    objective=accepted["objective"],
                    residual_over_noise=r_raw / noise_norm,
                    residual_relative_observed=r_raw / P["observed"].norm().item(),
                    image_relative_l2=bench.relative_l2(clamped, P["truth"]).item(),
                    raw_image_relative_l2=bench.relative_l2(raw, P["truth"]).item(),
                    psnr_db=-10 * math.log10(max(mse, 1e-30)),
                    clean_residual_raw=bench.relative_l2(pred_raw, P["clean"]).item(),
                    clean_residual_clamped=bench.relative_l2(pred_cl, P["clean"]).item(),
                    noisy_residual_clamped=bench.relative_l2(pred_cl, P["observed"]).item(),
                    noise_cosine=noise_cos, noise_fit_fraction=noise_fit,
                    source_norm_over_sqrt_d=z.norm().item() / math.sqrt(z.numel()),
                    source_step_norm=step_norm, gradient_norm=optimizer.g.norm().item(),
                    history_length=len(optimizer.S), capped_steps=optimizer.capped_steps,
                    failed_searches=optimizer.failed_searches,
                    line_search_evaluations=optimizer.last_function_evaluations,
                    function_evaluations=optimizer.total_function_evaluations,
                    step_attempts=optimizer.step_attempts,
                    elapsed_seconds=elapsed_before + time.perf_counter() - started, **counts)

    def save(event: str, row: dict | None = None) -> None:
        report["elapsed_seconds"] = elapsed_before + time.perf_counter() - started
        atomic_save({"protocol": protocol, "source": source.detach().cpu(),
                     "accepted": {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in accepted.items()},
                     "optimizer": {k: getattr(optimizer, k) for k in OPTIMIZER_ATTRIBUTES},
                     "previous_source": previous_source["z"].cpu(),
                     "counts": counts, "report": report}, checkpoint_path)
        write_json(report, out / "results.json")
        print(json.dumps({"event": event, "generator": args.generator, "sample": args.sample,
                          **({} if row is None else {k: row[k] for k in (
                              "attempt", "residual_over_noise", "image_relative_l2",
                              "noise_fit_fraction", "source_step_norm", "history_length",
                              "line_search_evaluations", "elapsed_seconds") if k in row})}),
              flush=True)

    def record(attempt: int) -> None:
        row = metrics(attempt)
        report["history"].append(row)
        report["attempts"].append({"attempt": attempt, "accepted": True,
                                   "residual_over_noise": row["residual_over_noise"]})
        path = out / "accepted" / f"state_{row['accepted_index']:05d}.pt"
        atomic_save({"source": source.detach().cpu(), "raw_reconstruction": accepted["x"].cpu(),
                     "metrics": row, "protocol": protocol}, path)
        for tau in (1.0, 1.05, 1.1, 1.2):
            if str(tau) not in report["first_crossings"] and row["residual_over_noise"] <= tau:
                report["first_crossings"][str(tau)] = dict(row, saved_state=str(path))
        save("accepted", row)

    if args.resume and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        budget_keys = {"max_steps", "stop_ratio"}      # a resume may extend the budget
        differing = {k: (saved["protocol"].get(k), protocol.get(k))
                     for k in (set(saved["protocol"]) | set(protocol)) - budget_keys
                     if saved["protocol"].get(k) != protocol.get(k)}
        if differing:
            raise ValueError(f"resume protocol mismatch: {differing}")
        with torch.no_grad():
            source.copy_(saved["source"].to(device))
        accepted = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in saved["accepted"].items()}
        counts, report = saved["counts"], saved["report"]
        for name, value in saved["optimizer"].items():
            if torch.is_tensor(value):
                value = value.to(device)
            elif isinstance(value, list):
                value = [v.to(device) for v in value]
            setattr(optimizer, name, value)
        previous_source["z"] = saved["previous_source"].to(device)
        elapsed_before, started = report["elapsed_seconds"], time.perf_counter()
        report["stop_reason"] = None
        report.setdefault("resume_events", []).append({"utc": now(), "attempt": optimizer.step_attempts})
        print(json.dumps({"event": "resumed", "attempt": optimizer.step_attempts}), flush=True)
    else:
        if checkpoint_path.exists():
            raise ValueError(f"{out} already has a checkpoint; pass --resume or use a new --out")
        optimizer.f, optimizer.g = optimizer._evaluate_gradient(closure)
        optimizer.state["initialised"] = True
        record(0)

    while optimizer.step_attempts < args.max_steps:
        if report["history"][-1]["residual_over_noise"] <= args.stop_ratio:
            report["stop_reason"] = "stop_ratio"
            break
        if elapsed_before + time.perf_counter() - started > args.max_minutes * 60:
            report["stop_reason"] = "wall_budget"
            break
        before = counts["gradients"]
        optimizer.step(closure)
        if counts["gradients"] > before:
            record(optimizer.step_attempts)
        else:
            report["attempts"].append({"attempt": optimizer.step_attempts, "accepted": False,
                                       "residual_over_noise": report["history"][-1]["residual_over_noise"]})
            save("failed_search", {"attempt": optimizer.step_attempts})
        if optimizer.stalled:
            report["stop_reason"] = "two_failed_searches"
            break
    if report["stop_reason"] is None:
        report["stop_reason"] = "step_budget"
    save("complete", None)
    print(json.dumps({"event": "done", "stop_reason": report["stop_reason"],
                      "first_crossings": {k: v["attempt"] for k, v in report["first_crossings"].items()},
                      "final": {k: report["history"][-1][k] for k in (
                          "attempt", "residual_over_noise", "image_relative_l2")}}), flush=True)


# ---------------------------------------------------------------------------
# radon: explicit A
# ---------------------------------------------------------------------------
def radon_path(res: int) -> Path:
    return OUT / f"radon_{res}_15views.pt"


def cmd_radon(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    res = args.resolution
    angles = torch.linspace(0, math.pi, 16, device=device)[:-1]
    d = res * res
    columns = []
    batch = args.batch
    with torch.no_grad():
        for start in range(0, d, batch):
            idx = torch.arange(start, min(start + batch, d), device=device)
            basis = torch.zeros(idx.numel(), d, device=device)
            basis[torch.arange(idx.numel(), device=device), idx] = 1.0
            proj = bench.sparse_radon(basis.view(-1, 1, res, res), angles)  # (b, 1, 15, res)
            columns.append(proj.reshape(idx.numel(), -1).T.cpu())            # (m, b)
    A = torch.cat(columns, dim=1)                                             # (m, d)
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_save({"A": A, "resolution": res, "angles": 15, "m": A.shape[0], "d": d}, radon_path(res))
    # sanity: A x for a random image vs sparse_radon
    x = torch.rand(1, 1, res, res, device=device)
    direct = bench.sparse_radon(x, angles).reshape(-1).cpu()
    via = A @ x.reshape(-1).cpu()
    print(f"A: {tuple(A.shape)}  check rel err {(direct - via).norm() / direct.norm():.2e}")


class Projector:
    """Range/null-space split for the explicit measurement operator A."""

    def __init__(self, res: int, device: torch.device):
        payload = torch.load(radon_path(res), map_location="cpu", weights_only=True)
        self.A = payload["A"].to(device)                     # (m, d)
        self.d = payload["d"]
        gram = self.A @ self.A.T                            # (m, m)
        evals, evecs = torch.linalg.eigh(gram.double())
        keep = evals > evals.max() * 1e-10
        self.rank = int(keep.sum())
        # pseudo-inverse of A A^T on its range
        self.gram_pinv = (evecs[:, keep] / evals[keep]) @ evecs[:, keep].T
        self.gram_pinv = self.gram_pinv.float()

    def split(self, e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """e (d,) -> (visible = A^+ A e, invisible = e - visible)."""
        v = self.A @ e
        visible = self.A.T @ (self.gram_pinv @ v)
        return visible, e - visible


# ---------------------------------------------------------------------------
# decompose: per-step error split on saved accepted states
# ---------------------------------------------------------------------------
def list_states(states_dir: Path) -> list[Path]:
    return sorted(states_dir.glob("state_*.pt"))


def cmd_decompose(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    res = args.resolution
    P = problem(res, args.sample, device)
    proj = Projector(res, device)
    truth = P["truth"].flatten()
    truth_norm = truth.norm().item()
    noise = P["noise"].flatten()
    noise_norm = noise.norm().item()
    A = proj.A
    clean = A @ truth
    y = clean + noise
    rows = []
    for path in list_states(args.states):
        s = torch.load(path, map_location=device, weights_only=False)
        raw = s["raw_reconstruction"].flatten().to(device)
        m = s["metrics"]
        row = {"accepted_index": m.get("accepted_index"), "attempt": m.get("attempt"),
               "residual_over_noise": m.get("residual_over_noise"),
               "elapsed_seconds": m.get("elapsed_seconds")}
        for kind, x in (("raw", raw), ("clamped", raw.clamp(0, 1))):
            e = x - truth
            vis, inv = proj.split(e)
            ae = A @ e                                  # visible error in measurement space
            beta = (ae @ noise) / noise_norm ** 2       # fraction of the noise fitted
            ae_orth = ae - beta * noise
            row.update({
                f"{kind}_image_rel": e.norm().item() / truth_norm,
                f"{kind}_visible_rel": vis.norm().item() / truth_norm,
                f"{kind}_invisible_rel": inv.norm().item() / truth_norm,
                f"{kind}_noisy_residual_over_noise": (A @ x - y).norm().item() / noise_norm,
                f"{kind}_clean_residual_over_noise": ae.norm().item() / noise_norm,
                f"{kind}_noise_fit_fraction": beta.item(),
                f"{kind}_visible_orth_over_noise": ae_orth.norm().item() / noise_norm,
                f"{kind}_noise_cosine": (ae @ noise / (ae.norm() * noise_norm + 1e-30)).item(),
            })
        # the noise-fitting image component itself: the visible image change that
        # produces beta*noise, i.e. A^+ (beta n)
        rows.append(row)
    write_json({"states": str(args.states), "resolution": res, "sample": args.sample,
                "radon_rank": proj.rank, "rows": rows}, args.out)
    last = rows[-1]
    print(f"{args.states}: {len(rows)} states; final clamped image {100*last['clamped_image_rel']:.3f}% "
          f"= visible {100*last['clamped_visible_rel']:.3f}% + invisible {100*last['clamped_invisible_rel']:.3f}%; "
          f"noise fit {last['raw_noise_fit_fraction']:.3f}")


# ---------------------------------------------------------------------------
# cross: evaluate saved sources under other generators
# ---------------------------------------------------------------------------
def cmd_cross(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    res = args.resolution
    model = load_model(res, device, epoch=args.model_epoch, template=args.model_template)
    P = problem(res, args.sample, device)
    noise_norm = P["noise"].norm().item()
    proj = Projector(res, device) if radon_path(res).exists() else None
    truth = P["truth"].flatten()
    results = {"resolution": res, "sample": args.sample, "sources": {}, "images": {}}
    images = {}
    for label, path in (item.split("=", 1) for item in args.sources):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        z = payload["source"].to(device)
        results["sources"][label] = {"path": path, "norm_over_sqrt_d": z.norm().item() / math.sqrt(z.numel())}
        for gen in args.generators:
            method, steps = parse_generator(gen)
            t0 = time.perf_counter()
            with torch.no_grad():
                x = bench.solve_flow(model, z, steps, method, False)
            seconds = time.perf_counter() - t0
            pred = bench.sparse_radon(x, P["angles"])
            ae = (pred - P["clean"]).flatten()
            e = x.flatten() - truth
            e_cl = x.clamp(0, 1).flatten() - truth
            entry = dict(seconds=seconds,
                         image_relative_l2=e_cl.norm().item() / truth.norm().item(),
                         raw_image_relative_l2=e.norm().item() / truth.norm().item(),
                         residual_over_noise=(pred - P["observed"]).norm().item() / noise_norm,
                         clean_residual_over_noise=ae.norm().item() / noise_norm,
                         noise_fit_fraction=(ae @ P["noise"].flatten()).item() / noise_norm ** 2)
            if proj is not None:
                vis, inv = proj.split(e_cl)
                entry.update(visible_rel=vis.norm().item() / truth.norm().item(),
                             invisible_rel=inv.norm().item() / truth.norm().item())
            images[(label, gen)] = x.flatten().cpu()
            results["images"][f"{label}|{gen}"] = entry
            print(f"{label:>10s} through {gen:>8s}: image {100*entry['image_relative_l2']:.3f}%  "
                  f"resid/noise {entry['residual_over_noise']:.3f}  noise fit {entry['noise_fit_fraction']:.3f}"
                  + (f"  invisible {100*entry['invisible_rel']:.3f}%" if proj is not None else ""), flush=True)
    # pairwise image distances (relative to truth norm)
    keys = list(images)
    results["pairwise_image_distance"] = {
        f"{a[0]}|{a[1]} vs {b[0]}|{b[1]}": (images[a] - images[b]).norm().item() / truth.norm().item()
        for i, a in enumerate(keys) for b in keys[i + 1:]}
    write_json(results, args.out)


# ---------------------------------------------------------------------------
# spectra: randomized SVD of dG/dz
# ---------------------------------------------------------------------------
def jacobian_ops(model, z: torch.Tensor, method: str, steps: int, checkpointing: bool):
    def G(zz):
        return bench.solve_flow(model, zz, steps, method, checkpointing and torch.is_grad_enabled())

    def jvp(V: torch.Tensor) -> torch.Tensor:          # V (k,1,H,W) -> J V
        Z = z.expand(V.shape[0], -1, -1, -1).contiguous()
        with torch.no_grad():
            _, out = torch.func.jvp(lambda zz: bench.solve_flow(model, zz, steps, method, False), (Z,), (V,))
        return out

    def vjp(Q: torch.Tensor) -> torch.Tensor:          # Q (k,1,H,W) -> J^T Q
        Z = z.expand(Q.shape[0], -1, -1, -1).contiguous().requires_grad_(True)
        X = G(Z)
        out, = torch.autograd.grad((X * Q).sum(), Z)
        return out.detach()

    return jvp, vjp


def randomized_svd(jvp, vjp, shape, k: int, oversample: int, power: int, device, seed: int,
                   batch: int):
    n_cols = k + oversample

    def apply(fn, M):                                  # M (d, n) -> fn columns in batches
        cols = []
        for start in range(0, M.shape[1], batch):
            block = M[:, start:start + batch].T.reshape(-1, *shape)
            cols.append(fn(block).reshape(block.shape[0], -1).T)
        return torch.cat(cols, dim=1)

    d = int(math.prod(shape))
    gen = torch.Generator(device=device).manual_seed(seed)
    omega = torch.randn(d, n_cols, device=device, generator=gen)
    Y = apply(jvp, omega)
    Q, _ = torch.linalg.qr(Y)
    for _ in range(power):
        Z = apply(vjp, Q)
        Qz, _ = torch.linalg.qr(Z)
        Y = apply(jvp, Qz)
        Q, _ = torch.linalg.qr(Y)
    B = apply(vjp, Q)                                  # (d, n) = (Q^T J)^T
    U_small, s, Vt = torch.linalg.svd(B.T, full_matrices=False)   # B^T = Q^T J = U s V^T
    return s[:k].cpu(), (Q @ U_small[:, :k]).cpu(), Vt[:k].T.cpu()   # left sing vecs in image space


def cmd_spectra(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(args.threads)
    res = args.resolution
    model = load_model(res, device, epoch=args.model_epoch, template=args.model_template)
    P = problem(res, args.sample, device)
    proj = Projector(res, device) if radon_path(res).exists() else None
    results = {"resolution": res, "sample": args.sample, "k": args.k, "oversample": args.oversample,
               "power": args.power, "entries": {}}
    out = args.out
    if out.exists():
        results = json.loads(out.read_text())
    for label, path in (item.split("=", 1) for item in args.sources):
        z = P["source0"] if path == "z0" else torch.load(path, map_location="cpu", weights_only=False)["source"].to(device)
        for gen in args.generators:
            key = f"{label}|{gen}"
            if key in results["entries"]:
                print(f"skip {key} (done)")
                continue
            method, steps = parse_generator(gen)
            jvp, vjp = jacobian_ops(model, z, method, steps, True)
            t0 = time.perf_counter()
            s, U, V = randomized_svd(jvp, vjp, tuple(z.shape[1:]), args.k, args.oversample,
                                     args.power, device, seed=0, batch=args.batch)
            seconds = time.perf_counter() - t0
            entry = {"singular_values": s.tolist(), "seconds": seconds,
                     "frobenius_estimate_topk": float((s ** 2).sum().sqrt())}
            if proj is not None:
                # how much of each left singular direction (image-space response) is
                # invisible to A: null-space fraction per direction
                inv_frac = []
                for j in range(U.shape[1]):
                    u = U[:, j].to(device)
                    vis, inv = proj.split(u)
                    inv_frac.append(inv.norm().item())
                entry["invisible_fraction_per_direction"] = inv_frac
            results["entries"][key] = entry
            torch.save({"U": U, "V": V, "s": s}, out.with_name(out.stem + f"_{label}_{gen}_vectors.pt"))
            write_json(results, out)
            print(f"{key}: top {s[:6].numpy().round(3).tolist()} ... s[{args.k-1}]={s[-1]:.4f}  "
                  f"({seconds:.0f}s)", flush=True)


# ---------------------------------------------------------------------------
# stepmaps: how much does ONE flow step contract random perturbations, per
# integrator, at each time along the flow (Euler-100 step vs RK4-100 step vs
# the "exact" step = RK4 with 8 substeps over the same interval)?
# ---------------------------------------------------------------------------
def one_step(model, x, t, h, method, substeps=1):
    hh = h / substeps
    for j in range(substeps):
        tt = t + j * hh
        if method == "forward_euler":
            x = x + hh * bench.velocity(model, tt, x)
        else:
            k1 = bench.velocity(model, tt, x)
            k2 = bench.velocity(model, tt + 0.5 * hh, x + 0.5 * hh * k1)
            k3 = bench.velocity(model, tt + 0.5 * hh, x + 0.5 * hh * k2)
            k4 = bench.velocity(model, tt + hh, x + hh * k3)
            x = x + (hh / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def cmd_stepmaps(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    res = args.resolution
    model = load_model(res, device, epoch=args.model_epoch, template=args.model_template)
    P = problem(res, args.sample, device)
    z = P["source0"] if args.source == "z0" else torch.load(args.source, map_location="cpu", weights_only=False)["source"].to(device)
    gen = torch.Generator(device=device).manual_seed(0)
    probes = torch.randn(args.probes, *z.shape[1:], device=device, generator=gen)
    probes = probes / probes.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
    results = {"resolution": res, "sample": args.sample, "source": args.source, "rows": []}
    # states along the reference trajectory (RK4-100 primal; the two 100-step
    # trajectories differ by ~3% which does not matter for a one-step gain)
    h = 1.0 / args.steps
    x = z.clone()
    with torch.no_grad():
        for i in range(args.steps):
            t = i * h
            if i in args.at or i == args.steps - 1:
                row = {"step": i, "t": t}
                X = x.expand(args.probes, -1, -1, -1).contiguous()
                for label, method, sub in (("euler", "forward_euler", 1), ("rk4", "rk4", 1),
                                           ("rk4_sub8", "rk4", 8), ("euler_sub4", "forward_euler", 4)):
                    _, out = torch.func.jvp(lambda X_: one_step(model, X_, t, h, method, sub), (X,), (probes,))
                    gains = out.flatten(1).norm(dim=1)
                    # Jacobian of the velocity itself: hutchinson trace / frobenius of h*J_v
                    row[label] = {"mean_gain": gains.mean().item(), "min_gain": gains.min().item(),
                                  "max_gain": gains.max().item(), "rms_gain": gains.square().mean().sqrt().item()}
                _, jv = torch.func.jvp(lambda X_: bench.velocity(model, t, X_), (X,), (probes,))
                row["h_Jv_rms"] = (h * jv.flatten(1).norm(dim=1)).square().mean().sqrt().item()
                row["h_trace_Jv_over_d"] = (h * (jv.flatten(1) * probes.flatten(1)).sum(1)).mean().item()
                results["rows"].append(row)
                print(json.dumps({k: (v if not isinstance(v, dict) else round(v["rms_gain"], 5)) for k, v in row.items()}), flush=True)
            x = one_step(model, x, t, h, "rk4", 1)
    write_json(results, args.out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("solve")
    p.add_argument("--resolution", type=int, default=250)
    p.add_argument("--model-epoch", type=int, default=100, help="checkpoint epoch to load")
    p.add_argument("--model-template", type=str, default=None, help="override the checkpoint run-dir template")
    p.add_argument("--sample", type=int, required=True)
    p.add_argument("--generator", required=True, help="euler100, euler400, rk425, rk4100, ...")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--cap", type=float, default=0.01, help="step cap as a multiple of sqrt(d)")
    p.add_argument("--max-steps", type=int, default=160)
    p.add_argument("--max-minutes", type=float, default=100000)
    p.add_argument("--stop-ratio", type=float, default=0.0, help="stop once residual/noise <= this")
    p.add_argument("--checkpointing", action="store_true", default=True)
    p.add_argument("--no-checkpointing", dest="checkpointing", action="store_false")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--init-source", default=None, help="warm start from a saved state's source")
    p.add_argument("--no-tf32", action="store_true", help="disable TF32 convolutions/matmuls on CUDA")
    p.add_argument("--gradient", choices=("exact", "dlr"), default="exact")
    p.add_argument("--dlr-rank", type=int, default=8)
    p.add_argument("--dlr-trace-probes", type=int, default=0, help="0 = analytic alpha_t = 1 - t")
    p.add_argument("--target", choices=("data", "image"), default="data",
                   help="'image' fits the ground-truth image directly (oracle representational error)")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_solve)

    p = sub.add_parser("radon")
    p.add_argument("--resolution", type=int, default=250)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch", type=int, default=250)
    p.set_defaults(func=cmd_radon)

    p = sub.add_parser("decompose")
    p.add_argument("--resolution", type=int, default=250)
    p.add_argument("--sample", type=int, required=True)
    p.add_argument("--states", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_decompose)

    p = sub.add_parser("cross")
    p.add_argument("--resolution", type=int, default=250)
    p.add_argument("--model-epoch", type=int, default=100, help="checkpoint epoch to load")
    p.add_argument("--model-template", type=str, default=None, help="override the checkpoint run-dir template")
    p.add_argument("--sample", type=int, required=True)
    p.add_argument("--sources", nargs="+", required=True, help="label=path/to/state.pt")
    p.add_argument("--generators", nargs="+", default=["euler100", "rk4100", "rk4800"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_cross)

    p = sub.add_parser("spectra")
    p.add_argument("--resolution", type=int, default=250)
    p.add_argument("--model-epoch", type=int, default=100, help="checkpoint epoch to load")
    p.add_argument("--model-template", type=str, default=None, help="override the checkpoint run-dir template")
    p.add_argument("--sample", type=int, required=True)
    p.add_argument("--sources", nargs="+", required=True, help="label=path/to/state.pt or label=z0")
    p.add_argument("--generators", nargs="+", default=["euler100", "rk425", "rk4100", "euler400"])
    p.add_argument("--k", type=int, default=24)
    p.add_argument("--oversample", type=int, default=8)
    p.add_argument("--power", type=int, default=2)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_spectra)

    p = sub.add_parser("stepmaps")
    p.add_argument("--resolution", type=int, default=250)
    p.add_argument("--model-epoch", type=int, default=100, help="checkpoint epoch to load")
    p.add_argument("--model-template", type=str, default=None, help="override the checkpoint run-dir template")
    p.add_argument("--sample", type=int, default=1)
    p.add_argument("--source", default="z0")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--at", type=int, nargs="+", default=[0, 10, 25, 50, 75, 85, 90, 93, 95, 96, 97, 98, 99])
    p.add_argument("--probes", type=int, default=16)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_stepmaps)

    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
