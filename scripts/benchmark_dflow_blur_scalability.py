"""Benchmark D-Flow Gaussian deblurring runtime across circle resolutions.

This reproduces the coarse LBFGS setup from
notebooks/coarse_fine_time_comparison/D_Flow_Blur_LBFGS_Coarse.ipynb:
Gaussian blur sigma=5, 5 Euler steps through the flow, and 10 outer LBFGS
steps with max_iter=20.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cmocean
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.unet.default import Unet


@dataclass
class SampleResult:
    resolution: int
    n_circles: int
    sample_idx: int
    elapsed_s: float
    best_loss: float
    final_loss: float
    best_objective: float
    final_objective: float
    best_tv: float
    final_tv: float
    best_step: int
    final_grad_norm: float
    measurement_mse_raw: float
    image_mse_raw: float
    image_mse_clamped: float
    steps_run: int
    figure_path: str


@dataclass
class IterationTiming:
    resolution: int
    n_circles: int
    sample_idx: int
    outer_step: int
    elapsed_s: float
    closure_evals: int
    elapsed_per_closure_s: float
    data_mse: float
    objective: float
    tv: float
    grad_norm: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolutions", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--n-circles", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument(
        "--sample-indices",
        type=int,
        nargs="+",
        default=None,
        help="Explicit test-set sample indices. Overrides --num-samples when provided.",
    )
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--blur-sigma", type=float, default=5.0)
    parser.add_argument("--euler-steps", type=int, default=5)
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--lbfgs-max-iter", type=int, default=20)
    parser.add_argument("--lbfgs-history-size", type=int, default=25)
    parser.add_argument(
        "--line-search-fn",
        choices=["strong_wolfe", "none"],
        default="strong_wolfe",
        help="LBFGS line search (use 'none' to disable it).",
    )
    parser.add_argument("--outer-steps", type=int, default=10)
    parser.add_argument("--stop-loss", type=float, default=1e-8)
    parser.add_argument(
        "--tv-weight",
        type=float,
        default=0.0,
        help="Weight for total-variation regularization on the D-Flow reconstruction.",
    )
    parser.add_argument(
        "--tv-eps",
        type=float,
        default=1e-6,
        help="Charbonnier epsilon for differentiable total variation.",
    )
    parser.add_argument("--base-seed", type=int, default=123)
    parser.add_argument(
        "--record-iteration-timings",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Synchronize CUDA around each outer optimizer step and save per-step timings.",
    )
    parser.add_argument(
        "--timing-warmup-steps",
        type=int,
        default=10,
        help="Initial outer steps excluded from steady-state timing summaries.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/dflow_scalability"))
    parser.add_argument("--save-figures", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_run_config(resolution: int, n_circles: int) -> dict:
    path = Path(f"outputs/fm_unet/circles-{resolution}/n{n_circles}/config.json")
    if not path.exists():
        raise FileNotFoundError(f"Missing run config: {path}")
    with path.open() as f:
        return json.load(f)


def load_model(resolution: int, n_circles: int, epoch: int, device: torch.device) -> torch.nn.Module:
    config = load_run_config(resolution, n_circles)
    arch = config["arch"]
    ckpt = Path(f"outputs/fm_unet/circles-{resolution}/n{n_circles}/checkpoints/epoch_{epoch:04d}.pt")
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

    model = Unet(
        ch=arch["ch"],
        ch_mul=arch["ch_mul"],
        att_channels=arch["att_channels"],
        groups=arch["groups"],
        dropout=0.0,
    ).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    state = {
        k.removeprefix("module._orig_mod.").removeprefix("module."): v
        for k, v in state.items()
        if k != "n_averaged"
    }
    model.load_state_dict(state)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def euler_integrate(model: torch.nn.Module, x0: torch.Tensor, n_steps: int) -> torch.Tensor:
    dt = 1.0 / n_steps
    x = x0
    for i in range(n_steps):
        t = torch.full((x.shape[0],), i * dt, device=x.device)
        x = x + dt * model(x, t)
    return x


def make_blur(sigma: float, device: torch.device):
    ks = int(6 * sigma + 1)
    ks = ks if ks % 2 == 1 else ks + 1
    half = ks // 2
    coords = torch.arange(ks, dtype=torch.float32, device=device) - half
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()

    conv_x = torch.nn.Conv2d(1, 1, kernel_size=(1, ks), padding=(0, half), bias=False).to(device)
    conv_y = torch.nn.Conv2d(1, 1, kernel_size=(ks, 1), padding=(half, 0), bias=False).to(device)
    with torch.no_grad():
        conv_x.weight.copy_(g.view(1, 1, 1, ks))
        conv_y.weight.copy_(g.view(1, 1, ks, 1))
    for param in (*conv_x.parameters(), *conv_y.parameters()):
        param.requires_grad_(False)

    def A(x: torch.Tensor) -> torch.Tensor:
        return conv_y(conv_x(x))

    return A


def total_variation(x: torch.Tensor, eps: float) -> torch.Tensor:
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    tv_x = torch.sqrt(dx.square() + eps * eps).mean()
    tv_y = torch.sqrt(dy.square() + eps * eps).mean()
    return tv_x + tv_y


def tensor_image(x: torch.Tensor):
    return x.detach().squeeze().float().cpu().numpy()


def save_diagnostic_figure(
    *,
    path: Path,
    x_gt: torch.Tensor,
    y_obs: torch.Tensor,
    x_hat: torch.Tensor,
    x_raw: torch.Tensor,
    Ax_raw: torch.Tensor,
    n_circles: int,
    blur_sigma: float,
    euler_steps: int,
    image_mse_clamped: float,
    measurement_mse_raw: float,
) -> None:
    err = x_hat - x_gt
    meas_resid = Ax_raw - y_obs

    fig = plt.figure(figsize=(10.6, 6.0), dpi=180, facecolor="white")
    gs = fig.add_gridspec(2, 4, hspace=0.42, wspace=0.08)
    axes_top = [fig.add_subplot(gs[0, i]) for i in range(4)]
    axes_bottom = [fig.add_subplot(gs[1, i]) for i in range(1, 4)]

    dense = cmocean.cm.dense_r
    balance = cmocean.cm.balance
    image_panels = [
        (axes_top[0], x_gt, dense, 0.0, 1.0, r"Sharp $x_{\mathrm{gt}}$"),
        (axes_top[1], y_obs, dense, 0.0, 1.0, r"Blurred $y_{\mathrm{obs}}$"),
        (axes_top[2], x_hat, dense, 0.0, 1.0, r"D-Flow $\hat{x}$"),
        (axes_top[3], err, balance, -1.0, 1.0, r"Error $\hat{x} - x_{\mathrm{gt}}$"),
        (axes_bottom[0], y_obs, dense, 0.0, 1.0, r"$y_{\mathrm{obs}} = A x_{\mathrm{gt}}$"),
        (axes_bottom[1], Ax_raw, dense, 0.0, 1.0, r"$A\hat{x}_{\mathrm{raw}}$"),
        (axes_bottom[2], meas_resid, balance, -1.0, 1.0, r"$A\hat{x}_{\mathrm{raw}} - y_{\mathrm{obs}}$"),
    ]

    ims = {}
    for ax, img, cmap, vmin, vmax, title in image_panels:
        im = ax.imshow(tensor_image(img), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=8, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ims[ax] = im

    axes_top[2].set_xlabel(f"MSE = {image_mse_clamped:.3e}", fontsize=7, labelpad=4)
    axes_bottom[1].set_xlabel(
        rf"MSE$(A\hat{{x}}, y_{{\mathrm{{obs}}}})$ = {measurement_mse_raw:.3e}",
        fontsize=7,
        labelpad=4,
    )

    for ax in (axes_top[3], axes_bottom[2]):
        cbar = fig.colorbar(
            ims[ax],
            ax=ax,
            orientation="horizontal",
            fraction=0.08,
            pad=0.05,
            aspect=16,
            shrink=0.9,
        )
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=5, length=2.0, width=0.3, pad=1)

    fig.suptitle(
        rf"D-Flow deblurring (LBFGS) n = {n_circles}, $\sigma$ = {blur_sigma}, "
        rf"{euler_steps} Euler steps",
        fontsize=9,
        y=0.98,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_one_sample(
    *,
    model: torch.nn.Module,
    A,
    x_gt: torch.Tensor,
    resolution: int,
    n_circles: int,
    sample_idx: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[SampleResult, list[IterationTiming]]:
    with torch.no_grad():
        y_obs = A(x_gt)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.base_seed + resolution * 1000 + n_circles * 100 + sample_idx)
    x0 = torch.randn(
        1, 1, resolution, resolution, device=device, generator=generator, requires_grad=True
    )
    optimizer = torch.optim.LBFGS(
        [x0],
        lr=args.lbfgs_lr,
        max_iter=args.lbfgs_max_iter,
        history_size=args.lbfgs_history_size,
        line_search_fn=None if args.line_search_fn == "none" else args.line_search_fn,
        tolerance_grad=1e-12,
        tolerance_change=1e-16,
    )

    loss_scale = float(resolution * resolution)
    best_loss = float("inf")
    best_objective = float("inf")
    best_tv = float("inf")
    best_x0 = x0.detach().clone()
    best_step = -1
    last_grad_norm = 0.0
    last_loss = float("inf")
    last_objective = float("inf")
    last_tv = 0.0
    current_step = 0
    steps_run = 0
    closure_calls = 0
    iteration_timings: list[IterationTiming] = []

    def closure():
        nonlocal closure_calls
        nonlocal best_loss, best_objective, best_tv, best_x0, best_step
        nonlocal last_grad_norm, last_loss, last_objective, last_tv
        closure_calls += 1
        optimizer.zero_grad()
        x = euler_integrate(model, x0, n_steps=args.euler_steps)
        Ax = A(x)
        diff = Ax - y_obs
        data_mse = diff.pow(2).mean()
        tv = total_variation(x, args.tv_eps)
        objective = data_mse + args.tv_weight * tv
        objective_scaled = objective * loss_scale
        objective_scaled.backward()
        last_grad_norm = x0.grad.norm().item() if x0.grad is not None else 0.0
        last_loss = data_mse.item()
        last_tv = tv.item()
        last_objective = objective.item()
        if last_objective < best_objective:
            best_loss = last_loss
            best_objective = last_objective
            best_tv = last_tv
            best_x0 = x0.detach().clone()
            best_step = current_step
        return objective_scaled

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for step in range(args.outer_steps):
        current_step = step
        closure_calls_before_step = closure_calls
        if args.record_iteration_timings and device.type == "cuda":
            torch.cuda.synchronize(device)
        step_start = time.perf_counter()
        optimizer.step(closure)
        if args.record_iteration_timings and device.type == "cuda":
            torch.cuda.synchronize(device)
        step_elapsed_s = time.perf_counter() - step_start
        steps_run = step + 1
        if args.record_iteration_timings:
            closure_evals = closure_calls - closure_calls_before_step
            iteration_timings.append(
                IterationTiming(
                    resolution=resolution,
                    n_circles=n_circles,
                    sample_idx=sample_idx,
                    outer_step=step,
                    elapsed_s=step_elapsed_s,
                    closure_evals=closure_evals,
                    elapsed_per_closure_s=(
                        step_elapsed_s / closure_evals if closure_evals else float("nan")
                    ),
                    data_mse=last_loss,
                    objective=last_objective,
                    tv=last_tv,
                    grad_norm=last_grad_norm,
                )
            )
        if last_loss < args.stop_loss:
            break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_s = time.perf_counter() - start

    with torch.no_grad():
        x_raw = euler_integrate(model, best_x0, n_steps=args.euler_steps)
        x_hat = x_raw.clamp(0, 1)
        Ax_raw = A(x_raw)
        measurement_mse_raw = (Ax_raw - y_obs).pow(2).mean().item()
        image_mse_raw = (x_raw - x_gt).pow(2).mean().item()
        image_mse_clamped = (x_hat - x_gt).pow(2).mean().item()

    figure_path = ""
    if args.save_figures:
        fig_path = (
            args.out_dir
            / "figures"
            / f"res_{resolution}"
            / f"n{n_circles}_sample_{sample_idx:03d}.png"
        )
        save_diagnostic_figure(
            path=fig_path,
            x_gt=x_gt,
            y_obs=y_obs,
            x_hat=x_hat,
            x_raw=x_raw,
            Ax_raw=Ax_raw,
            n_circles=n_circles,
            blur_sigma=args.blur_sigma,
            euler_steps=args.euler_steps,
            image_mse_clamped=image_mse_clamped,
            measurement_mse_raw=measurement_mse_raw,
        )
        figure_path = str(fig_path)

    result = SampleResult(
        resolution=resolution,
        n_circles=n_circles,
        sample_idx=sample_idx,
        elapsed_s=elapsed_s,
        best_loss=best_loss,
        final_loss=last_loss,
        best_objective=best_objective,
        final_objective=last_objective,
        best_tv=best_tv,
        final_tv=last_tv,
        best_step=best_step,
        final_grad_norm=last_grad_norm,
        measurement_mse_raw=measurement_mse_raw,
        image_mse_raw=image_mse_raw,
        image_mse_clamped=image_mse_clamped,
        steps_run=steps_run,
        figure_path=figure_path,
    )
    return result, iteration_timings


def summarize(rows: list[SampleResult]) -> list[dict]:
    summary = []
    for resolution in sorted({row.resolution for row in rows}):
        subset = [row for row in rows if row.resolution == resolution]
        times = [row.elapsed_s for row in subset]
        losses = [row.best_loss for row in subset]
        objectives = [row.best_objective for row in subset]
        tvs = [row.best_tv for row in subset]
        image_mses = [row.image_mse_clamped for row in subset]
        summary.append(
            {
                "resolution": resolution,
                "n_samples": len(subset),
                "total_elapsed_s": sum(times),
                "mean_elapsed_s": statistics.mean(times),
                "median_elapsed_s": statistics.median(times),
                "std_elapsed_s": statistics.stdev(times) if len(times) > 1 else 0.0,
                "min_elapsed_s": min(times),
                "max_elapsed_s": max(times),
                "mean_best_loss": statistics.mean(losses),
                "median_best_loss": statistics.median(losses),
                "mean_best_objective": statistics.mean(objectives),
                "median_best_objective": statistics.median(objectives),
                "mean_best_tv": statistics.mean(tvs),
                "median_best_tv": statistics.median(tvs),
                "mean_image_mse_clamped": statistics.mean(image_mses),
                "median_image_mse_clamped": statistics.median(image_mses),
            }
        )
    return summary


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_iteration_timings(
    rows: list[IterationTiming], warmup_steps: int
) -> list[dict]:
    summary = []
    for resolution in sorted({row.resolution for row in rows}):
        subset = [row for row in rows if row.resolution == resolution]
        steady = [row for row in subset if row.outer_step >= warmup_steps]
        if not steady:
            steady = subset
        times = [row.elapsed_s for row in steady]
        closure_evals = [row.closure_evals for row in steady]
        per_closure = [row.elapsed_per_closure_s for row in steady]
        summary.append(
            {
                "resolution": resolution,
                "n_samples": len({row.sample_idx for row in subset}),
                "n_iterations": len(subset),
                "n_steady_state_iterations": len(steady),
                "warmup_steps_excluded_per_sample": warmup_steps,
                "mean_iteration_s": statistics.mean(times),
                "median_iteration_s": statistics.median(times),
                "std_iteration_s": statistics.stdev(times) if len(times) > 1 else 0.0,
                "p10_iteration_s": percentile(times, 0.10),
                "p90_iteration_s": percentile(times, 0.90),
                "mean_closure_evals_per_iteration": statistics.mean(closure_evals),
                "median_closure_evals_per_iteration": statistics.median(closure_evals),
                "mean_elapsed_per_closure_s": statistics.mean(per_closure),
                "median_elapsed_per_closure_s": statistics.median(per_closure),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        torch.cuda.set_device(device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[SampleResult] = []
    all_iteration_timings: list[IterationTiming] = []
    run_start = time.perf_counter()

    print(f"Device: {device}", flush=True)
    print(
        "D-Flow blur benchmark: "
        f"resolutions={args.resolutions}, n={args.n_circles}, samples={args.num_samples}, "
        f"sigma={args.blur_sigma}, euler_steps={args.euler_steps}, "
        f"outer_steps={args.outer_steps}, max_iter={args.lbfgs_max_iter}, "
        f"line_search={args.line_search_fn}, "
        f"tv_weight={args.tv_weight}, tv_eps={args.tv_eps}, "
        f"save_figures={args.save_figures}",
        flush=True,
    )

    for resolution in args.resolutions:
        print(f"\n== {resolution}x{resolution} ==", flush=True)
        model = load_model(resolution, args.n_circles, args.epoch, device)
        A = make_blur(args.blur_sigma, device)
        dataset_path = Path(f"data/circles-{resolution}-n{args.n_circles}.pt")
        test_set = torch.load(dataset_path, map_location="cpu", weights_only=True)["test"]
        sample_indices = args.sample_indices or list(range(args.num_samples))
        invalid_indices = [index for index in sample_indices if index < 0 or index >= len(test_set)]
        if invalid_indices:
            raise ValueError(
                f"{dataset_path} has {len(test_set)} test samples; invalid indices: {invalid_indices}"
            )

        for sample_idx in sample_indices:
            x_gt = test_set[sample_idx].unsqueeze(0).to(device)
            result, iteration_timings = run_one_sample(
                model=model,
                A=A,
                x_gt=x_gt,
                resolution=resolution,
                n_circles=args.n_circles,
                sample_idx=sample_idx,
                args=args,
                device=device,
            )
            all_rows.append(result)
            all_iteration_timings.extend(iteration_timings)
            print(
                f"  sample {sample_idx:02d}: {result.elapsed_s:8.3f}s "
                f"best={result.best_loss:.3e} obj={result.best_objective:.3e} "
                f"tv={result.best_tv:.3e} meas={result.measurement_mse_raw:.3e} "
                f"fig={result.figure_path}"
                if result.figure_path
                else f"  sample {sample_idx:02d}: {result.elapsed_s:8.3f}s "
                f"best={result.best_loss:.3e} obj={result.best_objective:.3e} "
                f"tv={result.best_tv:.3e} meas={result.measurement_mse_raw:.3e}",
                flush=True,
            )

        del model, A
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_rows = summarize(all_rows)
    iteration_summary_rows = summarize_iteration_timings(
        all_iteration_timings, args.timing_warmup_steps
    )
    elapsed_total = time.perf_counter() - run_start
    output = {
        "config": vars(args)
        | {
            "out_dir": str(args.out_dir),
            "total_wall_s": elapsed_total,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
        },
        "samples": [asdict(row) for row in all_rows],
        "summary": summary_rows,
        "iteration_timing_summary": iteration_summary_rows,
    }
    json_path = args.out_dir / "dflow_blur_scalability_stats.json"
    with json_path.open("w") as f:
        json.dump(output, f, indent=2)
    write_csv(args.out_dir / "dflow_blur_scalability_samples.csv", [asdict(row) for row in all_rows])
    write_csv(args.out_dir / "dflow_blur_scalability_summary.csv", summary_rows)
    if all_iteration_timings:
        write_csv(
            args.out_dir / "dflow_iteration_timings.csv",
            [asdict(row) for row in all_iteration_timings],
        )
        write_csv(
            args.out_dir / "dflow_iteration_timing_summary.csv",
            iteration_summary_rows,
        )

    print("\nSummary:", flush=True)
    for row in summary_rows:
        print(
            f"  {row['resolution']}x{row['resolution']}: "
            f"mean={row['mean_elapsed_s']:.3f}s median={row['median_elapsed_s']:.3f}s "
            f"std={row['std_elapsed_s']:.3f}s total={row['total_elapsed_s']:.3f}s",
            flush=True,
        )
    if iteration_summary_rows:
        print("\nPer-iteration timing (steady state):", flush=True)
        for row in iteration_summary_rows:
            print(
                f"  {row['resolution']}x{row['resolution']}: "
                f"mean={row['mean_iteration_s']:.6f}s "
                f"median={row['median_iteration_s']:.6f}s "
                f"p10={row['p10_iteration_s']:.6f}s "
                f"p90={row['p90_iteration_s']:.6f}s "
                f"closures/iteration={row['mean_closure_evals_per_iteration']:.2f}",
                flush=True,
            )
    print(f"Total wall time: {elapsed_total:.3f}s", flush=True)
    print(f"Wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
