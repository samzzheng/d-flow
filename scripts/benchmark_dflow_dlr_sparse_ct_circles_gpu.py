"""D-Flow vs DLR-D-Flow sparse-CT comparison on circles, GPU.

Fork of benchmark_dflow_sparse_ct_circles_cpu.py that adds a "dlr" method
alongside "forward_euler" (and optionally "rk4"), reusing the exact same
optimizer protocol: fixed outer-step budget (no early stop on the noise
target), LBFGS max_iter>1, "relative_scaled" objective, and the
nonfinite-only rollback policy. This intentionally lets a method keep
optimizing past the noise floor, which is what exposes the sinogram
null-space overfitting failure mode on D-Flow.

The benchmark adds Gaussian measurement noise whose L2 norm is exactly 1% of
the clean sinogram norm. Relative L2 error is used for convergence and primary
reporting so results are comparable across image and measurement dimensions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import cmocean
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.unet.default import Unet
from sampler.optimizer_state import (
    LBFGS_HISTORY_RESET_POLICY, LBFGS_INITIAL_STEP_POLICY, reset_optimizer_history,
)
import dlr_dflow.dlr_dflow as dd
import dlr_dflow.tangent_unet as tangent
from dlr_dflow.graphed_products import GraphedAProducts
from dlr_dflow import parallel_bug as PB
from dlr_dflow import enriched


METHODS = ("forward_euler", "rk4", "dlr")


class ClosureBudgetReached(RuntimeError):
    """Raised before an LBFGS closure would exceed the exact run budget."""


@dataclass
class SampleResult:
    resolution: int
    sample_index: int
    method: str
    flow_steps: int
    velocity_evaluations_per_flow: int
    noise_relative_l2: float
    steps_run: int
    closure_evaluations: int
    closure_budget: int | None
    closure_budget_reached: bool
    best_step: int
    best_closure_evaluation: int
    image_relative_l2: float
    measurement_relative_l2_noisy: float
    measurement_relative_l2_clean: float
    elapsed_s: float
    mean_outer_time_s: float
    lbfgs_rollbacks: int
    device: str
    device_name: str
    peak_gpu_memory_mb: float
    checkpoint_sha256: str
    diagnostic_figure: str


@dataclass
class IterationResult:
    resolution: int
    sample_index: int
    method: str
    outer_step: int
    measurement_relative_l2: float
    best_measurement_relative_l2: float
    elapsed_s: float
    closure_evaluations: int
    function_evaluations: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolutions", type=int, nargs="+", default=[250])
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--angles", type=int, default=15)
    parser.add_argument("--noise-level", type=float, default=0.01)
    parser.add_argument("--euler-flow-steps", type=int, default=100)
    parser.add_argument("--rk4-flow-steps", type=int, default=100)
    parser.add_argument("--outer-steps", type=int, default=100)
    parser.add_argument("--rank", type=int, default=8, help="DLR rank")
    parser.add_argument("--dlr-speed-profile", choices=("reference", "fast"),
                        help="reference: 8 trace/4 enrichment; experimental fast: 4 trace/0 enrichment (changes approximation); explicit counts override")
    parser.add_argument("--trace-probes", type=int, default=None)
    parser.add_argument("--init-oversampling", type=int, default=8)
    parser.add_argument("--enrichment-probes", type=int, default=None)
    parser.add_argument("--factor-stride", type=int, default=1)
    parser.add_argument(
        "--dlr-integrator", choices=("ksl", "parallel", "bug"), default="ksl",
        help=("ksl: enriched projector splitting (algorithm of record); parallel: "
              "fixed-rank parallel BUG (Ceruti-Kusch-Lubich), S11 dropped; bug: same "
              "with the S11 block kept (= augmented BUG)"))
    parser.add_argument("--dlr-overlap", action="store_true",
                        help="parallel/bug: issue the JVP and VJP calls on separate CUDA streams")
    parser.add_argument("--use-cuda-graphs", action="store_true", default=True)
    parser.add_argument("--no-cuda-graphs", dest="use_cuda_graphs",
                         action="store_false")
    parser.add_argument("--dlr-max-jvp-columns", type=int, default=None,
                        help="Optional column limit for ordinary CUDA JVP graphs; complete-step graphs are unchanged")
    parser.add_argument("--dlr-max-vjp-columns", type=int, default=None,
                        help="Optional column limit for ordinary CUDA VJP graphs; complete-step graphs are unchanged")
    parser.add_argument("--dlr-step-jvp-columns", type=int, default=None,
                        help="Optional JVP column limit inside complete-step CUDA graphs; reuses one shared primal tape")
    parser.add_argument("--dlr-step-vjp-columns", type=int, default=None,
                        help="Optional VJP column limit inside complete-step CUDA graphs; reuses one shared primal tape")
    parser.add_argument(
        "--closure-budget", type=int, default=None,
        help=(
            "Exact maximum number of objective/gradient evaluations per method. "
            "When set, LBFGS advances one accepted iteration per outer call."
        ),
    )
    parser.add_argument("--optimizer", choices=("lbfgs", "adam"), default="lbfgs",
                        help="adam: one closure per outer step, --adam-lr step size")
    parser.add_argument("--adam-lr", type=float, default=0.05)
    parser.add_argument(
        "--source-prior", choices=("none", "l2", "shell"), default="none",
        help=("l2: + lam/2 ||z||^2 (pulls toward z=0); shell: + lam/2 "
              "(||z|| - sqrt(d))^2 (pulls toward the Gaussian typical set)"),
    )
    parser.add_argument("--source-prior-lambda", type=float, default=0.0)
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--lbfgs-max-iter", type=int, default=5)
    parser.add_argument(
        "--lbfgs-max-eval", type=int, default=None,
        help=(
            "Maximum closure evaluations available to one LBFGS step. "
            "Set this explicitly when --closure-budget forces max_iter=1."
        ),
    )
    parser.add_argument("--history-size", type=int, default=100)
    parser.add_argument(
        "--lbfgs-line-search", choices=("none", "strong_wolfe", "armijo"),
        default="none",
        help=("armijo: hand-written L-BFGS whose trial points are checked with a "
              "forward-only evaluation of the true objective; the gradient is "
              "computed only at the accepted point"),
    )
    parser.add_argument("--armijo-max-ls", type=int, default=30)
    parser.add_argument("--armijo-c1", type=float, default=1e-4)
    parser.add_argument(
        "--armijo-step-cap", type=float, default=0.0,
        help=("cap each proposed step at this fraction of sqrt(d) (the Gaussian "
              "shell radius), before backtracking; 0 = no cap"))
    parser.add_argument("--lbfgs-tolerance-grad", type=float, default=1e-10)
    parser.add_argument("--lbfgs-tolerance-change", type=float, default=1e-14)
    parser.add_argument(
        "--objective", choices=("relative_scaled", "measurement_mse"),
        default="relative_scaled",
        help=(
            "relative_scaled preserves the historical benchmark objective; "
            "measurement_mse matches the repository D-Flow data cost"
        ),
    )
    parser.add_argument(
        "--solution-state", choices=("best", "final"), default="best",
        help="Report either the lowest-cost closure or the final optimizer state.",
    )
    parser.add_argument("--rollback-factor", type=float, default=1.5)
    parser.add_argument(
        "--rollback-policy", choices=("factor", "nonfinite", "none"), default="nonfinite",
        help=(
            "factor restores the best latent and clears LBFGS history after a "
            "large loss increase; nonfinite preserves history unless the state "
            "or objective becomes nonfinite"
        ),
    )
    parser.add_argument("--base-seed", type=int, default=20260807)
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=False,
        help="Require deterministic PyTorch kernels and deterministic SDP attention.",
    )
    parser.add_argument(
        "--activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Checkpoint exact D-Flow velocity-model calls (defaults on for CUDA).",
    )
    parser.add_argument(
        "--dlr-activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Recompute local UNet block tapes in DLR Jacobian products; independent of exact D-Flow checkpointing.",
    )
    parser.add_argument(
        "--dlr-fused-norm", action=argparse.BooleanOptionalAction, default=False,
        help="Use the experimental fused CUDA normalization derivative kernel for DLR products.",
    )
    parser.add_argument(
        "--dlr-fused-primal", choices=("native", "affine", "full"), default="native",
        help="Optional CUDA float32 fusion of shared-tape GN affine/conditioning and SiLU; retains the existing manual GN statistics.",
    )
    parser.add_argument(
        "--dlr-specialized-algebra", action=argparse.BooleanOptionalAction, default=False,
        help="Use equivalent QR-factor recompression algebra for fixed-rank zero-enrichment DLR.",
    )
    parser.add_argument(
        "--channels-last", action=argparse.BooleanOptionalAction, default=False,
        help="Use channels-last model weights for every selected method.",
    )
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=["forward_euler", "dlr"],
        help="Integrators/gradient estimators to run.",
    )
    parser.add_argument(
        "--run-dir-template", type=str,
        default="outputs/fm_unet/circles-{res}-noattn-ch16-b8/n1",
        help="Training run directory holding config.json and checkpoints/.",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("outputs/dflow_dlr_sparse_ct_circles_gpu_matched"),
    )
    args = parser.parse_args()
    for name in ("dlr_max_jvp_columns", "dlr_max_vjp_columns",
                 "dlr_step_jvp_columns", "dlr_step_vjp_columns"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    probes, enrichment = {None: (32, 4), "reference": (8, 4), "fast": (4, 0)}[args.dlr_speed_profile]
    if args.trace_probes is None:
        args.trace_probes = probes
    if args.enrichment_probes is None:
        args.enrichment_probes = enrichment
    return args


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Union of keys in first-seen order: rows written by an older version of
    # this script (fewer columns) can be resumed without a DictWriter error.
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def run_dir(template: str, resolution: int) -> Path:
    return ROOT / template.format(res=resolution)


def load_model(
    resolution: int, epoch: int, device: torch.device, template: str,
) -> torch.nn.Module:
    run = run_dir(template, resolution)
    config = json.loads((run / "config.json").read_text())
    arch = config["arch"]
    model = Unet(
        ch=arch["ch"], ch_mul=arch["ch_mul"],
        att_channels=arch["att_channels"], groups=arch["groups"], dropout=0.0,
        mid_attn=arch.get("mid_attn", True),
    )
    state = torch.load(
        run / f"checkpoints/epoch_{epoch:04d}.pt", map_location="cpu", weights_only=True)
    state = {
        key.removeprefix("module._orig_mod.").removeprefix("module."): value
        for key, value in state.items() if key != "n_averaged"
    }
    model.load_state_dict(state)
    model.to(device).eval().requires_grad_(False)
    return model


def sparse_radon(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = x.shape
    count = angles.numel()
    repeated = x[:, None].expand(batch, count, channels, height, width)
    repeated = repeated.reshape(batch * count, channels, height, width)
    angle_batch = angles.repeat(batch)
    cosine, sine = torch.cos(angle_batch), torch.sin(angle_batch)
    theta = torch.zeros(batch * count, 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0], theta[:, 0, 1] = cosine, -sine
    theta[:, 1, 0], theta[:, 1, 1] = sine, cosine
    grid = F.affine_grid(theta, repeated.shape, align_corners=False)
    rotated = F.grid_sample(
        repeated, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    projections = rotated.mean(dim=-2)
    return projections.reshape(batch, count, channels, width).permute(0, 2, 1, 3)


def velocity(
    model: torch.nn.Module,
    t: float,
    x: torch.Tensor,
    activation_checkpointing: bool = False,
) -> torch.Tensor:
    time_batch = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
    if activation_checkpointing and torch.is_grad_enabled() and x.requires_grad:
        return checkpoint(model, x, time_batch, use_reentrant=False)
    return model(x, time_batch)


def solve_flow(
    model: torch.nn.Module, source: torch.Tensor, steps: int, method: str,
    activation_checkpointing: bool = False,
) -> torch.Tensor:
    h = 1.0 / steps
    x = source
    for index in range(steps):
        t = index * h
        if method == "forward_euler":
            x = x + h * velocity(model, t, x, activation_checkpointing)
        elif method == "rk4":
            k1 = velocity(model, t, x, activation_checkpointing)
            k2 = velocity(
                model, t + 0.5 * h, x + 0.5 * h * k1,
                activation_checkpointing,
            )
            k3 = velocity(
                model, t + 0.5 * h, x + 0.5 * h * k2,
                activation_checkpointing,
            )
            k4 = velocity(
                model, t + h, x + h * k3, activation_checkpointing,
            )
            x = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        else:
            raise ValueError(f"unknown flow solver: {method}")
    return x


def flow_steps_for_method(args: argparse.Namespace, method: str) -> int:
    if method == "forward_euler":
        return args.euler_flow_steps
    if method == "rk4":
        return args.rk4_flow_steps
    if method == "dlr":
        # DLR-D-Flow's forward pass is itself an Euler integration; only the
        # backward gradient estimate differs from forward_euler.
        return args.euler_flow_steps
    raise ValueError(f"unknown flow solver: {method}")


def velocity_evaluations(method: str, steps: int) -> int:
    if method == "forward_euler":
        return steps
    if method == "rk4":
        return 4 * steps
    if method == "dlr":
        # Not comparable to forward_euler/rk4: each DLR step evaluates the
        # UNet JVP/VJP roughly 2*(rank+probes) times, reported separately.
        return steps
    raise ValueError(f"unknown flow solver: {method}")


def configure_dlr_kernels(resolution: int) -> None:
    dd.HAND_JVP = True
    dd.HAND_VJP = True
    tangent.ATTENTION_MODE = "moments"
    tangent.ATTENTION_JVP_MODE = "chunked" if resolution <= 128 else "moments"
    tangent.ATTENTION_VJP_MODE = "chunked"
    tangent.GN_STATS_MODE = "native" if resolution <= 96 else "manual"


def configure_execution(model: torch.nn.Module, resolution: int,
                        args: argparse.Namespace) -> torch.nn.Module:
    """Apply the run's explicit kernel and model-layout settings before capture."""
    configure_dlr_kernels(resolution)
    tangent.GN_ACTION_BACKEND = "triton" if args.dlr_fused_norm else "native"
    tangent.GN_PRIMAL_BACKEND = args.dlr_fused_primal
    enriched.SPECIALIZE_ZERO_ENRICHMENT = args.dlr_specialized_algebra
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    return model


def relative_l2(estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(estimate - reference) / (
        torch.linalg.vector_norm(reference) + 1e-30)


def normalize_images(images: torch.Tensor) -> torch.Tensor:
    """Convert stored circle tensors to float32 intensities in [0, 1]."""
    if images.dtype == torch.uint8:
        normalized = images.float().div_(255.0)
    elif images.is_floating_point():
        normalized = images.float()
    else:
        raise TypeError(f"Unsupported image dtype: {images.dtype}")
    if normalized.numel() and (
        normalized.min().item() < 0.0 or normalized.max().item() > 1.0
    ):
        raise ValueError("Circle images must have intensities in [0, 1]")
    return normalized


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def add_relative_noise(
    clean: torch.Tensor, level: float, generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    noise = torch.randn(clean.shape, generator=generator, dtype=clean.dtype)
    noise = noise / (noise.norm() + 1e-30) * (level * clean.norm())
    noisy = clean + noise
    realized = relative_l2(noisy, clean).item()
    return noisy, noise, realized


def save_diagnostic_figure(
    path: Path, reconstruction: torch.Tensor, ground_truth: torch.Tensor,
    clean_measurement: torch.Tensor, noisy_measurement: torch.Tensor,
    predicted_measurement: torch.Tensor,
    history: list[float], best_history: list[float], method: str,
    image_relative_error: float, measurement_relative_error: float,
    noise_level: float,
    flow_steps: int,
) -> None:
    estimate = reconstruction[0, 0].detach().cpu().numpy()
    truth = ground_truth[0, 0].detach().cpu().numpy()
    clean_sinogram = clean_measurement[0, 0].detach().cpu().numpy()
    noisy_sinogram = noisy_measurement[0, 0].detach().cpu().numpy()
    predicted_sinogram = predicted_measurement[0, 0].detach().cpu().numpy()
    image_error = estimate - truth
    clean_measurement_error = predicted_sinogram - clean_sinogram
    noisy_measurement_residual = predicted_sinogram - noisy_sinogram
    label = {
        "forward_euler": "Forward Euler (D-Flow)", "rk4": "RK4",
        "dlr": "DLR-D-Flow",
    }[method]
    noise_percent = 100.0 * noise_level
    fig, axes = plt.subplots(3, 3, figsize=(14.5, 12.2))

    def image_panel(axis, array, title, cmap, vmin=None, vmax=None, label_text="Value"):
        artist = axis.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        colorbar = fig.colorbar(artist, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label(label_text)

    sinogram_min = min(float(clean_sinogram.min()), float(noisy_sinogram.min()))
    sinogram_max = max(float(clean_sinogram.max()), float(noisy_sinogram.max()))
    image_panel(axes[0, 0], truth, "Ground truth", cmocean.cm.dense_r, 0, 1, "Intensity")
    image_panel(
        axes[1, 0], estimate, f"Proposed solution ({label})", cmocean.cm.dense_r,
        0, 1, "Intensity")
    image_panel(
        axes[2, 0], image_error, "Signed image error: proposed − GT",
        cmocean.cm.balance, -1, 1, "Signed error")

    image_panel(
        axes[0, 1], clean_sinogram, "Sinogram (clean)", cmocean.cm.dense_r,
        sinogram_min, sinogram_max, "Projection value")
    image_panel(
        axes[1, 1], predicted_sinogram, "Radon transform of proposed solution",
        cmocean.cm.dense_r, sinogram_min, sinogram_max, "Projection value")
    image_panel(
        axes[2, 1], clean_measurement_error,
        "Signed sinogram error: proposed − clean",
        cmocean.cm.balance, -1, 1, "Signed error")

    image_panel(
        axes[0, 2], noisy_sinogram,
        f"Sinogram with {noise_percent:g}% noise", cmocean.cm.dense_r,
        sinogram_min, sinogram_max, "Projection value")
    image_panel(
        axes[1, 2], noisy_measurement_residual,
        "Signed residual: proposed − noisy",
        cmocean.cm.balance, -1, 1, "Signed residual")

    steps = range(1, len(history) + 1)
    axes[2, 2].semilogy(steps, best_history, linewidth=2, label="best so far")
    if noise_level > 0:
        axes[2, 2].axhline(
            noise_level, color="crimson", linestyle="--",
            label=f"{noise_percent:g}% noise level")
    axes[2, 2].set_title("Measurement relative ℓ₂ error")
    axes[2, 2].set_xlabel("Outer iteration")
    axes[2, 2].set_ylabel(r"$\|Ax-y_{noisy}\|_2/\|y_{noisy}\|_2$")
    axes[2, 2].grid(alpha=0.25)
    axes[2, 2].legend()
    fig.suptitle(
        f"{label}: {flow_steps} steps, "
        f"{velocity_evaluations(method, flow_steps)} velocity evaluations | "
        f"image relative ℓ₂={image_relative_error:.3e} | "
        f"measurement relative ℓ₂={measurement_relative_error:.3e}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def data_objective(
    predicted: torch.Tensor,
    measurement: torch.Tensor,
    resolution: int,
    objective: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the optimization objective and squared relative residual."""
    residual_sq = (predicted - measurement).square().sum()
    relative_sq = residual_sq / (measurement.square().sum() + 1e-30)
    if objective == "relative_scaled":
        return relative_sq * float(resolution * resolution), relative_sq
    if objective == "measurement_mse":
        return residual_sq / predicted.numel(), relative_sq
    raise ValueError(f"unknown objective: {objective}")


def source_prior(source: torch.Tensor, args: argparse.Namespace):
    """Return (value, analytic gradient) of the source prior, or (None, None).

    The value is differentiable when `source` requires grad (D-Flow path);
    the analytic gradient is what the DLR path adds to its approximate one.
    """
    lam = args.source_prior_lambda
    if args.source_prior == "none" or lam == 0.0:
        return None, None
    if args.source_prior == "l2":
        value = 0.5 * lam * source.square().sum()
        grad = lam * source.detach()
        return value, grad
    radius = math.sqrt(source.numel())
    norm = source.norm()
    value = 0.5 * lam * (norm - radius) ** 2
    detached = source.detach()
    grad = lam * (detached.norm() - radius) * detached / (detached.norm() + 1e-30)
    return value, grad


class ArmijoLBFGS:
    """L-BFGS with Armijo backtracking on the TRUE objective.

    step(closure) mirrors torch.optim.LBFGS with max_iter=1: the closure must
    set source.grad and is only ever called at accepted iterates, so the
    benchmark's best-iterate bookkeeping stays valid. Trial points are checked
    with `objective` (forward only). The first iteration uses torch's scaled
    steepest-descent step, t = min(1, 1/|g|_1) * lr; later ones start at t = lr
    and shrink by safeguarded quadratic interpolation, t <- clip(t_q, 0.1t, 0.5t).
    A non-descent direction resets the memory to steepest descent; a failed
    search resets the memory and leaves the iterate unchanged. Initial L1
    scaling is used only on the first attempted line search of this optimizer
    instance. Empty curvature memory, failed searches and state.clear()
    rollbacks do not restart that scaling. Clearing state invalidates cached
    values and curvature memory and resets the consecutive-failure streak;
    construct a new optimizer to start a new first-step schedule.

    Trials require a finite negative predicted slope, a representable parameter
    change and strict objective decrease. Compare the decrease to the Armijo
    decrement directly so a tiny decrement cannot round the bound back to f.
    """

    initial_step_policy = LBFGS_INITIAL_STEP_POLICY
    progress_policy = "strict_decrease_nonzero_step"

    def __init__(self, param, objective, lr=1.0, history=100, max_ls=30, c1=1e-4,
                 step_cap=0.0):
        self.param, self.objective = param, objective
        self.step_cap = step_cap
        self.capped_steps = 0
        self.consecutive_failures = 0
        self.lr, self.history, self.max_ls, self.c1 = lr, history, max_ls, c1
        self.S: list[torch.Tensor] = []
        self.Y: list[torch.Tensor] = []
        self.f = None
        self.g = None
        self.state: dict = {}          # rollback code calls optimizer.state.clear()
        self.last_function_evaluations = 0
        self.total_function_evaluations = 0
        self.failed_searches = 0
        # Deliberately outside state: rollback calls state.clear(), but must
        # not make a later attempt look like the first optimizer iteration.
        self.step_attempts = 0

    @property
    def stalled(self) -> bool:
        """Two failed searches in a row: the second already used plain
        steepest descent from the same point with the same cached gradient,
        so every later step would fail identically and the iterate is final."""
        return self.consecutive_failures >= 2

    def zero_grad(self, set_to_none: bool = True):
        self.param.grad = None

    def _direction(self, g):
        q = -g.clone()
        alphas = []
        for s, y in reversed(list(zip(self.S, self.Y))):
            rho = 1.0 / y.dot(s)
            a = rho * s.dot(q)
            q -= a * y
            alphas.append((rho, a))
        if self.S:
            q *= self.S[-1].dot(self.Y[-1]) / self.Y[-1].dot(self.Y[-1])
        for (s, y), (rho, a) in zip(zip(self.S, self.Y), reversed(alphas)):
            b = rho * y.dot(q)
            q += s * (a - b)
        return q

    def _evaluate_gradient(self, closure):
        value = float(closure())
        return value, self.param.grad.detach().flatten().clone()

    def _sufficient_decrease(self, f_new, f, t, gtd):
        return (math.isfinite(f_new) and f_new < f
                and f_new - f <= self.c1 * t * gtd)

    def step(self, closure):
        if self.state == {}:
            # The caller may have restored an earlier parameter value.
            self.S, self.Y = [], []
            self.f, self.g = None, None
            self.consecutive_failures = 0
        if self.f is None:
            self.f, self.g = self._evaluate_gradient(closure)
            self.state["initialised"] = True
        f, g = self.f, self.g
        d = self._direction(g)
        gtd = g.dot(d).item()
        if not math.isfinite(gtd) or gtd >= 0:
            self.S, self.Y = [], []
            d, gtd = -g, -g.dot(g).item()
        valid_direction = math.isfinite(f) and math.isfinite(gtd) and gtd < 0
        # A zero/nonfinite gradient cannot supply a descent direction, even
        # after the steepest-descent fallback. Do not divide by its L1 norm.
        t = 0.0
        if valid_direction:
            t = (min(1.0, 1.0 / g.abs().sum().item())
                 if self.step_attempts == 0 else 1.0) * self.lr
        if valid_direction and self.step_cap > 0:
            d_norm = d.norm().item()
            if t * d_norm > self.step_cap:
                t = self.step_cap / d_norm
                self.capped_steps += 1
        x0 = self.param.detach().flatten().clone()
        evaluations, accepted = 0, False
        self.step_attempts += 1
        for _ in range(self.max_ls):
            if not valid_direction or not math.isfinite(t) or t <= 0:
                break
            candidate = x0 + t * d
            if torch.equal(candidate, x0):
                # Further backtracking cannot recover a representable move.
                break
            f_new = self.objective(candidate.view_as(self.param))
            evaluations += 1
            if self._sufficient_decrease(f_new, f, t, gtd):
                accepted = True
                break
            denominator = 2.0 * (f_new - f - gtd * t)
            if math.isfinite(f_new) and denominator > 0:
                t_q = -gtd * t * t / denominator
            else:
                t_q = 0.1 * t
            t = min(max(t_q, 0.1 * t), 0.5 * t)
        self.last_function_evaluations = evaluations
        self.total_function_evaluations += evaluations
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
        if y_vec.dot(s_vec).item() > 1e-10:
            self.S.append(s_vec)
            self.Y.append(y_vec)
            if len(self.S) > self.history:
                self.S.pop(0)
                self.Y.pop(0)
        self.f, self.g = f_new, g_new
        return f_new


def accepted_trial_record(
    accepted_source: torch.Tensor,
    trial_records: list[tuple[torch.Tensor, float, int]],
) -> tuple[float, int]:
    """Return the objective diagnostic evaluated at the LBFGS-accepted source.

    Strong-Wolfe may return an earlier point from its bracket rather than the
    final trial it evaluated.  PyTorch then reconstructs that accepted point
    exactly from the saved line-search origin, step, and direction.
    """
    for trial_source, relative, closure_index in reversed(trial_records):
        if torch.equal(trial_source, accepted_source):
            return relative, closure_index
    raise RuntimeError(
        "LBFGS accepted a source that was not evaluated by its closure; "
        "cannot report the accepted-iterate objective without spending an "
        "additional closure evaluation."
    )


def run_inverse(
    model: torch.nn.Module,
    ground_truth: torch.Tensor,
    clean_measurement: torch.Tensor,
    noisy_measurement: torch.Tensor,
    angles: torch.Tensor,
    resolution: int,
    sample_index: int,
    method: str,
    source_initial: torch.Tensor,
    checkpoint_sha256: str,
    device_name: str,
    args: argparse.Namespace,
    products=None,
) -> tuple[SampleResult, list[IterationResult]]:
    flow_steps = flow_steps_for_method(args, method)
    source = source_initial.detach().clone().requires_grad_(True)
    effective_max_iter = 1 if args.closure_budget is not None else args.lbfgs_max_iter
    optimizer = torch.optim.LBFGS(
        [source], lr=args.lbfgs_lr, max_iter=effective_max_iter,
        max_eval=args.lbfgs_max_eval,
        history_size=args.history_size, line_search_fn=None,
        tolerance_grad=args.lbfgs_tolerance_grad,
        tolerance_change=args.lbfgs_tolerance_change,
    )
    optimizer.param_groups[0]["line_search_fn"] = (
        "strong_wolfe" if args.lbfgs_line_search == "strong_wolfe" else None
    )
    if args.optimizer == "adam":
        # Adam's step(closure) calls the closure exactly once, at the current
        # iterate, so the closure's best-iterate bookkeeping (line_search ==
        # "none" branch) stays valid unchanged.
        if args.lbfgs_line_search != "none":
            raise ValueError("--optimizer adam requires --lbfgs-line-search none")
        optimizer = torch.optim.Adam([source], lr=args.adam_lr)
    if args.lbfgs_line_search == "armijo":
        if args.optimizer != "lbfgs":
            raise ValueError("--lbfgs-line-search armijo requires --optimizer lbfgs")

        def true_objective(candidate: torch.Tensor) -> float:
            # Forward-only evaluation of the exact objective with the run's own
            # integrator (DLR's forward pass is Euler): no tape, no DLR factors.
            with torch.no_grad():
                x = solve_flow(model, candidate, flow_steps,
                               "forward_euler" if method == "dlr" else method)
                value, _ = data_objective(
                    sparse_radon(x.to(angles.device), angles), noisy_measurement,
                    resolution, args.objective)
                prior_value, _ = source_prior(candidate, args)
                if prior_value is not None:
                    value = value + prior_value
            return float(value)

        optimizer = ArmijoLBFGS(
            source, true_objective, lr=args.lbfgs_lr, history=args.history_size,
            max_ls=args.armijo_max_ls, c1=args.armijo_c1,
            step_cap=args.armijo_step_cap * math.sqrt(source.numel()))
    current_step = 0
    closure_count = 0
    best_relative = float("inf")
    best_source = source.detach().clone()
    best_step = 0
    best_closure = 0
    last_relative = float("inf")
    history: list[float] = []
    best_history: list[float] = []
    iteration_rows: list[IterationResult] = []
    rollback_count = 0
    closure_budget_reached = False
    step_trial_records: list[tuple[torch.Tensor, float, int]] = []

    def closure():
        nonlocal closure_count, best_relative, best_source, best_step
        nonlocal best_closure, last_relative
        if args.closure_budget is not None and closure_count >= args.closure_budget:
            raise ClosureBudgetReached
        closure_count += 1
        optimizer.zero_grad()
        if method == "dlr":
            if args.dlr_integrator == "ksl":
                reconstruction, factors, alpha = dd.sensitivity_factors(
                    model, source.detach(), flow_steps, rank=args.rank,
                    method="enriched", trace_probes=args.trace_probes,
                    initialization_oversampling=args.init_oversampling,
                    enrichment_probes=args.enrichment_probes,
                    factor_stride=args.factor_stride, products=products,
                    checkpoint_blocks=args.dlr_activation_checkpointing,
                )
            else:
                reconstruction, factors, alpha = PB.evolve_parallel_bug(
                    model, source.detach(), flow_steps, rank=args.rank,
                    trace_probes=args.trace_probes,
                    initialization_oversampling=args.init_oversampling,
                    enrichment_probes=args.enrichment_probes, products=products,
                    s11="euler" if args.dlr_integrator == "bug" else "zero",
                    overlap=args.dlr_overlap, zero_terminal_baseline=True,
                    checkpoint_blocks=args.dlr_activation_checkpointing,
                )
            detached_terminal = reconstruction.detach().requires_grad_(True)
            predicted = sparse_radon(detached_terminal.to(angles.device), angles)
            optimization_objective, relative_sq = data_objective(
                predicted, noisy_measurement, resolution, args.objective,
            )
            terminal_gradient, = torch.autograd.grad(
                optimization_objective, detached_terminal)
            source.grad = dd.approx_source_gradient(
                factors, terminal_gradient, alpha).reshape_as(source)
            prior_value, prior_grad = source_prior(source.detach(), args)
            if prior_grad is not None:
                source.grad = source.grad + prior_grad
                optimization_objective = optimization_objective + prior_value
        else:
            reconstruction = solve_flow(
                model, source, flow_steps, method, args.activation_checkpointing,
            )
            predicted = sparse_radon(reconstruction.to(angles.device), angles)
            optimization_objective, relative_sq = data_objective(
                predicted, noisy_measurement, resolution, args.objective,
            )
            prior_value, _ = source_prior(source, args)
            if prior_value is not None:
                optimization_objective = optimization_objective + prior_value
            optimization_objective.backward()
        last_relative = math.sqrt(max(relative_sq.item(), 0.0))
        step_trial_records.append(
            (source.detach().clone(), last_relative, closure_count)
        )
        # Without a line search, every closure is evaluated at the current
        # accepted iterate before the next fixed L-BFGS update is applied.
        if (
            args.lbfgs_line_search in ("none", "armijo")
            and math.isfinite(last_relative)
            and last_relative < best_relative
        ):
            best_relative = last_relative
            best_source = source.detach().clone()
            best_step = current_step
            best_closure = closure_count
        return optimization_objective.detach() if method == "dlr" else optimization_objective

    device = source.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    total_start = time.perf_counter()
    outer_times: list[float] = []
    maximum_outer_steps = (
        args.closure_budget if args.closure_budget is not None else args.outer_steps
    )
    for step in range(1, maximum_outer_steps + 1):
        if args.closure_budget is not None and closure_count >= args.closure_budget:
            closure_budget_reached = True
            break
        current_step = step
        closures_before = closure_count
        accepted_source_before_step = source.detach().clone()
        accepted_relative_before_step = last_relative
        step_trial_records.clear()
        synchronize(device)
        start = time.perf_counter()
        budget_interrupted_step = False
        try:
            optimizer.step(closure)
        except ClosureBudgetReached:
            # A line search requested another trial after consuming the budget.
            # Restore the last accepted iterate so rejected trials are never reported.
            with torch.no_grad():
                source.copy_(accepted_source_before_step)
            last_relative = accepted_relative_before_step
            closure_budget_reached = True
            budget_interrupted_step = True
        synchronize(device)
        elapsed = time.perf_counter() - start
        outer_times.append(elapsed)
        used_closures = closure_count - closures_before
        if budget_interrupted_step:
            print(
                f"res={resolution} sample={sample_index} method={method} "
                f"closure_budget={closure_count} reached during line search",
                flush=True,
            )
            break
        if args.lbfgs_line_search == "strong_wolfe":
            last_relative, accepted_closure = accepted_trial_record(
                source.detach(), step_trial_records,
            )
            if math.isfinite(last_relative) and last_relative < best_relative:
                best_relative = last_relative
                best_source = source.detach().clone()
                best_step = current_step
                best_closure = accepted_closure
        source_is_finite = bool(torch.isfinite(source).all().item())
        should_restore = False
        if args.rollback_policy in {"factor", "nonfinite"}:
            should_restore = not math.isfinite(last_relative) or not source_is_finite
        if args.rollback_policy == "factor":
            should_restore = (
                should_restore
                or last_relative > args.rollback_factor * best_relative
            )
        if should_restore:
            with torch.no_grad():
                source.copy_(best_source)
            reset_optimizer_history(optimizer)
            last_relative = best_relative
            rollback_count += 1
        history.append(last_relative)
        best_history.append(best_relative)
        iteration_rows.append(IterationResult(
            resolution=resolution, sample_index=sample_index, method=method,
            outer_step=step, measurement_relative_l2=last_relative,
            best_measurement_relative_l2=best_relative,
            elapsed_s=elapsed, closure_evaluations=used_closures,
            function_evaluations=getattr(optimizer, "last_function_evaluations", 0),
        ))
        print(
            f"res={resolution} sample={sample_index} method={method} step={step} "
            f"relative={last_relative:.3e} closures={used_closures} "
            f"fevals={getattr(optimizer, 'last_function_evaluations', 0)} "
            f"time={elapsed:.2f}s",
            flush=True,
        )
        if not math.isfinite(last_relative):
            break
        if getattr(optimizer, "stalled", False):
            print(f"res={resolution} sample={sample_index} method={method} "
                  f"stalled at step {step}: two failed line searches in a row",
                  flush=True)
            break
    if args.closure_budget is not None and closure_count >= args.closure_budget:
        closure_budget_reached = True
    synchronize(device)
    total_elapsed = time.perf_counter() - total_start

    with torch.no_grad():
        selected_source = best_source if args.solution_state == "best" else source.detach()
        raw = solve_flow(
            model, selected_source, flow_steps,
            "forward_euler" if method == "dlr" else method,
        )
        reconstruction = raw.clamp(0, 1)
        predicted_raw = sparse_radon(raw.to(angles.device), angles)
        predicted_clamped = sparse_radon(reconstruction.to(angles.device), angles)
        image_relative_error = relative_l2(reconstruction, ground_truth).item()
        noisy_relative_error = relative_l2(predicted_raw, noisy_measurement).item()
        clean_relative_error = relative_l2(predicted_clamped, clean_measurement).item()
    diagnostic_path = (
        args.out_dir / "figures" / f"res_{resolution}" / f"sample_{sample_index:03d}"
        / f"{method}.png"
    )
    save_diagnostic_figure(
        diagnostic_path, reconstruction, ground_truth, clean_measurement,
        noisy_measurement, predicted_clamped, history, best_history, method,
        image_relative_error, noisy_relative_error, args.noise_level,
        flow_steps,
    )
    # Keep the fields themselves: comparison figures across methods cannot be
    # rebuilt from the rendered diagnostic alone.
    np.savez_compressed(
        diagnostic_path.with_suffix(".npz"),
        reconstruction=reconstruction.detach().cpu().numpy(),
        ground_truth=ground_truth.detach().cpu().numpy(),
        clean_measurement=clean_measurement.detach().cpu().numpy(),
        noisy_measurement=noisy_measurement.detach().cpu().numpy(),
        predicted_measurement=predicted_clamped.detach().cpu().numpy(),
    )
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda" else 0.0
    )
    result = SampleResult(
        resolution=resolution, sample_index=sample_index, method=method,
        flow_steps=flow_steps,
        velocity_evaluations_per_flow=velocity_evaluations(method, flow_steps),
        noise_relative_l2=relative_l2(noisy_measurement, clean_measurement).item(),
        steps_run=len(history), closure_evaluations=closure_count, best_step=best_step,
        closure_budget=args.closure_budget,
        closure_budget_reached=closure_budget_reached,
        best_closure_evaluation=best_closure,
        image_relative_l2=image_relative_error,
        measurement_relative_l2_noisy=noisy_relative_error,
        measurement_relative_l2_clean=clean_relative_error,
        elapsed_s=total_elapsed,
        mean_outer_time_s=statistics.mean(outer_times),
        lbfgs_rollbacks=rollback_count,
        device=str(device), device_name=device_name,
        peak_gpu_memory_mb=peak_memory_mb,
        checkpoint_sha256=checkpoint_sha256,
        diagnostic_figure=str(diagnostic_path),
    )
    return result, iteration_rows


COMPARISON_LABEL = {"forward_euler": "D-Flow", "rk4": "RK4", "dlr": "DLR+D-Flow"}


def save_comparison_figure(
    out_dir: Path, resolution: int, sample_index: int, methods: Sequence[str],
    noise_level: float, angle_count: int,
) -> Path | None:
    """2x3 figure: ground truth / D-Flow / DLR+D-Flow images over their Ax.

    Mirrors recon_figure in make_euler_rk4_noise_figures.py. Every panel shows
    the clamped-to-[0,1] best iterate, and the measurement error is that
    clamped field's Ax against the noisy data, so each percentage describes
    exactly the panel beneath it.
    """
    sample_dir = out_dir / "figures" / f"res_{resolution}" / f"sample_{sample_index:03d}"
    fields = {}
    for method in methods:
        path = sample_dir / f"{method}.npz"
        if not path.exists():
            return None
        with np.load(path) as data:
            fields[method] = {key: data[key][0, 0] for key in data.files}
    first = fields[methods[0]]
    truth, noisy = first["ground_truth"], first["noisy_measurement"]

    rc = {"font.family": "TeX Gyre Pagella", "mathtext.fontset": "custom",
          "mathtext.rm": "TeX Gyre Pagella"}
    with matplotlib.rc_context(rc):
        fig, axes = plt.subplots(2, 1 + len(methods),
                                 figsize=(3.85 * (1 + len(methods)), 7.8))
        axes[0, 0].imshow(truth, cmap=cmocean.cm.dense_r, vmin=0, vmax=1)
        axes[0, 0].set_title("Ground truth", fontsize=11)
        axes[1, 0].imshow(noisy, cmap=cmocean.cm.dense_r, aspect="auto")
        axes[1, 0].set_title(
            f"Observed $Ax$ ({100 * noise_level:g}% noise)", fontsize=10)
        for column, method in enumerate(methods, start=1):
            recon = fields[method]["reconstruction"]
            predicted = fields[method]["predicted_measurement"]
            image_error = 100 * np.linalg.norm(recon - truth) / np.linalg.norm(truth)
            measurement_error = (
                100 * np.linalg.norm(predicted - noisy) / np.linalg.norm(noisy))
            label = COMPARISON_LABEL[method]
            if method == "forward_euler" and "dlr" not in methods:
                label = "Euler"      # integrator comparison: both are D-Flow
            axes[0, column].imshow(recon, cmap=cmocean.cm.dense_r, vmin=0, vmax=1)
            axes[0, column].set_title(
                f"{label}\nimage rel. $\\ell_2$ = {image_error:.2f}%", fontsize=10)
            axes[1, column].imshow(predicted, cmap=cmocean.cm.dense_r, aspect="auto")
            axes[1, column].set_title(
                f"{label} reconstructed $Ax$\n"
                f"meas. rel. $\\ell_2$ = {measurement_error:.3g}%", fontsize=10)
        for axis in axes.ravel():
            axis.axis("off")
        fig.suptitle(
            f"{resolution}×{resolution} sparse CT, {angle_count} angles, "
            f"{100 * noise_level:g}% noise — sample {sample_index}",
            fontsize=12.5, y=1.00)
        fig.tight_layout()
        path = out_dir / "figures" / f"recon_sample_{sample_index:03d}.png"
        fig.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(fig)
    return path


def summarize(rows: list[dict], methods: Sequence[str] = METHODS) -> list[dict]:
    summary = []
    for resolution in sorted({int(row["resolution"]) for row in rows}):
        for method in methods:
            group = [
                row for row in rows
                if int(row["resolution"]) == resolution and row["method"] == method
            ]
            if not group:
                continue
            values = lambda key: [float(row[key]) for row in group]
            summary.append({
                "resolution": resolution,
                "method": method,
                "samples": len(group),
                "mean_image_relative_l2": statistics.mean(values("image_relative_l2")),
                "median_image_relative_l2": statistics.median(values("image_relative_l2")),
                "mean_measurement_relative_l2_noisy": statistics.mean(values("measurement_relative_l2_noisy")),
                "mean_measurement_relative_l2_clean": statistics.mean(values("measurement_relative_l2_clean")),
                "mean_elapsed_s": statistics.mean(values("elapsed_s")),
                "median_elapsed_s": statistics.median(values("elapsed_s")),
                "mean_outer_steps": statistics.mean(values("steps_run")),
                "mean_closure_evaluations": statistics.mean(values("closure_evaluations")),
                "closure_budget_reached_count": sum(
                    str(row.get("closure_budget_reached", "")).lower() == "true"
                    for row in group
                ),
                "mean_peak_gpu_memory_mb": statistics.mean(values("peak_gpu_memory_mb")),
            })
    return summary


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_resume_settings(existing: dict, current: dict) -> None:
    immutable = (
        "epoch", "angles", "noise_level", "euler_flow_steps",
        "rk4_flow_steps", "outer_steps", "closure_budget", "lbfgs_lr", "lbfgs_max_iter",
        "lbfgs_max_eval",
        "history_size", "lbfgs_line_search", "lbfgs_tolerance_grad",
        "lbfgs_tolerance_change", "objective", "solution_state",
        "rollback_factor", "base_seed",
        "rollback_policy", "activation_checkpointing", "deterministic",
        "rank", "trace_probes", "init_oversampling", "enrichment_probes",
        "factor_stride", "dlr_integrator", "dlr_overlap", "use_cuda_graphs",
        "optimizer", "adam_lr", "source_prior", "source_prior_lambda",
        "armijo_max_ls", "armijo_c1", "armijo_step_cap", "armijo_initial_step_policy",
        "armijo_progress_policy",
        "dlr_max_jvp_columns", "dlr_max_vjp_columns",
        "dlr_step_jvp_columns", "dlr_step_vjp_columns",
        "dlr_activation_checkpointing",
        "dlr_fused_norm",
        "dlr_fused_primal",
        "dlr_specialized_algebra", "channels_last",
        "lbfgs_initial_step_policy", "lbfgs_history_reset_policy",
    )
    # New execution options were absent from historical settings. Missing
    # means the original disabled option, never an enabled experimental mode.
    legacy_defaults = {
        "dlr_activation_checkpointing": False, "dlr_fused_norm": False,
        "dlr_fused_primal": "native",
        "dlr_specialized_algebra": False, "channels_last": False,
    }
    mismatches = {
        key: (existing.get(key, legacy_defaults.get(key)),
              current.get(key, legacy_defaults.get(key)))
        for key in immutable
        if existing.get(key, legacy_defaults.get(key)) != current.get(key, legacy_defaults.get(key))
    }
    if mismatches:
        raise RuntimeError(f"Incompatible existing output settings: {mismatches}")
    existing_hashes = existing.get("checkpoint_sha256", {})
    current_hashes = current.get("checkpoint_sha256", {})
    hash_mismatches = {
        resolution: (digest, current_hashes.get(resolution))
        for resolution, digest in existing_hashes.items()
        if resolution in current_hashes and current_hashes[resolution] != digest
    }
    if hash_mismatches:
        raise RuntimeError(f"Checkpoint identity changed: {hash_mismatches}")


def main() -> None:
    args = parse_args()
    if args.closure_budget is not None and args.closure_budget <= 0:
        raise ValueError("--closure-budget must be positive")
    if args.lbfgs_max_eval is not None and args.lbfgs_max_eval <= 0:
        raise ValueError("--lbfgs-max-eval must be positive")
    if (
        args.closure_budget is not None
        and args.lbfgs_line_search == "strong_wolfe"
        and args.lbfgs_max_eval is None
    ):
        raise ValueError(
            "Fixed-budget strong-Wolfe LBFGS requires an explicit "
            "--lbfgs-max-eval; use 25 to match the reference allowance."
        )
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
    device = torch.device(args.device)
    if args.dlr_fused_norm and device.type != "cuda":
        raise ValueError("--dlr-fused-norm requires CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    if args.activation_checkpointing is None:
        args.activation_checkpointing = device.type == "cuda"
    euler_evaluations = velocity_evaluations("forward_euler", args.euler_flow_steps)
    rk4_evaluations = velocity_evaluations("rk4", args.rk4_flow_steps)
    torch.set_num_threads(args.cpu_threads)
    try:
        torch.set_num_interop_threads(min(4, args.cpu_threads))
    except RuntimeError:
        pass
    args.out_dir.mkdir(parents=True, exist_ok=True)
    settings = vars(args).copy()
    settings["out_dir"] = str(args.out_dir)
    settings["device"] = str(device)
    settings["device_name"] = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    )
    methods = tuple(args.methods)
    settings["methods"] = list(methods)
    settings["armijo_initial_step_policy"] = (
        ArmijoLBFGS.initial_step_policy if args.lbfgs_line_search == "armijo" else None
    )
    settings["armijo_progress_policy"] = (
        ArmijoLBFGS.progress_policy if args.lbfgs_line_search == "armijo" else None
    )
    settings["lbfgs_initial_step_policy"] = (
        LBFGS_INITIAL_STEP_POLICY if args.optimizer == "lbfgs" else None
    )
    settings["lbfgs_history_reset_policy"] = (
        LBFGS_HISTORY_RESET_POLICY if args.optimizer == "lbfgs" else None
    )
    settings["effective_lbfgs_max_iter"] = (
        1 if args.closure_budget is not None else args.lbfgs_max_iter
    )
    settings["effective_lbfgs_max_eval"] = (
        args.lbfgs_max_eval
        if args.lbfgs_max_eval is not None
        else int(1.25 * settings["effective_lbfgs_max_iter"])
    )
    settings["analysis_policy"] = (
        "intention_to_treat_all_runs_included"
        if args.closure_budget is not None else "standard"
    )
    settings["velocity_evaluations_per_flow"] = {
        "forward_euler": euler_evaluations,
        "rk4": rk4_evaluations,
        "dlr": args.euler_flow_steps,
    }
    settings["dlr_rank"] = args.rank
    settings["dlr_trace_probes"] = args.trace_probes
    settings["dlr_initialization_oversampling"] = args.init_oversampling
    settings["dlr_enrichment_probes"] = args.enrichment_probes
    settings["dlr_factor_stride"] = args.factor_stride
    settings["dlr_cuda_graphs"] = args.use_cuda_graphs
    settings["dlr_integrator"] = args.dlr_integrator
    settings["dlr_overlap"] = args.dlr_overlap
    settings["noise_definition"] = "||noise||_2 / ||clean measurement||_2 = noise_level"
    settings["primary_metric"] = "relative L2 error"
    checkpoint_paths = {
        resolution: (
            run_dir(args.run_dir_template, resolution)
            / "checkpoints" / f"epoch_{args.epoch:04d}.pt"
        )
        for resolution in args.resolutions
    }
    settings["checkpoint_sha256"] = {
        str(resolution): sha256_file(path)
        for resolution, path in checkpoint_paths.items()
    }
    settings_path = args.out_dir / "settings.json"
    if settings_path.exists():
        validate_resume_settings(json.loads(settings_path.read_text()), settings)
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")

    samples_path = args.out_dir / "samples.csv"
    iterations_path = args.out_dir / "iterations.csv"
    sample_rows = read_csv(samples_path)
    iteration_rows = read_csv(iterations_path)
    completed = {
        (int(row["resolution"]), int(row["sample_index"]), row["method"])
        for row in sample_rows
    }

    for resolution in args.resolutions:
        checkpoint_sha256 = settings["checkpoint_sha256"][str(resolution)]
        matching_rows = [
            row for row in sample_rows if int(row["resolution"]) == resolution
        ]
        wrong_hashes = {
            row.get("checkpoint_sha256") for row in matching_rows
            if row.get("checkpoint_sha256") != checkpoint_sha256
        }
        if wrong_hashes:
            raise RuntimeError(
                f"Existing rows for resolution {resolution} use different checkpoints: "
                f"{wrong_hashes}"
            )
        model = load_model(resolution, args.epoch, device, args.run_dir_template)
        model = configure_execution(model, resolution, args)
        products = (
            GraphedAProducts(
                model, torch.zeros(1, 1, resolution, resolution, device=device),
                max_jvp_columns=args.dlr_max_jvp_columns,
                max_vjp_columns=args.dlr_max_vjp_columns,
                checkpoint_blocks=args.dlr_activation_checkpointing,
                max_step_jvp_columns=args.dlr_step_jvp_columns,
                max_step_vjp_columns=args.dlr_step_vjp_columns)
            if "dlr" in methods and args.use_cuda_graphs and device.type == "cuda"
            else None
        )
        payload = torch.load(
            ROOT / f"data/circles-{resolution}-n1.pt",
            map_location="cpu", weights_only=True, mmap=True,
        )
        dataset = payload["test"]
        metadata = payload.get("metadata", {})
        if metadata.get("background_value") != 0 or metadata.get("circle_value") != 1:
            raise RuntimeError(
                f"Unexpected dataset polarity for resolution {resolution}: {metadata}"
            )
        radon_device = torch.device("cpu") if args.deterministic else device
        angles = torch.linspace(
            0, math.pi, args.angles + 1, device=radon_device,
        )[:-1]
        for local_index in range(args.samples):
            sample_index = args.sample_offset + local_index
            if all((resolution, sample_index, method) in completed for method in methods):
                print(f"Skipping completed res={resolution} sample={sample_index}", flush=True)
                continue
            ground_truth = normalize_images(
                dataset[sample_index:sample_index + 1]
            ).to(device)
            with torch.no_grad():
                clean_measurement = sparse_radon(
                    ground_truth.to(radon_device), angles,
                )
            noise_generator = torch.Generator(device="cpu").manual_seed(
                args.base_seed + resolution * 1000 + sample_index)
            clean_measurement_cpu = clean_measurement.cpu()
            noisy_measurement_cpu, _, realized_noise = add_relative_noise(
                clean_measurement_cpu, args.noise_level, noise_generator)
            if not math.isclose(
                realized_noise, args.noise_level, rel_tol=1e-5, abs_tol=1e-7,
            ):
                raise RuntimeError(
                    f"Realized noise {realized_noise} != requested {args.noise_level}"
                )
            noisy_measurement = noisy_measurement_cpu.to(radon_device)
            source_generator = torch.Generator(device="cpu").manual_seed(
                args.base_seed + 10_000_000 + resolution * 1000 + sample_index)
            source_initial = torch.randn(
                ground_truth.shape, generator=source_generator,
            ).to(device)

            for method in methods:
                key = (resolution, sample_index, method)
                if key in completed:
                    print(f"Skipping completed {key}", flush=True)
                    continue
                result, timed_rows = run_inverse(
                    model, ground_truth, clean_measurement, noisy_measurement,
                    angles, resolution, sample_index, method, source_initial,
                    checkpoint_sha256, settings["device_name"], args,
                    products=products if method == "dlr" else None,
                )
                sample_rows.append(asdict(result))
                iteration_rows.extend(asdict(row) for row in timed_rows)
                write_csv(samples_path, sample_rows)
                write_csv(iterations_path, iteration_rows)
                write_csv(args.out_dir / "summary.csv", summarize(sample_rows, methods))
                completed.add(key)
                print(
                    f"Completed res={resolution} sample={sample_index} method={method}: "
                    f"image_rel={result.image_relative_l2:.3e} "
                    f"measurement_rel={result.measurement_relative_l2_noisy:.3e} "
                    f"time={result.elapsed_s:.1f}s",
                    flush=True,
                )
            figure = save_comparison_figure(
                args.out_dir, resolution, sample_index, methods,
                args.noise_level, args.angles,
            )
            if figure is not None:
                print(f"Saved comparison figure {figure}", flush=True)
        del model, dataset


if __name__ == "__main__":
    main()
