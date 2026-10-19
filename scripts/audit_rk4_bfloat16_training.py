"""CPU-only quantization audit for terminal flow-matching training inputs.

Produces an analytic Gaussian conditional-variance bound and a small CPU
autocast proxy. The proxy is not a measurement of CUDA convolution kernels.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
from scipy.special import ndtr
import torch
from torch.utils._python_dispatch import TorchDispatchMode
import benchmark_dflow_dlr_sparse_ct_circles_gpu as bench


def main() -> None:
    torch.set_num_threads(4)
    out = Path("outputs/rk4_training_evidence_audit_2026_09_27")
    out.mkdir(parents=True, exist_ok=True)
    values = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).float().numpy()
    values = np.unique(np.sort(values[np.isfinite(values)].astype(np.float64)))
    edges = (values[1:] + values[:-1]) / 2

    def foreground_variance(t: float) -> float:
        # x_t=t+(1-t)Z on a known foreground pixel; Z~N(0,1).
        # E Var(Z|Q(x_t)) = 1 - sum_bin [phi(a)-phi(b)]^2/P(bin).
        a, b = (edges[:-1] - t) / (1 - t), (edges[1:] - t) / (1 - t)
        keep = (a > -13) & (b < 13)
        a, b = a[keep], b[keep]
        mass = ndtr(b) - ndtr(a)
        keep = mass > 1e-15
        a, b, mass = a[keep], b[keep], mass[keep]
        pa, pb = np.exp(-a * a / 2) / np.sqrt(2 * np.pi), np.exp(-b * b / 2) / np.sqrt(2 * np.pi)
        return float(1 - ((pa - pb) ** 2 / mass).sum())

    bin_average = float(np.mean([foreground_variance(t) for t in np.linspace(.990005, .999995, 1000)]))
    bound = {
        "interpretation": "Oracle conditional FM velocity variance caused solely by rounding x_t to bfloat16, given exact x1 and t; foreground-only whole-image contribution is a lower bound.",
        "last_bin_midpoint_points": 1000,
        "last_bin_foreground_variance_mean": bin_average,
        "rows": [], "last_bin_bounds": {},
    }
    for resolution in (64, 250):
        truth = bench.normalize_images(torch.load(f"data/circles-{resolution}-n1.pt", weights_only=True, mmap=True)["test"][:16])
        fraction = float((truth == 1).float().mean())
        bound["last_bin_bounds"][str(resolution)] = fraction * bin_average
        for t in (.9, .99, .995, .999, .9999):
            variance = foreground_variance(t)
            bound["rows"].append({"resolution": resolution, "foreground_fraction": fraction, "t": t,
                                  "foreground_x0_conditional_variance_after_bfloat16": variance,
                                  "foreground_only_mse_lower_bound": fraction * variance})
    (out / "bfloat16_training_input_bound.json").write_text(json.dumps(bound, indent=2) + "\n")

    seen = []

    class Capture(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if str(func) == "aten.convolution.default" and not seen:
                seen.append({"operator": str(func), "input_dtype": str(args[0].dtype),
                             "weight_dtype": str(args[1].dtype), "shape": list(args[0].shape)})
            return func(*args, **(kwargs or {}))

    model = bench.load_model(64, 100, torch.device("cpu"), "outputs/fm_unet/circles-{res}-noattn-ch16-b8/n1")
    truth = bench.normalize_images(torch.load("data/circles-64-n1.pt", mmap=True, weights_only=True)["test"][:8])
    x0 = torch.randn(truth.shape, generator=torch.Generator().manual_seed(170927))
    rows = []
    for t in (.99, .995, .999, .9999):
        xt, tb = (1 - t) * x0 + t * truth, torch.full((8,), t)
        with torch.no_grad():
            fp = model(xt, tb)
            with torch.autocast("cpu", dtype=torch.bfloat16), Capture():
                bp = model(xt, tb)
        foreground = truth == 1
        rows.append({"t": t, "float32_velocity_mse": float((fp - (truth - x0)).square().mean()),
                     "cpu_bfloat16_velocity_mse": float((bp - (truth - x0)).square().mean()),
                     "fraction_fg_inputs_quantized_to_1": float((xt[foreground].to(torch.bfloat16) == 1).float().mean())})
    proxy = {
        "note": "CPU autocast proxy; no GPU call. Actual dispatcher first convolution captured under CPU bfloat16 autocast. Training requests CUDA bfloat16 autocast; kernels need not yield identical outputs.",
        "first_convolution": seen[0], "rows": rows,
    }
    (out / "bfloat16_cpu_proxy.json").write_text(json.dumps(proxy, indent=2) + "\n")
    print(json.dumps({"last_bin_bounds": bound["last_bin_bounds"], "first_convolution": seen[0]}))


if __name__ == "__main__":
    main()
