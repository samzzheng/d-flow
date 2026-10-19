"""CPU-only counterexample: more accurate integration need not reconstruct better.

The velocity field is known exactly: v(x) = (0, -2*x[1]). Both numerical
generators have full range. We minimize one noisy measurement, x[0]+x[1],
by source gradient flow from zero and stop at the identical discrepancy.
This is an analytic illustration, not an experiment on the circle model.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_dflow_dlr_sparse_ct_circles_gpu as bench


class ExactContraction(torch.nn.Module):
    def __init__(self, rate: float):
        super().__init__()
        self.rate = rate
        self.calls = 0

    def forward(self, x, t):
        self.calls += 1
        return x * x.new_tensor([0.0, -self.rate])


def row(a: float, rate: float, noise: float, tau: float) -> dict:
    # G=diag(1,a), A=[1,1]. For source gradient flow starting at zero,
    # x(s) stays parallel to G G^T A^T=(1,a^2). At first discrepancy
    # crossing, A x = y-tau*noise (positive residual starts below y).
    observed = 1.0 + noise
    predicted = observed - tau * abs(noise)
    x = predicted * np.array([1.0, a * a]) / (1.0 + a * a)
    truth = np.array([1.0, 0.0])
    flow_time = -math.log(tau * abs(noise) / observed) / (1.0 + a * a)
    return {
        "second_coordinate_gain": a,
        "gain_error_vs_exact_flow": abs(a - math.exp(-rate)),
        "source_gradient_flow_stop_time": flow_time,
        "reconstruction": x.tolist(),
        "source": (x / np.array([1.0, a])).tolist(),
        "residual_over_noise": abs(float(x.sum()) - observed) / abs(noise),
        "image_relative_l2": float(np.linalg.norm(x - truth)),
        "generator_determinant": a,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "outputs/rk4_theory_audit_2026_09_28")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    rate, noise, tau = 2.0, 0.01, 1.0
    rows = {}
    for steps in (25, 50, 100, 200, 400, 800):
        for method in ("forward_euler", "rk4"):
            model = ExactContraction(rate)
            result = bench.solve_flow(model, torch.tensor([[1., 1.]],
                                      dtype=torch.float64), steps, method, False)
            a = result[0, 1].item()
            q = rate / steps
            analytic = ((1 - q) if method == "forward_euler" else
                        1 - q + q*q/2 - q**3/6 + q**4/24) ** steps
            assert abs(a - analytic) < 1e-13
            assert model.calls == steps * (4 if method == "rk4" else 1)
            data = row(a, rate, noise, tau)
            assert abs(data["residual_over_noise"] - tau) < 1e-12
            rows[f"{method}_{steps}"] = dict(steps=steps, method=method,
                                              velocity_evaluations=model.calls, **data)
    exact = row(math.exp(-rate), rate, noise, tau)
    assert rows["rk4_100"]["gain_error_vs_exact_flow"] < rows["forward_euler_100"]["gain_error_vs_exact_flow"]
    assert rows["forward_euler_100"]["image_relative_l2"] < rows["rk4_100"]["image_relative_l2"]
    output = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "analytic_counterexample_not_trained_model_evidence",
        "device": "cpu", "dtype": "float64", "rate": rate,
        "truth": [1.0, 0.0], "measurement_operator": [[1.0, 1.0]],
        "noise": noise, "threshold_multiplier": tau,
        "optimizer": "continuous_source_gradient_flow_from_zero",
        "stopping": "first_exact_discrepancy_crossing",
        "source_prior_penalty": None, "rows": rows, "exact_flow": exact,
    }
    (args.out / "counterexample.json").write_text(json.dumps(output, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    for method, label, color in (("forward_euler", "Euler", "#1866a4"),
                                 ("rk4", "RK4", "#b84b33")):
        values = [v for v in rows.values() if v["method"] == method]
        n = [v["steps"] for v in values]
        axes[0].loglog(n, [v["gain_error_vs_exact_flow"] for v in values],
                       "o-", label=label, color=color)
        axes[1].semilogx(n, [100*v["image_relative_l2"] for v in values],
                        "o-", label=label, color=color)
    axes[1].axhline(100*exact["image_relative_l2"], color="0.4", ls="--",
                   label="Exact ODE flow")
    axes[0].set(ylabel="ODE gain error", xlabel="Flow steps", title="RK4 integrates more accurately")
    axes[1].set(ylabel="Reconstruction relative L2 (%)", xlabel="Flow steps",
                title="Euler reconstructs better in this example")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Known smooth field; identical noise-floor stopping; no model error", fontsize=11)
    for suffix in ("png", "pdf"):
        fig.savefig(args.out / f"counterexample.{suffix}", dpi=180)
    print(json.dumps({k: rows[k] for k in ("forward_euler_100", "rk4_100")}, indent=2))


if __name__ == "__main__":
    main()
