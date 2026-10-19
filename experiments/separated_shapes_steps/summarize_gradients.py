"""Report same-source gradient errors and the failed-source precision audit."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    root = Path("outputs/separated_shapes_steps_20261019/gradients")
    rows = json.loads((root / "results.json").read_text())["rows"]
    lines = ["# Same-source gradient diagnostics", "",
             "Original larger models, noiseless data, FP32/TF32 off. Exact gradients differentiate the actual Euler map. "
             "Both gradients use the identical native endpoint and terminal gradient. DLR uses a fresh seeded sketch, "
             "analytical alpha=1-t, zero enrichment and oversampling 8. This isolates factor accuracy; it does not reproduce "
             "the previous-V history of an entire optimization trajectory.", "",
             "At the 64 MRI initial source, rank-32 gradient error decreases from 53.3% at five steps to 4.7% at twenty. "
             "The 128 CT source does not show the same monotonic improvement: its initial errors are 12.4%, 63.9%, 43.2%. "
             "Thus more steps can improve DLR's gradient approximation, but this is source-dependent. "
             "A relatively accurate gradient of a poor coarse-grid map can still lead to a failed reconstruction.", "",
             "Subspace error is the distance from the exact gradient to its orthogonal projection into DLR's right basis. "
             "Coefficient error is the remaining error inside that basis. Their squared sum matches total squared error "
             "to within 6.3e-6 times the exact gradient's squared norm in this audit. Eager DLR endpoints match native endpoints exactly.", "",
             "| Case | Source | Steps | Rank | Gradient error % | Cosine | Norm ratio | Subspace error % | Coefficient error % |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['case']} | {row['source']} | {row['steps']} | {row['rank']} | "
                     f"{100*row['relative_gradient_error']:.2f} | {row['cosine']:.4f} | {row['norm_ratio']:.4f} | "
                     f"{100*row['subspace_error_over_exact']:.2f} | {100*row['coefficient_error_over_exact']:.2f} |")
    lines.extend(["", "## Narrow descent region at the failed 128 CT source", "",
        "The initial FP32 finite-difference check failed at this source on the five-step map. "
        "A refined epsilon sweep and FP64 replay resolve the discrepancy: the derivative is correct locally, "
        "but the linear neighborhood is very small. Along the normalized exact descent direction, a displacement "
        "of 1e-4 increases the objective, while 1e-5 decreases it by only about 3.6e-7. "
        "FP32 rounding is comparable to these small decreases. The FP64 check reaches relative derivative error below 1e-7. "
        "FP32 and FP64 gradients differ by 1.28% at five steps, versus 0.00061% at twenty steps at the same source.", "",
        "This is evidence of sharp local curvature plus a precision-sensitive objective decrease in one failed case. "
        "It is not evidence that all failures are floating-point bugs, nor a recommendation to use FP64 in production. "
        "The audit casts the existing FP32 time embeddings to FP64 before the time projection and retains the same CT grid coordinates.", "",
        "## Mathematical interpretation", "",
        "For an Euler step, x_(k+1)=x_k+h*v(t_k,x_k), so its derivative is I+h*A_k, "
        "where A_k is the spatial derivative of the velocity. The exact endpoint Jacobian is the ordered product "
        "(I+h*A_(N-1)) ... (I+h*A_0). Changing N changes both these factors and the states where they are evaluated. "
        "It changes the inverse objective itself, not just the accuracy of a gradient of one fixed objective.", "",
        "The training objective in train_fm_unet.py regresses x_1-x_0 along straight interpolants. "
        "For its ideal conditional-mean velocity, x+(1-t)*v(t,x)=E[x_1 | x_t=x]. "
        "The last uniform Euler step has h=1-t, so it evaluates this denoiser at t=1-h. "
        "Five steps therefore make the last denoising operation at t=0.8, versus t=0.99 for 100 steps. "
        "Conditional averaging can attenuate uncertain structures. For a learned, discretized trajectory this relation "
        "motivates a mechanism; it does not prove the observed failures are caused by conditional averaging.", "",
        "Low-rank approximation cannot guarantee escape from an unfavorable coarse-map basin. "
        "P-Flow changes the source update direction by bypassing the endpoint Jacobian, so it can behave differently "
        "even with the same generator and objective. The paired reconstructions are required to test which configuration actually works."])
    (root / "report.md").write_text("\n".join(lines) + "\n")
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for index, case in enumerate(dict.fromkeys(row["case"] for row in rows)):
        for column, stage in enumerate(("initial", "failed_dflow_final")):
            axis = axes[index, column]
            for rank in (32, 128):
                selected = [row for row in rows if row["case"] == case and row["source"] == stage and row["rank"] == rank]
                axis.plot([row["steps"] for row in selected], [100*row["relative_gradient_error"] for row in selected],
                          marker="o", label=f"rank {rank}")
            axis.set(title=f"{case.replace('_shapes', '')}: {stage}", xlabel="Euler steps", ylabel="Relative gradient error (%)")
            axis.grid(alpha=.2)
            axis.legend()
    figure.savefig(root / "gradient_error.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
