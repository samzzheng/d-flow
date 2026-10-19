"""Summarize a completed non-attention spatial-scaling benchmark.

Fits are empirical descriptions of the measured range, not asymptotic bounds.
The input must contain every requested suite/resolution and every timed repeat;
incomplete sweeps fail before any output artifact is written.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("dflow", "dflow_checkpointed", "fast_dlr")
LABELS = {
    "dflow": "D-Flow, no checkpointing",
    "dflow_checkpointed": "D-Flow, checkpoint every step",
    "fast_dlr": "Fast DLR D-Flow",
}
COLORS = {"dflow": "#2864a0", "dflow_checkpointed": "#8496a9", "fast_dlr": "#c55030"}
SUITE_LABELS = {
    "trained": "Resolution-specific non-attention models",
    "fixed250": "Same 250px non-attention model at every size",
}


def completed_rows(report: dict) -> dict[str, list[dict]]:
    """Validate completion independently of the writer's console events."""
    settings = report["settings"]
    suites, resolutions = settings["suites"], settings["resolutions"]
    if len(set(suites)) != len(suites) or len(set(resolutions)) != len(resolutions):
        raise ValueError("Settings contain duplicate suites or resolutions")
    if len(resolutions) < 3:
        raise ValueError("At least three completed resolutions are required for scaling fits")
    expected = {(suite, resolution) for suite in suites for resolution in resolutions}
    found = {}
    for row in report["rows"]:
        key = (row["suite"], row["resolution"])
        if key in found:
            raise ValueError(f"Duplicate benchmark row: {key}")
        found[key] = row
    if set(found) != expected:
        missing, extra = sorted(expected - set(found)), sorted(set(found) - expected)
        raise ValueError(f"Sweep is incomplete or inconsistent: missing={missing}, unexpected={extra}")
    for key, row in found.items():
        if row["pixels"] != row["resolution"] ** 2:
            raise ValueError(f"Pixel count does not match square resolution: {key}")
        arch = row["architecture"]
        if any(arch["att_channels"]) or arch["mid_attn"]:
            raise ValueError(f"Attention is enabled in row {key}")
        for method in METHODS:
            result = row.get("methods", {}).get(method, {})
            if result.get("status") != "ok":
                raise ValueError(f"Cannot make complete comparison: {key}/{method} status={result.get('status')}")
            samples = result.get("samples", [])
            if len(samples) != settings["repeats"] or "stages_median_s" not in result:
                raise ValueError(f"Timing repeats or stage summaries are incomplete: {key}/{method}")
            values = [result.get("median_s", float("nan"))] + [s["wall_s"] for s in samples]
            if not all(np.isfinite(v) and v > 0 for v in values):
                raise ValueError(f"Nonpositive or nonfinite runtime: {key}/{method}")
            if method not in row.get("accuracy", {}):
                raise ValueError(f"Final accuracy/completion record is absent: {key}/{method}")
            stages = result["stages_median_s"]
            if not stages or not all(np.isfinite(v) and v > 0 for v in stages.values()):
                raise ValueError(f"Missing, nonpositive or nonfinite stage timing: {key}/{method}")
    return {suite: [found[suite, r] for r in sorted(resolutions)] for suite in suites}


def r_squared(observed: np.ndarray, predicted: np.ndarray) -> float | None:
    denominator = float(np.square(observed - observed.mean()).sum())
    return None if denominator == 0 else 1 - float(np.square(observed - predicted).sum()) / denominator


def scaling_fit(pixels: list[int], seconds: list[float], resolutions: list[int]) -> dict:
    d, t = np.asarray(pixels, dtype=float), np.asarray(seconds, dtype=float)
    beta, log_constant = np.polyfit(np.log(d), np.log(t), 1)
    # Normalizing the abscissa avoids poor conditioning without changing a+b*d.
    slope_scaled, intercept = np.polyfit(d / d.max(), t, 1)
    slope = slope_scaled / d.max()
    predicted = np.exp(log_constant) * d ** beta
    local = np.diff(np.log(t)) / np.diff(np.log(d))
    return {
        "interpretation": "Empirical fits over this measured pixel range; not asymptotic complexity",
        "number_of_points": len(d), "pixel_range": [int(d[0]), int(d[-1])],
        "power_law": {
            "formula": "seconds = coefficient * pixels ** beta",
            "beta": float(beta), "coefficient": float(np.exp(log_constant)),
            "r_squared_log_space": r_squared(np.log(t), np.log(predicted)),
            "r_squared_seconds_space": r_squared(t, predicted),
            "endpoint_beta": float(np.log(t[-1] / t[0]) / np.log(d[-1] / d[0])),
        },
        "affine": {
            "formula": "seconds = intercept_s + seconds_per_pixel * pixels",
            "intercept_s": float(intercept), "seconds_per_pixel": float(slope),
            "r_squared_seconds_space": r_squared(t, intercept + slope * d),
            "note": "Unconstrained descriptive fit; intercept is not a measured dispatch cost",
        },
        "adjacent_slopes": [
            {"resolution_from": a, "resolution_to": b, "beta": float(exponent)}
            for a, b, exponent in zip(resolutions, resolutions[1:], local)
        ],
    }


def summarize(report: dict, suites: dict[str, list[dict]], source: Path) -> dict:
    summary = {
        "source": str(source.resolve()), "settings": report["settings"],
        "environment": report["environment"], "protocol": report["protocol"],
        "caveats": [
            "Exponents describe measured wall time over the sampled range, not asymptotic work.",
            "d means total pixels H*W; a power in side length H is twice its power in d for square images.",
            "Checkpointed and uncheckpointed D-Flow are separate fixed-policy series.",
            "The trained family changes depth; fixed250 holds weights and architecture constant.",
            "CUDA-event stage times include host-dispatch idle gaps and are not kernel busy times.",
            "Separate stage medians need not sum to the median full-gradient wall time.",
            "Peak memory is additional allocation above existing live buffers, not isolated process memory.",
        ],
        "suites": {},
    }
    for suite, rows in suites.items():
        resolutions = [r["resolution"] for r in rows]
        pixels = [r["pixels"] for r in rows]
        out = {"resolutions": resolutions, "pixels": pixels, "methods": {}}
        for method in METHODS:
            series = [r["methods"][method] for r in rows]
            seconds = [s["median_s"] for s in series]
            stage_names = set(series[0]["stages_median_s"])
            if any(set(s["stages_median_s"]) != stage_names for s in series):
                raise ValueError(f"Stage labels change across {suite}/{method}; refusing mixed fits")
            out["methods"][method] = {
                "median_s": seconds,
                "timed_samples_s": [[p["wall_s"] for p in s["samples"]] for s in series],
                "peak_increment_mib": [s["peak_increment_mib"] for s in series],
                "runtime_fit": scaling_fit(pixels, seconds, resolutions),
                "stages": {
                    stage: {
                        "median_s": [s["stages_median_s"][stage] for s in series],
                        "fit": scaling_fit(pixels, [s["stages_median_s"][stage] for s in series], resolutions),
                    } for stage in sorted(stage_names)
                },
            }
        out["fast_speedup_vs_dflow"] = [
            r["methods"]["dflow"]["median_s"] / r["methods"]["fast_dlr"]["median_s"] for r in rows]
        out["fast_speedup_vs_checkpointed_dflow"] = [
            r["methods"]["dflow_checkpointed"]["median_s"] / r["methods"]["fast_dlr"]["median_s"] for r in rows]
        summary["suites"][suite] = out
    return summary


def make_plots(summary: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    suites = summary["suites"]
    fig, axes = plt.subplots(1, len(suites), figsize=(6.2 * len(suites), 4.7),
                             squeeze=False, sharey=True, layout="constrained")
    for ax, (suite, data) in zip(axes[0], suites.items()):
        d = np.asarray(data["pixels"])
        for method in METHODS:
            result = data["methods"][method]
            t = np.asarray(result["median_s"])
            samples = result["timed_samples_s"]
            interval = np.array([[median - min(v), max(v) - median] for median, v in zip(t, samples)]).T
            beta = result["runtime_fit"]["power_law"]["beta"]
            ax.errorbar(d, t, yerr=interval, color=COLORS[method], marker="o", capsize=2,
                        linewidth=1.7, markersize=4, label=f"{LABELS[method]}  (β={beta:.2f})")
        reference = data["methods"]["fast_dlr"]["median_s"][-1] * d / d[-1]
        ax.plot(d, reference, "--", color="#555555", linewidth=1, label="Linear in pixels (β=1)")
        ax.set(xscale="log", yscale="log", xlabel="Total pixels d = H × W",
               title=SUITE_LABELS.get(suite, suite))
        ax.xaxis.set_major_locator(FixedLocator(d))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}k"))
        ax.minorticks_off()
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc="upper left")
    axes[0, 0].set_ylabel("Seconds per fresh gradient")
    fig.suptitle("Non-attention spatial scaling: empirical timing, no optimizer or line search\n"
                 "Points: warm medians; bars: measured min–max; β: log–log fitted slope", fontsize=12)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"scaling.{extension}", dpi=220)
    plt.close(fig)

    if "fixed250" not in suites:
        return
    data = suites["fixed250"]
    groups = {
        "dflow": [("Forward flow", ["forward"]), ("Backward flow", ["backward"]),
                  ("Terminal objective gradient", ["terminal_gradient"])],
        "fast_dlr": [("Initial factor step", ["initialization"]),
                     ("Recurring factor evolution", ["recurring_evolution"]),
                     ("Final canonicalization + return", ["final_canonicalization", "return_bookkeeping"]),
                     ("Terminal gradient + pullback", ["terminal_gradient", "factor_transpose_action"])],
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharey=True, layout="constrained")
    for ax, method in zip(axes, groups):
        stages = data["methods"][method]["stages"]
        for label, names in groups[method]:
            values = np.sum([stages[name]["median_s"] for name in names], axis=0)
            ax.loglog(data["pixels"], values, "o-", markersize=4, label=label)
        ax.set(title=LABELS[method], xlabel="Total pixels d = H × W")
        ax.xaxis.set_major_locator(FixedLocator(data["pixels"]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}k"))
        ax.minorticks_off()
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc="best")
    axes[0].set_ylabel("Elapsed stage seconds per gradient")
    fig.suptitle("Gradient stages with the same non-attention architecture and weights\n"
                 "CUDA-event elapsed medians include host-dispatch gaps; grouped medians are summed", fontsize=12)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"stages_fixed250.{extension}", dpi=220)
    plt.close(fig)


def summarize_components(directory: Path, resolutions: list[int]) -> dict:
    """Read a complete fixed-architecture component sweep; never fit partial rows."""
    paths = [directory / f"fixed250_{r}.json" for r in resolutions]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"Component sweep is incomplete: missing={missing}")
    rows = [json.loads(path.read_text()) for path in paths]
    expected_names = set(rows[0]["components"])
    if len(expected_names) != 21:
        raise ValueError(f"Expected 21 profiled components, found {len(expected_names)}")
    invariant_keys = ("rank", "trace_probes", "oversampling", "steps", "model_parameters",
                      "architecture", "flags", "dtype", "repeats", "iterations_per_sample")
    for resolution, row in zip(resolutions, rows):
        if row["image_shape"] != [1, 1, resolution, resolution] or row["dimension"] != resolution ** 2:
            raise ValueError(f"Wrong tensor dimensions in component row {resolution}")
        if set(row["components"]) != expected_names:
            raise ValueError(f"Component names change at {resolution}")
        if any(row[key] != rows[0][key] for key in invariant_keys):
            raise ValueError(f"Component protocol or architecture changes at {resolution}")
        if any(row["architecture"]["att_channels"]) or row["architecture"]["mid_attn"]:
            raise ValueError(f"Component model has attention at {resolution}")
        for name, stage in row["components"].items():
            for execution in ("eager", "graph"):
                timing = stage[execution]
                if timing is None:
                    if execution != "graph":
                        raise ValueError(f"Missing eager timing: {resolution}/{name}")
                    continue
                for prefix in ("wall", "cuda_event"):
                    samples = timing[f"{prefix}_ms_samples"]
                    values = [timing[f"{prefix}_ms_median"], *samples]
                    if len(samples) != row["repeats"] or not all(
                            isinstance(v, (float, int)) and np.isfinite(v) and v > 0 for v in values):
                        raise ValueError(f"Incomplete component timing: {resolution}/{name}/{execution}/{prefix}")
    pixels = [resolution ** 2 for resolution in resolutions]
    summary = {
        "sources": [str(path.resolve()) for path in paths],
        "resolutions": resolutions, "pixels": pixels,
        "protocol": {key: rows[0][key] for key in invariant_keys},
        "notes": rows[0]["notes"] + [
            "Power and adjacent slopes are empirical; very small stage timings are noisy.",
            "Timing arrays are milliseconds per component call; fit coefficients are in seconds.",
            "The graph sum and complete-step graph are measured separately, and are not interchangeable.",
            "SVD-containing components are intentionally eager only; no graph timings are imputed.",
        ],
        "components": {},
    }
    for name in sorted(expected_names):
        stages = [row["components"][name] for row in rows]
        metadata = {key: value for key, value in stages[0].items()
                    if key not in ("eager", "graph", "graph_setup_wall_s", "graph_unavailable_reason")}
        result = {"metadata": metadata}
        for execution in ("eager", "graph"):
            timings = [stage[execution] for stage in stages]
            if all(timing is None for timing in timings):
                result[execution] = None
                result["graph_unavailable_reasons"] = [stage.get("graph_unavailable_reason") for stage in stages]
                continue
            if any(timing is None for timing in timings):
                raise ValueError(f"Mixed graph availability for {name}; refusing partial fit")
            result[execution] = {}
            for prefix in ("wall", "cuda_event"):
                medians = [timing[f"{prefix}_ms_median"] for timing in timings]
                result[execution][prefix] = {
                    "median_ms": medians,
                    "samples_ms": [timing[f"{prefix}_ms_samples"] for timing in timings],
                    "fit_seconds": scaling_fit(pixels, [v / 1000 for v in medians], resolutions),
                }
        summary["components"][name] = result
    steady_names = [name for name, item in summary["components"].items()
                    if item["metadata"]["group"] == "steady"]
    attribution = {}
    for execution in ("eager", "graph"):
        times = [summary["components"][name][execution] for name in steady_names]
        if any(timing is None for timing in times):
            raise ValueError(f"Steady components require complete {execution} timings for attribution")
        attribution[execution] = {}
        for prefix in ("wall", "cuda_event"):
            summed = np.sum([timing[prefix]["median_ms"] for timing in times], axis=0)
            combined = summary["components"]["complete_fixed_state_step"][execution][prefix]["median_ms"]
            attribution[execution][prefix] = {
                "sum_independent_stages_ms": summed.tolist(),
                "complete_fixed_state_step_ms": combined,
                "sum_over_combined": (summed / combined).tolist(),
                "sum_fit_seconds": scaling_fit(pixels, (summed / 1000).tolist(), resolutions),
            }
    summary["steady_attribution"] = attribution
    return summary


def make_component_plot(summary: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), layout="constrained")
    components = summary["components"]
    groups = [
        ("Shared primal", ["steady_primal_linearization"], "#577da9"),
        ("12-column JVP", ["steady_jvp"], "#85aeca"),
        ("8-column VJP", ["steady_vjp"], "#e0a675"),
        ("Two thin QRs", ["steady_k_qr", "steady_l_qr"], "#b886aa"),
        ("Remaining factor algebra", ["steady_factor_algebra"], "#8dab87"),
        ("State update + buffer copies", ["steady_trace_state_update", "steady_state_buffer_copies"], "#bab6a6"),
    ]
    x = np.arange(len(summary["resolutions"]))
    bottom = np.zeros(len(x))
    for label, names, color in groups:
        values = np.sum([components[name]["graph"]["cuda_event"]["median_ms"] for name in names], axis=0)
        axes[0].bar(x, values, bottom=bottom, width=.7, label=label, color=color)
        bottom += values
    combined = components["complete_fixed_state_step"]["graph"]["cuda_event"]["median_ms"]
    axes[0].plot(x, combined, "D--", color="#222222", markersize=4, linewidth=1.2,
                 label="Complete step, separate measurement")
    axes[0].set(xticks=x, xticklabels=[str(r) for r in summary["resolutions"]],
                xlabel="Image side length H (pixels)", ylabel="Milliseconds per recurring flow step",
                title="Fast DLR: independently graphed components")
    axes[0].legend(fontsize=7.5, loc="upper left")
    axes[0].grid(axis="y", alpha=.2)
    for name, label, color in (("native_forward", "Native forward", "#2864a0"),
                               ("native_input_vjp", "Native input VJP", "#c55030")):
        for execution, style in (("eager", "o-"), ("graph", "s--")):
            timing = components[name][execution]["cuda_event"]
            beta = timing["fit_seconds"]["power_law"]["beta"]
            axes[1].loglog(summary["pixels"], timing["median_ms"], style, color=color,
                           markersize=4, label=f"{label}, {execution} (β={beta:.2f})")
    axes[1].set(xlabel="Total pixels d = H × W", ylabel="Milliseconds per network operation",
                title="Native network: eager and graph replay")
    axes[1].xaxis.set_major_locator(FixedLocator(summary["pixels"]))
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}k"))
    axes[1].minorticks_off()
    axes[1].grid(alpha=.2)
    axes[1].legend(fontsize=8, loc="upper left")
    fig.suptitle("Same non-attention model at every size: isolated component timings\n"
                 "CUDA elapsed medians; component sums differ from combined latency; slopes are empirical", fontsize=12)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"components_fixed250.{extension}", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "outputs/nonattention_scaling_2026_09_27/results.json")
    parser.add_argument("--out", type=Path, help="Defaults to the input JSON's parent directory")
    parser.add_argument("--components-dir", type=Path,
                        help="Also analyze a complete directory of fixed250_{resolution}.json component profiles")
    parser.add_argument("--components-only", action="store_true",
                        help="Write only component artifacts, preserving existing gradient figures and summary")
    args = parser.parse_args()
    if args.components_only and args.components_dir is None:
        parser.error("--components-only requires --components-dir")
    try:
        report = json.loads(args.input.read_text())
        suites = completed_rows(report)
        summary = summarize(report, suites, args.input)
        component_summary = None
        if args.components_dir is not None:
            if "fixed250" not in suites:
                raise ValueError("Component analysis requires the fixed250 suite")
            component_summary = summarize_components(args.components_dir,
                                                       summary["suites"]["fixed250"]["resolutions"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f"No artifacts written: {error}")
    output = args.out or args.input.parent
    output.mkdir(parents=True, exist_ok=True)
    if not args.components_only:
        make_plots(summary, output)
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        for suite, data in summary["suites"].items():
            for method in METHODS:
                fit = data["methods"][method]["runtime_fit"]
                print(f"{suite:8s} {method:20s}: beta={fit['power_law']['beta']:.3f}, "
                      f"affine R²={fit['affine']['r_squared_seconds_space']:.4f}")
    if component_summary is not None:
        make_component_plot(component_summary, output)
        (output / "component_summary.json").write_text(
            json.dumps(component_summary, indent=2, allow_nan=False) + "\n")
    print(f"Wrote complete-sweep summaries and figures to {output}")


if __name__ == "__main__":
    main()
