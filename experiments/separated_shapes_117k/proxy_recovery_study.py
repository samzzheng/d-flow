"""Paired reset-DLR, one-proxy-then-DLR, and proxy-only stalled-source recovery."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import signal
import time

import torch

from dlr_dflow.api import load_flow_model
from dlr_dflow.implementation import enriched, tangent_unet
from experiments.ct_mri_sweep import run
from experiments.ct_mri_sweep.operators import CT, MRI
from experiments.ct_mri_sweep.optimizer import SweepLBFGS
from experiments.separated_shapes_117k.bounded_products import BoundedGraphedProducts
from experiments.separated_shapes_117k.proxy_fallback import proxy_step


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda:1")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.08, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    interrupted = [False]

    def stop(signum: int, frame: object) -> None:
        interrupted[0] = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    parent = torch.load(args.parent, map_location=device, weights_only=True)
    inputs = torch.load(args.inputs, map_location=device, weights_only=True)
    protocol = parent["protocol"]
    settings = deepcopy(protocol["settings"])
    assert parent["status"] == "stalled"
    assert parent["optimizer"]["consecutive_failures"] >= settings["stopping"]["failed_searches"]
    assert settings["dlr"]["trace_probes"] == settings["dlr"]["enrichment_probes"] == 0
    for key in ("checkpoint_sha256", "operator", "source_seed", "dlr_seed"):
        assert protocol[key] == inputs["protocol"][key]
    settings["execution"].update(memory_fraction=.08, dlr_full_step_graphs=False,
        dlr_jvp_columns=20, dlr_vjp_columns=16, dlr_fused_norm="native", dlr_fused_primal="native")
    source = parent["source"].detach().clone()
    initial_attempt = parent["optimizer"]["step_attempts"]
    limit = settings["stopping"]["max_updates"]
    assert initial_attempt < limit
    cap = settings["optimizer"]["step_cap"]*math.sqrt(source.numel())
    resolution = protocol["job"]["resolution"]
    folder = Path(settings["run_template"].format(resolution=resolution))
    assert hashlib.sha256((folder/"checkpoints/epoch_0500.pt").read_bytes()).hexdigest() == protocol["checkpoint_sha256"]
    model, _ = load_flow_model(resolution, 0, 500, str(device), folder)
    info = protocol["operator"]
    operator = (CT(resolution, protocol["job"]["level"], device) if protocol["job"]["problem"] == "ct"
                else MRI(resolution, 4, info["mask_seed"], device, info["pattern"]))
    if isinstance(operator, MRI):
        assert torch.equal(operator.mask, inputs["mask"])
    enriched.SPECIALIZE_ZERO_ENRICHMENT = enriched.SKIP_UNTRUNCATED_SVD = True
    tangent_unet.GN_STATS_MODE = tangent_unet.GN_ACTION_BACKEND = tangent_unet.GN_PRIMAL_BACKEND = "native"
    experiment = dict(parent=str(args.parent), parent_sha256=hashlib.sha256(args.parent.read_bytes()).hexdigest(),
        initial_source_sha256=hashlib.sha256(source.cpu().numpy().tobytes()).hexdigest(),
        settings=settings, initial_attempt=initial_attempt, attempt_limit=limit,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        fallback_sha256=hashlib.sha256(Path(proxy_step.__wrapped__.__code__.co_filename).read_bytes()).hexdigest(),
        note="All arms start at the same stalled source. Reset DLR uses fresh LBFGS with existing attempt count and previous V as a sketch seed. Hybrid tries exactly one objective-safeguarded proxy step, then fresh LBFGS/DLR; no curvature spans the switch. Proxy-only repeats that safeguarded proxy, without LBFGS or a proxy Armijo slope. Projection unchanged; all attempts including the rescue count toward the original1000 limit.")
    args.out.mkdir(parents=True, exist_ok=True)
    config = args.out/"config.json"
    if config.exists():
        assert json.loads(config.read_text()) == experiment
    run.write_json(config, experiment)

    for arm in ("reset_dlr", "proxy_then_dlr", "proxy_only"):
        if interrupted[0]:
            break
        path = args.out/arm
        path.mkdir(exist_ok=True)
        method = "P-Flow" if arm == "proxy_only" else "DLR D-Flow"
        checkpoint = path/("p_flow.pt" if method == "P-Flow" else "dlr_d_flow.pt")
        paired = {**protocol, "proxy_recovery": {**experiment, "arm": arm}}
        if checkpoint.exists():
            existing = torch.load(checkpoint, map_location=device, weights_only=True)
            assert existing["protocol"] == paired
            if existing["status"] in run.TERMINAL or existing["status"] == "proxy_rejected":
                continue
        products = (None if arm == "proxy_only" else BoundedGraphedProducts(model, source,
            cache_size=4, share_linearization=True, full_step_graphs=False,
            max_jvp_columns=20, max_vjp_columns=16))
        evaluator = run.Evaluator(model, operator, inputs["observed"], settings, protocol["dlr_seed"], products)
        evaluator.method = method
        start = time.monotonic()
        current = source.clone()
        evaluator.objective(current)
        torch.testing.assert_close(evaluator.image, parent["image"], rtol=0, atol=0)
        history = [dict(step=initial_attempt, phi=evaluator.value, phase="stalled_parent",
            **run.metrics(evaluator.image, evaluator.measurement, inputs["truth"], inputs["clean"], inputs["observed"]))]
        attempt = initial_attempt
        status = "running"
        failures = 0
        prior_seconds = 0.
        if arm == "proxy_only" and checkpoint.exists():
            current = existing["source"]
            history, attempt = existing["history"], existing["history"][-1]["step"]
            failures, prior_seconds = existing["failures"], existing["elapsed_seconds"]
            evaluator.counts = existing["evaluations"]
            evaluator.objective(current)
        while arm != "reset_dlr" and (not checkpoint.exists() or arm == "proxy_only"):
            if interrupted[0]:
                status = "paused"
                break
            _, gradient = run.endpoint_gradient(evaluator.image, operator, inputs["observed"])
            evaluator.counts["gradients"] += 1
            trial = proxy_step(current, gradient, evaluator.objective, cap,
                settings["optimizer"]["max_ls"], settings["source_projection"])
            attempt += 1
            displacement = float((trial.source-current).norm())
            current = trial.source
            failures = 0 if trial.accepted else failures+1
            event = dict(accepted=trial.accepted, value=trial.value, margin=trial.margin,
                baseline_values=trial.baseline_values, trials=trial.trials)
            event.update(step=attempt, displacement=displacement)
            with (path/"proxy_trials.jsonl").open("a") as output:
                output.write(json.dumps(event)+"\n")
            history.append(dict(step=attempt, phi=evaluator.value, phase="proxy",
                source_step_norm=displacement, line_search_trials=len(trial.trials),
                **run.metrics(evaluator.image, evaluator.measurement, inputs["truth"], inputs["clean"], inputs["observed"])))
            if arm == "proxy_then_dlr":
                status = ("proxy_rejected" if not trial.accepted else
                          "step_limit" if attempt >= limit else "running")
                break
            status = ("stalled" if failures >= settings["stopping"]["failed_searches"] else
                      "step_limit" if attempt >= limit else "running")
            if status != "running" or attempt % 10 == 0:
                saved = dict(protocol=paired, method=method, status=status, source=current,
                    image=evaluator.image, measurement=evaluator.measurement, history=history,
                    evaluations=evaluator.counts, failures=failures,
                    elapsed_seconds=prior_seconds+time.monotonic()-start)
                run.save_tensor(checkpoint, saved)
                run.write_json(checkpoint.with_suffix(".json"), {k:v for k,v in saved.items()
                    if k not in ("source", "image", "measurement")})
                print(arm, status, history[-1], flush=True)
            if status != "running":
                break
        if arm == "proxy_only" and status == "paused":
            saved = dict(protocol=paired, method=method, status=status, source=current,
                image=evaluator.image, measurement=evaluator.measurement, history=history,
                evaluations=evaluator.counts, failures=failures,
                elapsed_seconds=prior_seconds+time.monotonic()-start)
            run.save_tensor(checkpoint, saved)
            run.write_json(checkpoint.with_suffix(".json"), {k:v for k,v in saved.items()
                if k not in ("source", "image", "measurement")})
        if arm != "proxy_only":
            if not checkpoint.exists():
                optimizer = SweepLBFGS(current, evaluator.objective, **{
                    **settings["optimizer"], "step_cap": cap, "project": settings["source_projection"]})
                optimizer.step_attempts = attempt
                # No gradient or curvature from the proxy proposal enters DLR.
                assert not optimizer.S and not optimizer.Y and optimizer.g is None
                saved = dict(protocol=paired, method=method, status=status, source=current,
                    image=evaluator.image, measurement=evaluator.measurement, history=history,
                    previous_v=parent.get("previous_v"), evaluations=evaluator.counts,
                    optimizer={key:value for key,value in vars(optimizer).items() if key not in ("source", "objective")},
                    elapsed_seconds=time.monotonic()-start)
                run.save_tensor(checkpoint, saved)
                run.write_json(checkpoint.with_suffix(".json"), {key:saved[key] for key in
                    ("protocol", "method", "status", "history", "evaluations", "elapsed_seconds")})
            if status != "proxy_rejected" and not interrupted[0]:
                result = run.solve(evaluator, method, current, inputs["truth"], inputs["clean"],
                    checkpoint, paired, interrupted)
                print(arm, result["status"], result["history"][-1], flush=True)
        del evaluator, products
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
