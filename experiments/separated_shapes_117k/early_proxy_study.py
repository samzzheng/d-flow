"""Five-update proxy initialization versus a matched DLR reset control."""
import argparse
from contextlib import suppress
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import torch

from dlr_dflow.api import load_flow_model
from dlr_dflow.implementation import enriched, tangent_unet
from experiments.ct_mri_sweep import run
from experiments.ct_mri_sweep.operators import MRI
from experiments.separated_shapes_117k.bounded_products import BoundedGraphedProducts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", default="r250_shapes_mri4/sample_002")
    parser.add_argument("--after-queue", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    status = dict(status="waiting", pid=os.getpid(), case=args.case, script_sha256=script_hash)
    run.write_json(args.out/"status.json", status)
    if args.after_queue:
        queue = json.loads(args.after_queue.read_text())
        pid = queue["coordinator_pid"]
        process_path = Path(f"/proc/{pid}/stat")
        state = None
        with suppress(FileNotFoundError):
            state = process_path.read_text().split()
        identity = state[21] if state else None
        while True:
            state = None
            with suppress(FileNotFoundError):
                state = process_path.read_text().split()
            if state is None or state[21] != identity or state[2] == "Z":
                break
            time.sleep(10)
        if json.loads(args.after_queue.read_text())["status"] != "complete":
            raise RuntimeError("Preceding same-sample queue did not complete")
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != script_hash:
        raise RuntimeError("Study changed while queued")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda:1")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.04, device)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    interrupted = [False]

    def stop(signum: int, frame: object) -> None:
        interrupted[0] = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    baseline = Path("outputs/separated_shapes_117k_noiseless_20261021/production")
    inputs = torch.load(baseline/args.case/"inputs.pt", map_location=device, weights_only=True)
    protocol = inputs["protocol"]
    settings = json.loads((baseline/"manifest.json").read_text())["settings"]
    assert protocol["job"]["problem"] == "mri" and settings["flow_steps"] == 5
    assert torch.equal(inputs["clean"], inputs["observed"])
    settings["dlr"].update(rank=32, adaptive_rank=False)
    assert settings["dlr"]["trace_probes"] == settings["dlr"]["enrichment_probes"] == 0
    settings["execution"].update(memory_fraction=.04, cudnn_deterministic=True,
        dlr_full_step_graphs=False, dlr_jvp_columns=20, dlr_vjp_columns=16,
        dlr_fused_norm="native", dlr_fused_primal="native")
    resolution = protocol["job"]["resolution"]
    folder = Path(settings["run_template"].format(resolution=resolution))
    assert hashlib.sha256((folder/"checkpoints/epoch_0500.pt").read_bytes()).hexdigest() == protocol["checkpoint_sha256"]
    model, _ = load_flow_model(resolution, 0, 500, str(device), folder)
    info = protocol["operator"]
    operator = MRI(resolution, 4, info["mask_seed"], device, info["pattern"])
    assert torch.equal(operator.mask, inputs["mask"])
    enriched.SPECIALIZE_ZERO_ENRICHMENT = enriched.SKIP_UNTRUNCATED_SVD = True
    tangent_unet.GN_STATS_MODE = tangent_unet.GN_ACTION_BACKEND = tangent_unet.GN_PRIMAL_BACKEND = "native"
    sketch = torch.randint(0, 2, (inputs["initial"].numel(), 40), device=device,
        generator=torch.Generator(device=device).manual_seed(protocol["dlr_seed"]), dtype=torch.int8).float()
    sketch.mul_(2).sub_(1)
    common = dict(settings=settings, script_sha256=script_hash, prefix_attempts=5,
        source_sha256=hashlib.sha256(inputs["initial"].cpu().numpy().tobytes()).hexdigest(),
        sketch_sha256=hashlib.sha256(sketch.cpu().numpy().tobytes()).hexdigest(),
        note="One five-attempt prefix, then fresh LBFGS/gradient/factors with the same original sketch. No repeated alternation; total budget1000 includes prefix attempts. Same projection/objective/acceptance, no terminal reweighting.")
    run.write_json(args.out/"config.json", common)
    for label, method in (("dlr_prefix", "DLR D-Flow"), ("proxy_prefix", "P-Flow")):
        (args.out/label).mkdir(parents=True, exist_ok=True)
        status.update(status="running", arm=label, phase="prefix")
        run.write_json(args.out/"status.json", status)
        prefix_settings = deepcopy(settings)
        prefix_settings["stopping"]["max_updates"] = 5
        products = BoundedGraphedProducts(model, inputs["initial"], cache_size=4, share_linearization=True,
            full_step_graphs=False, max_jvp_columns=20, max_vjp_columns=16)
        evaluator = run.Evaluator(model, operator, inputs["observed"], prefix_settings, protocol["dlr_seed"], products)
        evaluator.previous_v = sketch.clone()
        experiment = {**common, "prefix_method": method}
        prefix = run.solve(evaluator, method, inputs["initial"], inputs["truth"], inputs["clean"],
            args.out/label/"prefix.pt", {**protocol, "experiment": {**experiment,
                "settings": prefix_settings, "phase": "prefix"}}, interrupted)
        del evaluator, products
        if interrupted[0]:
            status["status"] = "paused"
            run.write_json(args.out/"status.json", status)
            return
        used = prefix["history"][-1]["step"]
        remaining = deepcopy(settings)
        remaining["stopping"]["max_updates"] -= used
        handoff = prefix["source"].to(device)
        status["phase"] = "dlr_continuation"
        run.write_json(args.out/"status.json", status)
        products = BoundedGraphedProducts(model, handoff, cache_size=4, share_linearization=True,
            full_step_graphs=False, max_jvp_columns=20, max_vjp_columns=16)
        evaluator = run.Evaluator(model, operator, inputs["observed"], remaining, protocol["dlr_seed"], products)
        evaluator.previous_v = sketch.clone()
        continued = run.solve(evaluator, "DLR D-Flow", handoff, inputs["truth"], inputs["clean"],
            args.out/label/"dlr_d_flow.pt", {**protocol, "experiment": {**experiment,
                "settings": remaining,
                "phase": "dlr_continuation", "prefix_attempts_used": used,
                "handoff_sha256": hashlib.sha256(prefix["source"].numpy().tobytes()).hexdigest()}}, interrupted)
        assert continued["history"][0]["phi"] == prefix["history"][-1]["phi"]
        history = [{**row, "phase": "prefix"} for row in prefix["history"]]
        history += [{**row, "step": used+row["step"], "phase": "dlr_continuation",
                     "elapsed_seconds": prefix["elapsed_seconds"]+row["elapsed_seconds"]}
                    for row in continued["history"][1:]]
        run.write_json(args.out/label/"combined.json", dict(status=continued["status"], history=history,
            experiment=experiment, total_gradient_evaluations=prefix["evaluations"]["gradients"]+continued["evaluations"]["gradients"]))
        del evaluator, products, prefix, continued
        if interrupted[0]:
            status["status"] = "paused"
            run.write_json(args.out/"status.json", status)
            return
        torch.cuda.empty_cache()
    status["status"] = "complete"
    run.write_json(args.out/"status.json", status)


if __name__ == "__main__":
    main()
