"""Verify completed or live progressive-rank checkpoints from saved arrays."""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--baseline", type=Path,
        default=Path("outputs/separated_shapes_117k_noiseless_20261021/production"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    rows = []
    for path in sorted(args.root.glob("r*/sample_*/*/dlr_d_flow.pt")):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        case = path.parent.parent.relative_to(args.root)
        inputs = torch.load(args.baseline/case/"inputs.pt", map_location="cpu", weights_only=True)
        config = json.loads((path.parent.parent/"config.json").read_text())
        digest = hashlib.sha256(inputs["initial"].numpy().tobytes()).hexdigest()
        assert digest == saved["protocol"]["experiment"]["initial_source_sha256"] == config["initial_source_sha256"]
        for key in ("checkpoint_sha256", "operator", "source_seed", "dlr_seed"):
            assert saved["protocol"][key] == inputs["protocol"][key]
        history = saved["history"]
        snapshot = path.with_name("before_cache_resume.pt")
        if snapshot.exists():
            before = torch.load(snapshot, map_location="cpu", weights_only=True)
            assert saved["protocol"] == before["protocol"]
            assert history[:len(before["history"])] == before["history"]
            assert saved["optimizer"]["step_attempts"] >= before["optimizer"]["step_attempts"]
            assert all(saved["evaluations"][key] >= value for key, value in before["evaluations"].items())
        assert all(right["phi"] <= left["phi"] for left, right in zip(history, history[1:]))
        cap = config["settings"]["optimizer"]["step_cap"] * math.sqrt(inputs["initial"].numel())
        assert all(row.get("source_step_norm", 0) <= cap*(1+1e-4) for row in history)
        # Accumulate CPU norms in double: FP32 CPU reductions can differ from
        # CUDA by over 1e-5 on these arrays even when their entries are identical.
        truth = inputs["truth"].double()
        measurement_type = torch.complex128 if inputs["observed"].is_complex() else torch.float64
        observed = inputs["observed"].to(measurement_type)
        image_error = float((saved["image"].double()-truth).norm()/truth.norm())
        residual = float((saved["measurement"].to(measurement_type)-observed).norm()/observed.norm())
        assert abs(image_error-history[-1]["image_relative_l2"]) < 1e-6
        assert abs(residual-history[-1]["measurement_relative_l2_noisy"]) < 1e-6
        trajectories = list(map(json.loads, (path.parent/"rank_history.jsonl").read_text().splitlines()))
        for record in trajectories:
            ranks = [entry["rank"] for entry in record["ranks"][1:]]
            assert len(ranks) == config["settings"]["flow_steps"]
            assert all(32 <= rank <= 256 for rank in ranks)
            assert all(left >= right for left, right in zip(ranks, ranks[1:]))
            fraction = saved["protocol"]["retain_fraction"]
            if fraction is not None:
                assert ranks[0] == 256
                assert all(right >= math.ceil(fraction*left) for left, right in zip(ranks, ranks[1:]))
            assert all(abs(entry["alpha"]-(1-entry["flow_step"]/len(ranks))) < 1e-14 for entry in record["ranks"])
        rows.append(dict(case=str(case), variant=path.parent.name, status=saved["status"],
            step=history[-1]["step"], image_error_percent=100*image_error,
            residual_percent=100*residual, verified_rank_evaluations=len(trajectories)))
    evidence = dict(checked_at=datetime.datetime.now().astimezone().isoformat(), rows=rows,
        checks=["cached initial source hash", "model/operator/seeds", "monotone saved objective",
                "displacement cap", "saved image/complex residual metrics in FP64",
                "rank bounds, monotonicity and gradual retention", "analytical alpha path",
                "pre-OOM history/protocol preserved where a resume snapshot exists"],
        note="Live checkpoints are partial evidence, not a convergence or accuracy conclusion.")
    (args.root/"live_verification.json").write_text(json.dumps(evidence, indent=2)+"\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
