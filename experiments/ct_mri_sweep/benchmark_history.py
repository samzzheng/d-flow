"""Measure literal LBFGS two-loop CUDA replay, including history upload costs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch


from p_flow.armijo import lbfgs_direction
from .optimizer import GraphedDirection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_device(args.device)
    rows = []
    for size in (64**2, 250**2):
        for length in (10, 100):
            generator = torch.Generator(device=args.device).manual_seed(91)
            g = torch.randn(size, generator=generator, device=args.device)
            s = [torch.randn_like(g) for _ in range(length)]
            y = [v + .1 * torch.randn_like(g) for v in s]
            started = time.perf_counter()
            replay = GraphedDirection(g, s, y)
            torch.cuda.synchronize()
            capture_seconds = time.perf_counter() - started
            timings = {}
            for label in ("eager", "replay"):
                torch.cuda.synchronize()
                started = time.perf_counter()
                for _ in range(50):
                    if label == "eager":
                        expected = lbfgs_direction(g, s, y)
                    else:
                        result = replay(g, s, y)
                torch.cuda.synchronize()
                timings[label] = (time.perf_counter() - started) / 50
            rows.append({"dimension": size, "history": length, "seconds": timings,
                         "capture_seconds": capture_seconds, "bitwise_equal": torch.equal(result, expected)})
            print(rows[-1], flush=True)
    args.out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
