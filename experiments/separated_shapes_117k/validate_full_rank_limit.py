"""Check the fast residual evolution against dense autodiff at full rank."""
import argparse
import json
from pathlib import Path

import torch

from dlr_dflow.implementation import enriched
from experiments.ct_mri_sweep.validation import ChunkedAutodiffProducts
from sampler.ode_samplers import euler_flow


class NonlinearFlow(torch.nn.Module):
    def __init__(self, dimension: int, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer("matrix", torch.randn(dimension, dimension,
            generator=generator, dtype=torch.float64)/dimension**.5)

    def forward(self, image: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        values = image.flatten(1)
        velocity = torch.tanh(values @ self.matrix.T)*(1+time[:, None])-.6*values
        return velocity.reshape_as(image)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    rows = []
    for seed in (0, 1):
        model = NonlinearFlow(16, seed)
        source = torch.randn((1, 1, 4, 4), dtype=torch.float64,
            generator=torch.Generator().manual_seed(seed+100))
        for steps in (5, 100):
            exact_endpoint = euler_flow(model, source, steps)
            exact_jacobian = torch.func.jacrev(lambda value: euler_flow(model, value, steps))(source).reshape(16, 16)
            for specialized in (False, True):
                enriched.SPECIALIZE_ZERO_ENRICHMENT = specialized
                for prefix in (None, 2):
                    endpoint, (left, core, right), alpha = enriched.evolve_trace_centered_enriched(
                        model, source, steps, rank=16, trace_probes=0, enrichment_probes=0,
                        initialization_oversampling=0, zero_terminal_baseline=True,
                        products=ChunkedAutodiffProducts(model, columns=16),
                        residual_sketch_steps=prefix, _canonicalize_terminal=False)
                    approximate = alpha*torch.eye(16, dtype=source.dtype)+left @ core @ right.T
                    error = float((approximate-exact_jacobian).norm()/exact_jacobian.norm())
                    torch.testing.assert_close(endpoint, exact_endpoint, rtol=0, atol=1e-13)
                    assert alpha == 0 and error < 1e-10
                    rows.append(dict(seed=seed, steps=steps, specialized=specialized,
                        initialization_prefix=prefix, relative_jacobian_error=error))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(rows=rows,
        conclusion="The fast residual update recovers the actual discrete Euler Jacobian at full rank on these nonlinear CPU models.",
        limitation="This isolates factor algebra and startup. It does not certify truncated approximations, trained U-Net products, FP32 conditioning or optimizer convergence."), indent=2)+"\n")
    print(f"{len(rows)} checks passed; maximum relative error {max(row['relative_jacobian_error'] for row in rows):.3e}")


if __name__ == "__main__":
    main()
