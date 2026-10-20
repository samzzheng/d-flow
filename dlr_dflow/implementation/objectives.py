"""Real-valued data misfits for real or complex measurements."""

import torch


def squared_data_misfit(residual: torch.Tensor) -> torch.Tensor:
    """Return 0.5 * ||residual||², retaining autograd through complex operators."""
    return 0.5 * residual.abs().square().sum()
