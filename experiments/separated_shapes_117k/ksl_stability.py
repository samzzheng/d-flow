"""Experimental exact frozen S/L substeps, retaining the fast Euler target core."""
from collections.abc import Callable

import torch

from dlr_dflow.implementation.types import Factors


def increment_ksl_step(
    A: Callable[[torch.Tensor], torch.Tensor],
    AT: Callable[[torch.Tensor], torch.Tensor],
    factors: Factors,
    probe_sensitivities: torch.Tensor,
    probes: torch.Tensor,
    step_size: float,
    alpha: float,
) -> tuple[Factors, torch.Tensor, float]:
    """Apply KSL to the frozen Euler increment F=A(alpha I+Y)+I.

    K=US+h F V; QR gives U1,R_K. With F held fixed, the S and L
    substeps give L=V(R_K-h U1.T F V).T+h F.T U1=(Y+h F).T U1.
    Thus both the bases and the core approximate the same Euler target.
    This is an explicit increment/retraction variant, not exact integration
    of the continuously varying sensitivity ODE. No extra products are used.
    """
    U, S, V = factors
    US = U @ S
    U1, _ = torch.linalg.qr(US + step_size * (A(alpha * V + US) + V))
    ATU1 = AT(U1)
    L1 = V @ (S.T @ (U.T @ (U1 + step_size * ATU1))) + step_size * (alpha * ATU1 + U1)
    V1, R_L = torch.linalg.qr(L1)
    return (U1, R_L.T, V1), probe_sensitivities, -1.0


def stable_sl_step(
    A: Callable[[torch.Tensor], torch.Tensor],
    AT: Callable[[torch.Tensor], torch.Tensor],
    factors: Factors,
    probe_sensitivities: torch.Tensor,
    probes: torch.Tensor,
    step_size: float,
    alpha: float,
) -> tuple[Factors, torch.Tensor, float]:
    """Analytical-alpha, zero-enrichment experiment; K still uses Euler.

    At fixed U1, let M=U1.T A U1, W=alpha A.T U1+U1, C=W.T V.
    The exact frozen substeps S'=-MS-C and L'=LM.T+W combine to
    L1=V R_K.T+(W-V C.T) integral_0^h exp(s M.T) ds.
    A block exponential evaluates the integral without inverting M or
    explicitly forming the potentially unstable intermediate exp(-h M).
    Recompression still targets the original discrete Euler sensitivity.
    """
    U, S, V = factors
    rank = S.shape[0]
    U1, R_K = torch.linalg.qr(U @ S + step_size * (A(alpha * V + U @ S) + V))
    ATU1 = AT(U1)
    M = ATU1.T @ U1
    W = alpha * ATU1 + U1
    C = W.T @ V
    block = S.new_zeros(2 * rank, 2 * rank)
    block[:rank, :rank] = step_size * M.T
    block[:rank, rank:] = step_size * torch.eye(rank, device=S.device, dtype=S.dtype)
    integral = torch.matrix_exp(block)[:rank, rank:]
    L1 = V @ R_K.T + (W - V @ C.T) @ integral
    V1, _ = torch.linalg.qr(L1)
    cross = V.T @ V1
    core = R_K @ cross + step_size * (W.T @ V1 - C @ cross)
    return (U1, core, V1), probe_sensitivities, -1.0
