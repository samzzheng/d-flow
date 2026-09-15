"""Dynamical Low-Rank Approximation (DLRA).

Implementation of the factor differential equations from

    O. Koch and C. Lubich,
    "Dynamical Low-Rank Approximation",
    SIAM J. Matrix Anal. Appl. 29 (2007), pp. 434-454.

The rank-r approximation Y(t) = U(t) S(t) V(t)^T is evolved so that its
derivative is the orthogonal projection of the target derivative onto the
tangent space of the manifold M_r of rank-r matrices (paper eq. (1.2)).

Two regimes are supported through a single integrator:

  * Data approximation (paper eq. (1.2)):
        given the derivative Adot(t) of a known time-dependent matrix A(t),
        the projected rank-r flow uses Z(t) = Adot(t).

  * Matrix differential equation (paper eq. (6.6), Sec. 6.4-6.5):
        for Abar' = F(Abar), the target derivative is Z = F(Y), evaluated at
        the current low-rank approximation Y = U S V^T.

Both reduce to integrating the factor ODEs (paper eq. (2.8)):

        Sdot = U^T Z V
        Udot = P_U^perp Z V S^{-1}          (P_U^perp = I - U U^T)
        Vdot = P_V^perp Z^T U S^{-T}

with U, V kept orthonormal.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

Factors = Tuple[np.ndarray, np.ndarray, np.ndarray]  # (U, S, V)
RHS = Callable[[float, np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def truncated_svd(A: np.ndarray, r: int) -> Factors:
    """Best rank-r approximation factors (U, S, V) of A, S = diag of top sing. vals.

    This provides the initial value Y(t0) = X(t0) recommended in the paper
    (the pointwise best rank-r approximation of the data at the start).
    """
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    U = U[:, :r]
    S = np.diag(s[:r])
    V = Vt[:r, :].T
    return U, S, V


def regularized_inv(S: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """Inverse of S, optionally regularized (paper Sec. 6.1).

    With eps = 0 this is the plain inverse. With eps > 0 each singular value
    sigma_i of S is replaced by sqrt(sigma_i^2 + eps^2) before inverting, which
    tames the S^{-1} terms in eq. (2.8) when S has tiny singular values
    (the over-approximation / small-gap regime of the paper).
    """
    if eps <= 0.0:
        return np.linalg.inv(S)
    P, sigma, Qt = np.linalg.svd(S)
    sigma_reg = np.sqrt(sigma ** 2 + eps ** 2)
    return (Qt.T * (1.0 / sigma_reg)) @ P.T


def _factor_derivative(
    t: float, U: np.ndarray, S: np.ndarray, V: np.ndarray, rhs: RHS, reg: float
) -> Factors:
    """Right-hand side of the factor ODEs (paper eq. (2.8))."""
    Z = rhs(t, U, S, V)                     # m x n target derivative
    Sinv = regularized_inv(S, reg)

    ZV = Z @ V                              # m x r
    Sdot = U.T @ ZV                         # r x r  = U^T Z V
    Udot = (ZV - U @ (U.T @ ZV)) @ Sinv     # P_U^perp Z V S^{-1}

    ZtU = Z.T @ U                           # n x r
    Vdot = (ZtU - V @ (V.T @ ZtU)) @ Sinv.T  # P_V^perp Z^T U S^{-T}
    return Udot, Sdot, Vdot


def _axpy(a: float, x: Factors, y: Factors) -> Factors:
    return (y[0] + a * x[0], y[1] + a * x[1], y[2] + a * x[2])


def _reorthonormalize(U: np.ndarray, S: np.ndarray, V: np.ndarray) -> Factors:
    """Restore orthonormality of U, V without changing Y = U S V^T.

    The exact factor flow keeps U^T U = V^T V = I (paper Sec. 2.2), but an
    explicit integrator drifts slightly; a thin QR folded into S corrects it.
    """
    QU, RU = np.linalg.qr(U)
    QV, RV = np.linalg.qr(V)
    return QU, RU @ S @ RV.T, QV


def rk4_step(
    t: float, factors: Factors, h: float, rhs: RHS, reg: float = 0.0
) -> Factors:
    """One classical RK4 step of the factor ODEs, then re-orthonormalize."""
    U, S, V = factors
    k1 = _factor_derivative(t, U, S, V, rhs, reg)
    y2 = _axpy(h / 2, k1, factors)
    k2 = _factor_derivative(t + h / 2, *y2, rhs=rhs, reg=reg)
    y3 = _axpy(h / 2, k2, factors)
    k3 = _factor_derivative(t + h / 2, *y3, rhs=rhs, reg=reg)
    y4 = _axpy(h, k3, factors)
    k4 = _factor_derivative(t + h, *y4, rhs=rhs, reg=reg)

    U = U + (h / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
    S = S + (h / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    V = V + (h / 6) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
    return _reorthonormalize(U, S, V)


def integrate(
    rhs: RHS,
    factors0: Factors,
    t_grid: np.ndarray,
    reg: float = 0.0,
):
    """Integrate the DLRA factor ODEs over t_grid, returning factors at each node.

    Parameters
    ----------
    rhs : callable (t, U, S, V) -> Z
        The target derivative matrix Z of shape (m, n). Use Z = Adot(t) for
        data approximation, or Z = F(U S V^T) for a matrix ODE Abar' = F(Abar).
    factors0 : (U, S, V)
        Initial rank-r factors, U (m,r) and V (n,r) orthonormal, S (r,r).
    t_grid : array of monotonically increasing times.
    reg : float
        Regularization for S^{-1} (paper Sec. 6.1); 0 disables it.
    """
    factors = factors0
    out = [factors]
    for k in range(len(t_grid) - 1):
        h = float(t_grid[k + 1] - t_grid[k])
        factors = rk4_step(float(t_grid[k]), factors, h, rhs, reg)
        out.append(factors)
    return out


def to_dense(factors: Factors) -> np.ndarray:
    U, S, V = factors
    return U @ S @ V.T
