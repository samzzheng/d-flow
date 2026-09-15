"""Toy example: a matrix ODE whose solution is (approximately) low rank.

Following Koch & Lubich (2007), Sec. 6.5, the linear matrix differential
equation

        Abar'(t) = L Abar(t) + Abar(t) R,        Abar(0) = A0

has the closed-form solution

        Abar(t) = expm(t L) @ A0 @ expm(t R).

Because expm(tL) and expm(tR) are invertible for all t, the rank of Abar(t)
equals the rank of A0 for every t. So if A0 has rank r, the ODE solution lives
*exactly* on the manifold M_r of rank-r matrices -- a clean toy example of "an
ODE solution that is of low rank."

We solve this ODE with Dynamical Low-Rank Approximation (dlra.py): we never
form the (m x n) solution, only the thin factors U (m x r), S (r x r),
V (n x r), evolved by the projected factor ODEs (paper eq. (2.8), (6.16)).

Two experiments:

  Case 1 (exact low rank):  A0 has rank r.  DLRA with rank r reproduces the
      exact solution up to time-integration error.

  Case 2 (approximate low rank):  A0 is full rank with a fast-decaying
      spectrum, so Abar(t) is only approximately rank r.  DLRA with rank r
      tracks the pointwise best rank-r approximation X(t) (from a full SVD),
      reproducing the error behaviour of the paper's Fig. 3.2:
      ||Y - X|| stays proportional to the best-approximation error ||X - A||.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

import dlra


def random_generator(n: int, rng: np.random.Generator, skew_scale: float,
                     sym_scale: float) -> np.ndarray:
    """A generator matrix = (skew rotation) + (small symmetric growth/decay).

    The skew part rotates the singular vectors of Abar(t); the symmetric part
    slowly grows/shrinks the singular values, so S(t) genuinely evolves while
    the rank stays fixed.
    """
    Msk = rng.standard_normal((n, n))
    skew = skew_scale * (Msk - Msk.T) / np.sqrt(n)
    Msy = rng.standard_normal((n, n))
    sym = sym_scale * (Msy + Msy.T) / np.sqrt(n)
    return skew + sym


def make_core(n: int, r: int, rng: np.random.Generator, decay: float,
              exact_low_rank: bool) -> np.ndarray:
    """Build A0 = W1 @ diag(sv) @ W2^T with prescribed singular values.

    exact_low_rank=True  -> only r nonzero singular values (rank exactly r).
    exact_low_rank=False -> full spectrum decaying like `decay`**i (approx rank r).
    """
    W1, _ = np.linalg.qr(rng.standard_normal((n, n)))
    W2, _ = np.linalg.qr(rng.standard_normal((n, n)))
    sv = decay ** np.arange(n)
    if exact_low_rank:
        sv[r:] = 0.0
    return (W1 * sv) @ W2.T


def solve(n: int, r: int, exact_low_rank: bool, n_steps: int = 800,
          seed: int = 0):
    rng = np.random.default_rng(seed)
    L = random_generator(n, rng, skew_scale=1.5, sym_scale=0.3)
    R = random_generator(n, rng, skew_scale=1.5, sym_scale=0.3)
    A0 = make_core(n, r, rng, decay=0.5, exact_low_rank=exact_low_rank)

    t_grid = np.linspace(0.0, 1.0, n_steps + 1)

    # Exact solution of Abar' = L Abar + Abar R, sampled on the grid.
    expL = [expm(t * L) for t in t_grid]
    expR = [expm(t * R) for t in t_grid]
    A_exact = [eL @ A0 @ eR for eL, eR in zip(expL, expR)]

    # DLRA: target derivative is F(Y) = L Y + Y R evaluated at the low-rank Y.
    def rhs(t, U, S, V):
        Y = (U @ S) @ V.T
        return L @ Y + Y @ R

    factors0 = dlra.truncated_svd(A0, r)
    factors = dlra.integrate(rhs, factors0, t_grid)

    # Diagnostics along the trajectory.
    err_YA = np.zeros(len(t_grid))   # ||Y - A||   (DLRA vs exact ODE solution)
    err_XA = np.zeros(len(t_grid))   # ||X - A||   (best rank-r vs exact)
    err_YX = np.zeros(len(t_grid))   # ||Y - X||   (DLRA vs best rank-r)
    sv_exact = np.zeros((len(t_grid), r))
    sv_dlra = np.zeros((len(t_grid), r))
    for k, (A, fac) in enumerate(zip(A_exact, factors)):
        Y = dlra.to_dense(fac)
        X = dlra.to_dense(dlra.truncated_svd(A, r))  # pointwise best rank-r
        err_YA[k] = np.linalg.norm(Y - A)
        err_XA[k] = np.linalg.norm(X - A)
        err_YX[k] = np.linalg.norm(Y - X)
        sv_exact[k] = np.linalg.svd(A, compute_uv=False)[:r]
        sv_dlra[k] = np.linalg.svd(fac[1], compute_uv=False)

    return dict(
        t=t_grid, err_YA=err_YA, err_XA=err_XA, err_YX=err_YX,
        sv_exact=sv_exact, sv_dlra=sv_dlra, A_exact=A_exact, factors=factors,
        n=n, r=r, exact_low_rank=exact_low_rank,
    )


def print_summary(res: dict) -> None:
    tag = "exact rank-r" if res["exact_low_rank"] else "approx rank-r"
    print(f"\n=== {tag} :  n={res['n']}, r={res['r']} ===")
    print(f"  ||A(1)||_F                         = {np.linalg.norm(res['A_exact'][-1]):.4e}")
    print(f"  max_t ||X - A||_F  (best rank-r)   = {res['err_XA'].max():.4e}")
    print(f"  max_t ||Y - X||_F  (DLRA vs best)  = {res['err_YX'].max():.4e}")
    print(f"  max_t ||Y - A||_F  (DLRA vs exact) = {res['err_YA'].max():.4e}")
    denom = res["err_XA"].max()
    if denom > 1e-14:
        print(f"  ratio  max||Y-X|| / max||X-A||     = {res['err_YX'].max()/denom:.3f}")


def make_plots(res_exact: dict, res_approx: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # Top-left: singular values, exact-low-rank case (exact vs DLRA).
    ax = axes[0, 0]
    t = res_exact["t"]
    for i in range(res_exact["r"]):
        ax.plot(t, res_exact["sv_exact"][:, i], "-", color=f"C{i}")
        ax.plot(t[::40], res_exact["sv_dlra"][::40, i], "o", color=f"C{i}", ms=4)
    ax.set_title(f"Exact low rank (r={res_exact['r']}): singular values\n"
                 "line = exact ODE, dots = DLRA")
    ax.set_xlabel("t"); ax.set_ylabel(r"$\sigma_i$")

    # Top-right: DLRA reproduction error for the exact-low-rank case.
    ax = axes[0, 1]
    ax.semilogy(t, np.maximum(res_exact["err_YA"], 1e-16))
    ax.set_title("Exact low rank: DLRA vs exact ODE\n"
                 r"$\|Y-A\|_F$ (integration error only)")
    ax.set_xlabel("t"); ax.set_ylabel(r"$\|Y-A\|_F$")

    # Bottom-left: singular values, approx-low-rank case.
    ax = axes[1, 0]
    t = res_approx["t"]
    for i in range(res_approx["r"]):
        ax.plot(t, res_approx["sv_exact"][:, i], "-", color=f"C{i}")
        ax.plot(t[::40], res_approx["sv_dlra"][::40, i], "o", color=f"C{i}", ms=4)
    ax.set_title(f"Approx low rank (r={res_approx['r']}): top singular values\n"
                 "line = exact, dots = DLRA")
    ax.set_xlabel("t"); ax.set_ylabel(r"$\sigma_i$")

    # Bottom-right: error curves like paper Fig. 3.2.
    ax = axes[1, 1]
    ax.plot(t, res_approx["err_XA"], label=r"$\|X-A\|$ (best rank-r)")
    ax.plot(t, res_approx["err_YX"], label=r"$\|Y-X\|$ (DLRA vs best)")
    ax.plot(t, res_approx["err_YA"], label=r"$\|Y-A\|$ (DLRA vs exact)")
    ax.set_title("Approx low rank: error behaviour (cf. paper Fig. 3.2)")
    ax.set_xlabel("t"); ax.set_ylabel("Frobenius error"); ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"\nsaved figure -> {path}")


def main() -> None:
    res_exact = solve(n=40, r=4, exact_low_rank=True)
    res_approx = solve(n=40, r=4, exact_low_rank=False)
    print_summary(res_exact)
    print_summary(res_approx)

    import os
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    make_plots(res_exact, res_approx, os.path.join(out_dir, "dlra_toy.png"))


if __name__ == "__main__":
    main()
