"""Coupled AR(1) and VAR(1) simulators with closed-form transfer-entropy ground truth.

For a stationary linear-Gaussian VAR process, the transfer entropy from
source x to target y has the analytic form

    TE(x -> y) = 0.5 * log( det(Sigma_y_marginal) / det(Sigma_y_conditional) )

where ``Sigma_y_marginal`` is the residual variance of y after regressing
on its own past, and ``Sigma_y_conditional`` further conditions on the
past of x. For the specific :func:`coupled_ar1` process here, both
quantities evaluate in closed form via the discrete Lyapunov solution of
the stationary cross-covariance.

References
----------
Barnett L, Barrett AB, Seth AK (2009). "Granger causality and transfer
entropy are equivalent for Gaussian variables." Phys. Rev. Lett. 103,
238701.
Kaiser A, Schreiber T (2002). "Information transfer in continuous
processes." Physica D 166, 43.
"""
from __future__ import annotations

import numpy as np


def _spectral_radius(A: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def _solve_lyapunov_discrete(A: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Solve the discrete Lyapunov equation ``P = A P A.T + Q`` for P.

    Uses the standard vec-Kronecker form ``(I - A⊗A) vec(P) = vec(Q)``.
    """
    n = A.shape[0]
    I = np.eye(n * n)
    vecQ = Q.flatten()
    kron = np.kron(A, A)
    vecP = np.linalg.solve(I - kron, vecQ)
    return vecP.reshape(n, n)


def coupled_ar1(
    *,
    c: float,
    a: float = 0.5,
    sigma: float = 1.0,
    n_samples: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Simulate the coupled AR(1) pair

        x_t = a x_{t-1} + e_t
        y_t = a y_{t-1} + c x_{t-1} + f_t,    e, f ~ N(0, sigma^2)

    and return ``(x, y, TE_true_nats)`` where the analytic TE(x → y) is
    computed from the stationary cross-covariance.
    """
    if abs(a) >= 1:
        raise ValueError("|a| must be < 1 for stationary AR(1)")
    rng = np.random.default_rng(seed)
    e = rng.normal(0, sigma, n_samples)
    f = rng.normal(0, sigma, n_samples)
    x = np.zeros(n_samples)
    y = np.zeros(n_samples)
    for t in range(1, n_samples):
        x[t] = a * x[t - 1] + e[t]
        y[t] = a * y[t - 1] + c * x[t - 1] + f[t]

    # Analytic TE via stationary covariance of [x_t; y_t] under the VAR(1)
    # update matrix A = [[a, 0], [c, a]] with innovation cov sigma^2 I.
    A = np.array([[a, 0.0], [c, a]])
    Q = (sigma ** 2) * np.eye(2)
    if _spectral_radius(A) >= 1:
        raise ValueError("AR parameters yield non-stationary process")
    P = _solve_lyapunov_discrete(A, Q)
    var_x = P[0, 0]
    var_y = P[1, 1]
    cov_xy = P[0, 1]
    # Residual variance of x_{t-1} after conditioning on y_{t-1}.
    var_x_given_y = var_x - (cov_xy ** 2) / var_y if var_y > 0 else var_x
    # y_t = a y_{t-1} + c x_{t-1} + f_t. Conditional on y_past only, the
    # unexplained component is c x_{t-1} + f_t; its variance is
    # c^2 Var(x|y) + sigma^2. Conditional on (y_past, x_past) it is just
    # sigma^2 (the innovation).
    sigma_y_marginal = sigma ** 2 + (c ** 2) * var_x_given_y
    sigma_y_conditional = sigma ** 2
    te = 0.5 * np.log(sigma_y_marginal / sigma_y_conditional)
    return x, y, float(te)


def var1(
    *,
    A: np.ndarray,
    Sigma: np.ndarray,
    n_samples: int,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate a stationary VAR(1) ``x_t = A x_{t-1} + e_t``, ``e_t ~ N(0, Sigma)``.

    Raises ``ValueError`` if the spectral radius of ``A`` is >= 1.
    """
    A = np.asarray(A, dtype=np.float64)
    Sigma = np.asarray(Sigma, dtype=np.float64)
    if _spectral_radius(A) >= 1:
        raise ValueError("VAR(1) unstable: spectral radius >= 1")
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    L = np.linalg.cholesky(Sigma)
    X = np.zeros((n_samples, n))
    for t in range(1, n_samples):
        X[t] = A @ X[t - 1] + L @ rng.normal(size=n)
    return X
