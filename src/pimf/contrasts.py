"""Contrast functions for the local location fits.

A contrast supplies the loss rho (via __call__), its derivative psi (the
score), and an upper bound on rho'' (curvature) that sets a stable gradient
step. A contrast with a closed-form minimizer may also provide
solve(windows, weights); the decomposition then skips gradient descent.
"""

from numbers import Real

import numpy as np

from ._erf import SQRT_2, SQRT_2_OVER_PI, erf_approx


class Contrast:
    """Base contrast. Subclass and implement __call__, psi, and curvature."""

    def __call__(self, r):
        """Loss rho(r), vectorized."""
        raise NotImplementedError

    def psi(self, r):
        """Score rho'(r), vectorized; drives the gradient-descent update."""
        raise NotImplementedError

    def curvature(self):
        """Upper bound on rho''; the gradient step size is 0.95 / curvature()."""
        raise NotImplementedError


class Quadratic(Contrast):
    """rho(r) = r^2 / 2: the local weighted mean, i.e. the linear IMF."""

    def __call__(self, r):
        return 0.5 * np.asarray(r, dtype=float) ** 2

    def psi(self, r):
        return np.asarray(r, dtype=float)

    def curvature(self):
        return 1.0

    def solve(self, windows, weights):
        """Closed-form minimizer: the weighted mean of each window row."""
        return windows @ weights


class SmoothAbs(Contrast):
    """Smoothed absolute value: |r| convolved with a N(0, H^2) density.

    rho_H(r) = r * erf(r / (sqrt(2) H)) + sqrt(2 / pi) * H * exp(-r^2 / (2 H^2))
    psi_H(r) = erf(r / (sqrt(2) H)), bounded in [-1, 1].

    H > 0 controls the transition from quadratic near zero to |r| in the
    tails: smaller H is more median-like, larger H closer to the local mean.
    H ~ 2 * noise sigma is the research-validated default choice.
    """

    def __init__(self, H):
        if (
            isinstance(H, (bool, np.bool_))
            or not isinstance(H, Real)
            or not np.isfinite(H)
            or H <= 0
        ):
            raise ValueError("H must be finite and positive")
        self.H = float(H)

    def __call__(self, r):
        r = np.asarray(r, dtype=float)
        return r * self.psi(r) + SQRT_2_OVER_PI * self.H * np.exp(-0.5 * (r / self.H) ** 2)

    def psi(self, r):
        return erf_approx(np.asarray(r, dtype=float) / (SQRT_2 * self.H))

    def curvature(self):
        return SQRT_2_OVER_PI / self.H
