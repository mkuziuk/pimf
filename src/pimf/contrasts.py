"""Contrast functions for the local location fits.

A contrast supplies the loss rho (via __call__), its derivative psi (the
score), and an upper bound on rho'' (curvature) that sets a stable gradient
step. A contrast with a closed-form minimizer may also provide
solve(windows, weights); the decomposition then skips gradient descent.
"""

from numbers import Real

import numpy as np

from ._erf import SQRT_2, SQRT_2_OVER_PI, erf_approx

_GRID = np.linspace(-8.0, 8.0, 4097)
_SCORE = erf_approx(_GRID / SQRT_2)
_EXP = np.exp(-0.5 * _GRID**2)
_SCORE_DELTA = np.r_[np.diff(_SCORE), 0.0]
_EXP_DELTA = np.r_[np.diff(_EXP), 0.0]
for _table in (_GRID, _SCORE, _EXP, _SCORE_DELTA, _EXP_DELTA):
    _table.setflags(write=False)


def _lookup_coordinates(r, H):
    with np.errstate(over="ignore"):
        position = np.asarray(np.asarray(r, dtype=float) / H)
    np.clip(position, -8.0, 8.0, out=position)
    position += 8.0
    position *= 256.0
    # NaNs need safe indices, but still propagate through the fractions.
    index = np.fmin(position, 4096.0).astype(np.intp)
    position -= index
    return index, position


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

    By default, interpolate precomputed score and exponential values on 4097
    uniformly spaced standardized residuals in [-8, 8], clamping at the ends.
    This approximates both rho and psi; psi is not the exact derivative of
    the interpolated rho. Use lookup=False for the direct erf_approx path
    used before version 0.4.0.
    """

    def __init__(self, H, *, lookup=True):
        if (
            isinstance(H, (bool, np.bool_))
            or not isinstance(H, Real)
            or not np.isfinite(H)
            or H <= 0
        ):
            raise ValueError("H must be finite and positive")
        if not isinstance(lookup, (bool, np.bool_)):
            raise ValueError("lookup must be a boolean")
        self.H = float(H)
        self.lookup = bool(lookup)

    def __call__(self, r):
        r = np.asarray(r, dtype=float)
        if not self.lookup:
            return r * self.psi(r) + SQRT_2_OVER_PI * self.H * np.exp(-0.5 * (r / self.H) ** 2)
        index, fraction = _lookup_coordinates(r, self.H)
        score = _SCORE[index] + fraction * _SCORE_DELTA[index]
        exponential = _EXP[index] + fraction * _EXP_DELTA[index]
        return r * score + SQRT_2_OVER_PI * self.H * exponential

    def psi(self, r):
        if not self.lookup:
            return erf_approx(np.asarray(r, dtype=float) / (SQRT_2 * self.H))
        index, fraction = _lookup_coordinates(r, self.H)
        result = _SCORE_DELTA[index] * fraction
        result += _SCORE[index]
        return result

    def curvature(self):
        return SQRT_2_OVER_PI / self.H
