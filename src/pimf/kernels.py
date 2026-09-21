"""Kernel window weights for the local fits.

A kernel is defined by its profile k(u) on [-1, 1]; the base class turns the
profile into a normalized, symmetric weight vector for an odd window size.
"""

from numbers import Real

import numpy as np


class Kernel:
    """Base kernel. Subclass and implement profile(u).

    profile(u) must be vectorized and nonnegative on [-1, 1]; any constant
    factor cancels under normalization.
    """

    def profile(self, u):
        """Unnormalized kernel profile k(u) on [-1, 1]."""
        raise NotImplementedError

    def weights(self, window_size, *, bandwidth=None):
        """Normalized weights; bandwidth is a radius in samples, defaulting to half the window."""
        if (
            isinstance(window_size, (bool, np.bool_))
            or not isinstance(window_size, Real)
            or not np.isfinite(window_size)
            or window_size < 1
            or window_size % 2 != 1
        ):
            raise ValueError("window_size must be a positive odd integer")
        window_size = int(window_size)
        if bandwidth is not None and (
            isinstance(bandwidth, (bool, np.bool_))
            or not isinstance(bandwidth, Real)
            or not np.isfinite(bandwidth)
            or bandwidth <= 0
        ):
            raise ValueError("bandwidth must be finite and positive")

        radius = window_size // 2
        if radius == 0:
            return np.array([1.0])

        if bandwidth is not None:
            rounded = round(bandwidth)
            if rounded >= 1 and abs(bandwidth - rounded) < 1e-9:
                bandwidth = float(rounded)

        offsets = np.arange(-radius, radius + 1)
        u = offsets / (radius if bandwidth is None else bandwidth)
        support = np.abs(u) <= 1
        profile = np.asarray(self.profile(u[support]), dtype=float)
        if (
            profile.shape != u[support].shape
            or np.any(~np.isfinite(profile))
            or np.any(profile < 0)
        ):
            raise ValueError("kernel profile must return finite nonnegative weights matching u")
        weights = np.zeros(window_size)
        weights[support] = profile
        total = weights.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError("kernel weights must have finite positive sum")
        return weights / total

    __call__ = weights


class SquaredTriangle(Kernel):
    """Squared triangular profile k(u) = 0.75 * (1 - |u|)^2.

    The default kernel of the IMF research project, where it is called
    "Epanechnikov" — a misnomer: the classical Epanechnikov kernel is
    (3/4)(1 - u^2). The 0.75 factor cancels under normalization and is kept
    for parity with the research code. The endpoint weights (|u| = 1) are
    exactly zero.
    """

    def profile(self, u):
        return 0.75 * np.maximum(0.0, 1.0 - np.abs(u)) ** 2


class Epanechnikov(Kernel):
    """Classical Epanechnikov profile k(u) = 0.75 * (1 - u^2) on [-1, 1]."""

    def profile(self, u):
        return 0.75 * np.maximum(0.0, 1.0 - np.asarray(u) ** 2)


class Triangle(Kernel):
    """Triangular profile k(u) = 1 - |u| on [-1, 1]."""

    def profile(self, u):
        return np.maximum(0.0, 1.0 - np.abs(u))


class Uniform(Kernel):
    """Uniform profile k(u) = 0.5 on [-1, 1]."""

    def profile(self, u):
        return 0.5 * (np.abs(u) <= 1)


squared_triangle = SquaredTriangle()
epanechnikov = Epanechnikov()
triangle = Triangle()
uniform = Uniform()
