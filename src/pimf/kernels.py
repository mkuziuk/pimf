"""Kernel window weights for the local fits.

A kernel is defined by its profile k(u) on [-1, 1]; the base class turns the
profile into a normalized, symmetric weight vector for an odd window size.
"""

import numpy as np


class Kernel:
    """Base kernel. Subclass and implement profile(u).

    profile(u) must be vectorized and nonnegative on [-1, 1]; any constant
    factor cancels under normalization.
    """

    def profile(self, u):
        """Unnormalized kernel profile k(u) on [-1, 1]."""
        raise NotImplementedError

    def weights(self, window_size):
        """Normalized weight vector for an odd window size."""
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd")

        radius = window_size // 2
        if radius == 0:
            return np.array([1.0])

        offsets = np.arange(-radius, radius + 1)
        u = offsets / radius
        weights = np.asarray(self.profile(u), dtype=float)
        if np.any(weights < 0):
            raise ValueError("kernel profile must be nonnegative")
        total = weights.sum()
        if total <= 0:
            raise ValueError("kernel weights must have positive sum")
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
