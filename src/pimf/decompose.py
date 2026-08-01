"""Intrinsic multiscale filtering: the decomposition driver."""

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .contrasts import Quadratic, SmoothAbs
from .kernels import SquaredTriangle
from .schedule import make_window_schedule


@dataclass(frozen=True)
class StageInfo:
    """Per-stage diagnostics; the closed-form path reports iterations=1 and
    final_max_delta=nan."""

    stage: int
    window_size: int
    iterations: int
    final_max_delta: float


@dataclass
class IMFResult:
    """Decomposition output; y == imfs.sum(axis=0) + residual up to float rounding."""

    imfs: np.ndarray
    residual: np.ndarray
    stages: list[StageInfo]
    window_sizes: list[int]

    @property
    def reconstruction(self):
        return self.imfs.sum(axis=0) + self.residual


def _gd_fit_windows(windows, weights, contrast, max_iter, tol):
    """Minimize sum_u w_u * rho(window_u - x) per row by clipped gradient descent.

    Port of robust_gd_fit_windows from the research notebooks, generalized to
    any contrast via psi and curvature. Returns (x, iterations, max_delta).
    """
    weights = weights / weights.sum()
    row_weights = weights.reshape(1, -1)

    x = np.median(windows, axis=1)
    lower = windows.min(axis=1)
    upper = windows.max(axis=1)

    # The weighted score is Lipschitz with constant curvature() because the
    # weights are normalized, so this step size keeps the iteration stable.
    step = 0.95 / contrast.curvature()

    iterations = 0
    max_delta = float("nan")
    for _ in range(max_iter):
        local_score = np.sum(row_weights * contrast.psi(windows - x[:, None]), axis=1)
        x_next = np.clip(x + step * local_score, lower, upper)
        max_delta = float(np.max(np.abs(x_next - x)))
        x = x_next
        iterations += 1
        if max_delta <= tol * (1.0 + float(np.max(np.abs(x)))):
            break

    return x, iterations, max_delta


def _smooth_stage(residual, window_size, kernel, contrast, boundary, max_iter, tol):
    """One smoothing pass: fit the local location at every position."""
    weights = kernel.weights(window_size)
    radius = window_size // 2
    padded = np.pad(residual, pad_width=radius, mode=boundary)
    windows = sliding_window_view(padded, window_size)

    solve = getattr(contrast, "solve", None)
    if callable(solve):
        return solve(windows, weights), 1, float("nan")
    return _gd_fit_windows(windows, weights, contrast, max_iter, tol)


def imf(y, window_sizes=None, contrast=None, kernel=None, boundary="wrap", max_iter=60, tol=1e-6):
    """Decompose a 1-D signal into multiscale components plus a residual.

    At each stage the current residual is smoothed by a local M-estimator
    defined by kernel and contrast; the smooth becomes that stage's component
    and the recursion continues on what is left:
    r_1 = y, S_k = smooth(r_k), r_{k+1} = r_k - S_k.

    Defaults: window_sizes = make_window_schedule(len(y)), contrast =
    Quadratic() (the linear IMF), kernel = SquaredTriangle(). boundary is
    passed to np.pad ("wrap", "reflect", "edge", ...). max_iter and tol apply
    only when the contrast has no closed-form solve.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if len(y) == 0:
        raise ValueError("y must not be empty")

    if window_sizes is None:
        window_sizes = make_window_schedule(len(y))
    window_sizes = [int(size) for size in window_sizes]
    if contrast is None:
        contrast = Quadratic()
    if kernel is None:
        kernel = SquaredTriangle()

    residual = y.copy()
    imfs = []
    stages = []
    for stage, window_size in enumerate(window_sizes, start=1):
        component, iterations, final_max_delta = _smooth_stage(
            residual, window_size, kernel, contrast, boundary, max_iter, tol
        )
        imfs.append(component)
        residual = residual - component
        stages.append(StageInfo(stage, window_size, iterations, final_max_delta))

    return IMFResult(np.array(imfs), residual, stages, window_sizes)


def linear_imf(y, window_sizes=None, kernel=None, boundary="wrap"):
    """imf() with the Quadratic contrast: the linear (weighted local mean) IMF."""
    return imf(y, window_sizes=window_sizes, contrast=Quadratic(), kernel=kernel, boundary=boundary)


def robust_imf(y, h, window_sizes=None, kernel=None, boundary="wrap", max_iter=60, tol=1e-6):
    """imf() with the SmoothAbs(h) contrast; h ~ 2 * noise sigma works well."""
    return imf(
        y,
        window_sizes=window_sizes,
        contrast=SmoothAbs(h),
        kernel=kernel,
        boundary=boundary,
        max_iter=max_iter,
        tol=tol,
    )
