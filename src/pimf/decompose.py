"""Intrinsic multiscale filtering: the decomposition driver."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import repeat
from numbers import Integral, Real

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ._erf import SQRT_2
from .contrasts import Quadratic
from .kernels import squared_triangle


@dataclass(frozen=True)
class StageInfo:
    """Per-stage diagnostics; the closed-form path reports iterations=1 and
    final_max_delta=nan. Parallel fits report maxima across chunks and converge
    only when every chunk converges."""

    stage: int
    window_size: int
    iterations: int
    final_max_delta: float
    bandwidth: float
    converged: bool


@dataclass
class IMFResult:
    """Decomposition output; y == imfs.sum(axis=0) + residual up to float rounding."""

    imfs: np.ndarray
    residual: np.ndarray
    stages: list[StageInfo]
    window_sizes: list[int]
    bandwidths: list[float]

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
    curvature = contrast.curvature()
    if not np.isfinite(curvature) or curvature <= 0:
        raise ValueError("contrast curvature must be finite and positive")
    step = 0.95 / curvature

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


class IMF:
    """Reusable local M-estimator configuration for one-dimensional signals.

    Quadratic contrasts use their closed-form solver. Other contrasts use
    clipped gradient descent. Each decompose() call returns an independent result.
    """

    def __init__(
        self, contrast=None, kernel=None, *, boundary="wrap", max_iter=60, tol=1e-6, workers=1
    ):
        if (
            isinstance(workers, (bool, np.bool_))
            or not isinstance(workers, Integral)
            or workers <= 0
        ):
            raise ValueError("workers must be a positive integer")
        if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, Integral):
            raise ValueError("max_iter must be a positive integer")
        if max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")
        if (
            isinstance(tol, (bool, np.bool_))
            or not isinstance(tol, Real)
            or not np.isfinite(tol)
            or tol <= 0
        ):
            raise ValueError("tol must be finite and positive")
        self.contrast = Quadratic() if contrast is None else contrast
        self.kernel = squared_triangle if kernel is None else kernel
        self.boundary = boundary
        self.max_iter = max_iter
        self.tol = tol
        self.workers = int(workers)

    def decompose(self, y, *, h1=0.25, a=SQRT_2, k_max=8, h_min=None, window_sizes=None):
        """Extract up to k_max components with h[k+1] = h[k] / a.

        Bandwidths are half-widths on a unit-period regular grid, with sample
        spacing 1 / len(y). h1=0.25 covers about half the observations. Kernel
        weights use the exact bandwidth; support windows are rounded outwards
        to odd sizes. Stop before h < h_min, or after a center-only window.

        Set k_max=None to stop by h_min alone. Explicit positive odd window_sizes
        use integer radii and cannot be combined with nondefault schedule options.
        The final residual is separate from the extracted component count.

        workers > 1 fits contiguous chunks of windows in threads, with at least
        64 windows per chunk except the last. Stages remain sequential. Each
        chunk stops independently, so robust results can vary with workers near
        the solver tolerance. Custom contrasts must support concurrent calls
        without mutating shared state or their input arrays.
        """
        y = np.asarray(y, dtype=float)
        if y.ndim != 1 or y.size == 0 or not np.all(np.isfinite(y)):
            raise ValueError("y must be a nonempty, finite one-dimensional signal")

        explicit_windows = window_sizes is not None
        if explicit_windows:
            if h1 != 0.25 or a != SQRT_2 or k_max != 8 or h_min is not None:
                raise ValueError("window_sizes cannot be combined with schedule parameters")
            try:
                window_sizes = list(window_sizes)
            except TypeError as error:
                raise ValueError(
                    "window_sizes must be a sequence of positive odd integers"
                ) from error
            if not window_sizes:
                raise ValueError("window_sizes must not be empty")
            for size in window_sizes:
                if (
                    isinstance(size, (bool, np.bool_))
                    or not isinstance(size, Real)
                    or not np.isfinite(size)
                    or size < 1
                    or size % 2 != 1
                ):
                    raise ValueError("window sizes must be positive odd integers")
            window_sizes = [int(size) for size in window_sizes]
            bandwidths = [size // 2 / len(y) for size in window_sizes]
        else:
            if not isinstance(h1, Real) or not np.isfinite(h1) or not 0 < h1 <= 0.5:
                raise ValueError("h1 must be finite and in (0, 0.5]")
            if not isinstance(a, Real) or not np.isfinite(a) or a <= 1:
                raise ValueError("a must be finite and greater than one")
            if k_max is not None and (
                isinstance(k_max, (bool, np.bool_)) or not isinstance(k_max, Integral) or k_max <= 0
            ):
                raise ValueError("k_max must be a positive integer or None")
            if h_min is not None and (
                not isinstance(h_min, Real) or not np.isfinite(h_min) or not 0 < h_min <= h1
            ):
                raise ValueError("h_min must be finite and in (0, h1]")
            if k_max is None and h_min is None:
                raise ValueError("provide h_min when k_max is None")
            window_sizes, bandwidths = [], []
            h = float(h1)
            while k_max is None or len(bandwidths) < k_max:
                if h_min is not None and h < h_min:
                    if not np.isclose(h, h_min, rtol=1e-14, atol=0):
                        break
                    h = float(h_min)
                radius = h * len(y)
                # Snap grid-boundary roundoff in both support and weights.
                rounded = round(radius)
                if rounded >= 1 and abs(radius - rounded) < 1e-9:
                    radius = float(rounded)
                support_radius = 0 if radius < 1 else int(np.ceil(radius))
                bandwidths.append(h)
                window_sizes.append(2 * support_radius + 1)
                if radius < 1 or h == h_min:
                    break
                h /= a

        residual = y.copy()
        components, stages = [], []
        solve = getattr(self.contrast, "solve", None)

        def fit(windows, weights):
            if callable(solve):
                return solve(windows, weights), 1, float("nan"), True
            component, iterations, delta = _gd_fit_windows(
                windows, weights, self.contrast, self.max_iter, self.tol
            )
            converged = delta <= self.tol * (1 + float(np.max(np.abs(component))))
            return component, iterations, delta, bool(converged)

        chunk_size = (
            len(y) if self.workers == 1 else max(64, (len(y) + self.workers - 1) // self.workers)
        )
        chunks = [slice(start, start + chunk_size) for start in range(0, len(y), chunk_size)]
        with (
            ThreadPoolExecutor(max_workers=len(chunks)) if len(chunks) > 1 else nullcontext()
        ) as executor:
            for stage, (size, h) in enumerate(zip(window_sizes, bandwidths, strict=True), start=1):
                if explicit_windows:
                    weights = self.kernel.weights(size)
                else:
                    weights = self.kernel.weights(size, bandwidth=h * len(y))
                padded = np.pad(residual, size // 2, mode=self.boundary)
                windows = sliding_window_view(padded, size)
                if executor is None:
                    component, iterations, delta, converged = fit(windows, weights)
                else:
                    fits = list(
                        executor.map(fit, (windows[chunk] for chunk in chunks), repeat(weights))
                    )
                    component = np.concatenate([part[0] for part in fits])
                    iterations = max(part[1] for part in fits)
                    delta = max(part[2] for part in fits)
                    converged = all(part[3] for part in fits)
                components.append(component)
                residual = residual - component
                stages.append(StageInfo(stage, size, iterations, delta, h, converged))
        return IMFResult(np.stack(components), residual, stages, window_sizes, bandwidths)
