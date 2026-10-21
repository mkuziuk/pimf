"""Intrinsic multiscale filtering: the decomposition driver."""

import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import repeat
from numbers import Integral, Real

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ._erf import SQRT_2
from ._newton import _newton_fit_windows
from .contrasts import Quadratic
from .kernels import squared_triangle


@dataclass(frozen=True)
class StageInfo:
    """Per-stage diagnostics; converged means unconverged_windows is zero.

    GD counts rows failing delta <= tol * (1 + max(abs(x))) using each chunk's
    final estimates, ignoring NaNs in the maximum but counting NaN updates.
    A zero count does not bound root distance.
    Newton counts rows without a positional root bracket or resolved flat
    interval at the cap; non-finite estimates or scores remain unconverged.
    GD reports the largest absolute score at the last evaluated iterate, before
    the final update; Newton evaluates it at the returned estimates.
    Parallel fits take maxima of iterations, final_max_delta and max_abs_score,
    propagating NaN, and sum unconverged_windows and bisection_fallbacks.
    Bisection counts include both flat-boundary searches and are zero for GD.
    Direct solvers report iterations=1, final_max_delta=max_abs_score=nan,
    bisection_fallbacks=0 and no unconverged windows.
    """

    stage: int
    window_size: int
    iterations: int
    final_max_delta: float
    bandwidth: float
    converged: bool
    unconverged_windows: int = 0
    bisection_fallbacks: int = 0
    max_abs_score: float = float("nan")


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
    any contrast via psi and curvature. Returns the estimates, iteration count,
    maximum final update, unfinished count, zero fallbacks and largest absolute
    score at the last evaluated iterate, before the final update.
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
        delta = np.abs(x_next - x)
        max_delta = float(np.max(delta))
        x = x_next
        iterations += 1
        # Keep the research stopping rule to preserve regression values.
        threshold = tol * (1.0 + float(np.max(np.abs(x))))
        if max_delta <= threshold:
            break

    count_threshold = tol * (1.0 + float(np.nanmax(np.abs(x), initial=0.0)))
    unconverged_windows = int(np.count_nonzero(~(delta <= count_threshold)))
    return x, iterations, max_delta, unconverged_windows, 0, float(np.max(np.abs(local_score)))


class IMF:
    """Reusable local M-estimator configuration for one-dimensional signals.

    Contrasts with solve() use their closed form regardless of solver. Others
    use clipped gradient descent by default, or safeguarded Newton with
    solver="newton" and a vectorized psi_prime(r). Unspecified max_iter/tol
    use 60/1e-6 for GD and 100/1e-12 for Newton. Each decompose() call returns
    an independent result.
    on_unconverged controls stages that hit max_iter with unconverged windows:
    "warn" emits RuntimeWarning, "raise" raises RuntimeError, "ignore" is silent.
    """

    def __init__(
        self,
        contrast=None,
        kernel=None,
        *,
        boundary="wrap",
        max_iter=None,
        tol=None,
        workers=1,
        solver="gd",
        on_unconverged="warn",
    ):
        if not isinstance(solver, str) or solver not in ("gd", "newton"):
            raise ValueError("solver must be 'gd' or 'newton'")
        if max_iter is None:
            max_iter = 100 if solver == "newton" else 60
        if tol is None:
            tol = 1e-12 if solver == "newton" else 1e-6
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
        if not isinstance(on_unconverged, str) or on_unconverged not in (
            "warn",
            "raise",
            "ignore",
        ):
            raise ValueError('on_unconverged must be "warn", "raise" or "ignore"')
        self.contrast = Quadratic() if contrast is None else contrast
        self.kernel = squared_triangle if kernel is None else kernel
        self.boundary = boundary
        self.max_iter = max_iter
        self.tol = tol
        self.workers = int(workers)
        self.solver = solver
        self.on_unconverged = on_unconverged
        if solver == "newton" and not callable(getattr(self.contrast, "solve", None)):
            if not callable(getattr(self.contrast, "psi_prime", None)):
                raise ValueError("solver='newton' requires contrast.psi_prime(r)")
            try:
                self.contrast.psi_prime(np.zeros(1))
            except NotImplementedError as error:
                raise ValueError("solver='newton' requires contrast.psi_prime(r)") from error

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
        GD chunk stops independently, so GD results can vary with workers near
        the solver tolerance. Newton stops rows independently; built-in contrasts
        return bit-identical values and diagnostics across worker counts. Custom
        contrasts must support concurrent calls without mutating shared state
        or their input arrays.
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
                return solve(windows, weights), 1, float("nan"), 0, 0, float("nan")
            if self.solver == "newton":
                return _newton_fit_windows(windows, weights, self.contrast, self.max_iter, self.tol)
            return _gd_fit_windows(windows, weights, self.contrast, self.max_iter, self.tol)

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
                    component, iterations, delta, unconverged_windows, fallbacks, score = fit(
                        windows, weights
                    )
                else:
                    fits = list(
                        executor.map(fit, (windows[chunk] for chunk in chunks), repeat(weights))
                    )
                    component = np.concatenate([part[0] for part in fits])
                    iterations = max(part[1] for part in fits)
                    delta = float(np.max([part[2] for part in fits]))
                    unconverged_windows = sum(part[3] for part in fits)
                    fallbacks = sum(part[4] for part in fits)
                    score = float(np.max([part[5] for part in fits]))
                if unconverged_windows:
                    noun = "window" if unconverged_windows == 1 else "windows"
                    iteration_noun = "iteration" if iterations == 1 else "iterations"
                    message = (
                        f"Stage {stage} (window size {size}): {unconverged_windows} {noun} "
                        f"did not converge after {iterations} {iteration_noun}. "
                        'Increase max_iter or set on_unconverged="ignore".'
                    )
                    if self.on_unconverged == "raise":
                        raise RuntimeError(message)
                    if self.on_unconverged == "warn":
                        warnings.warn(message, RuntimeWarning, stacklevel=2)
                components.append(component)
                residual = residual - component
                stages.append(
                    StageInfo(
                        stage,
                        size,
                        iterations,
                        delta,
                        h,
                        unconverged_windows == 0,
                        unconverged_windows,
                        fallbacks,
                        score,
                    )
                )
        return IMFResult(np.stack(components), residual, stages, window_sizes, bandwidths)
