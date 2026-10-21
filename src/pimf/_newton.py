"""Safeguarded, independently stopped local score roots."""

import numpy as np


def _newton_fit_windows(windows, weights, contrast, max_iter, tol):
    """Return roots, max iterations/delta, unfinished count, fallbacks and score.

    Each row brackets a nonincreasing weighted score. The rtsafe safeguard
    bisects unless Newton stays inside the bracket and reduces the step to
    less than half the step before last. Flat zero intervals use the midpoint
    of their two boundaries, each located by bisection within the same cap.
    """
    positive = weights > 0
    windows = windows[:, positive]
    weights = weights[positive] / weights[positive].sum()
    x = np.sum(windows * weights, axis=1)
    lo, hi = windows.min(axis=1), windows.max(axis=1)
    previous_step = hi - lo
    older_step = previous_step.copy()
    delta = np.zeros(len(x))
    done = np.zeros(len(x), dtype=bool)
    flat = np.zeros(len(x), dtype=bool)
    zero_left, zero_right = x.copy(), x.copy()
    fallbacks = 0
    iterations = 0

    for _ in range(max_iter):
        iterations += 1
        rows = np.flatnonzero(~done & ~flat)
        if rows.size:
            residuals = windows[rows] - x[rows, None]
            score = np.sum(weights * contrast.psi(residuals), axis=1)
            slope = np.sum(weights * contrast.psi_prime(residuals), axis=1)
            at_zero = score == 0
            plateau = at_zero & (slope == 0) & (lo[rows] < hi[rows])
            flat[rows[plateau]] = True
            zero_left[rows[plateau]] = x[rows[plateau]]
            zero_right[rows[plateau]] = x[rows[plateau]]
            done[rows[at_zero & ~plateau]] = True
            delta[rows[at_zero & ~plateau]] = 0.0

            moving = ~at_zero
            rows, score, slope = rows[moving], score[moving], slope[moving]
            lo[rows] = np.where(score > 0, x[rows], lo[rows])
            hi[rows] = np.where(score < 0, x[rows], hi[rows])
            step = np.full(len(rows), np.inf)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                np.divide(score, slope, out=step, where=slope > 0)
                candidate = x[rows] + step
            bisect = (
                ~np.isfinite(candidate)
                | (candidate < lo[rows])
                | (candidate > hi[rows])
                | (np.abs(step) >= 0.5 * older_step[rows])
            )
            midpoint = 0.5 * lo[rows] + 0.5 * hi[rows]
            candidate = np.where(bisect, midpoint, candidate)
            narrow = hi[rows] - lo[rows] <= tol * (1.0 + np.abs(x[rows]))
            candidate = np.where(narrow, midpoint, candidate)
            fallbacks += int(np.count_nonzero(bisect & ~narrow))
            delta[rows] = np.abs(candidate - x[rows])
            older_step[rows] = previous_step[rows]
            previous_step[rows] = delta[rows]
            x[rows] = candidate
            done[rows] = narrow | (delta[rows] <= tol * (1.0 + np.abs(candidate)))

        rows = np.flatnonzero(flat & ~done)
        if rows.size:
            left = 0.5 * lo[rows] + 0.5 * zero_left[rows]
            right = 0.5 * zero_right[rows] + 0.5 * hi[rows]
            left_score = np.sum(weights * contrast.psi(windows[rows] - left[:, None]), axis=1)
            right_score = np.sum(weights * contrast.psi(windows[rows] - right[:, None]), axis=1)
            lo[rows] = np.where(left_score > 0, left, lo[rows])
            zero_left[rows] = np.where(left_score > 0, zero_left[rows], left)
            hi[rows] = np.where(right_score < 0, right, hi[rows])
            zero_right[rows] = np.where(right_score < 0, zero_right[rows], right)
            midpoint = 0.5 * zero_left[rows] + 0.5 * zero_right[rows]
            delta[rows] = np.abs(midpoint - x[rows])
            x[rows] = midpoint
            uncertainty = np.maximum(zero_left[rows] - lo[rows], hi[rows] - zero_right[rows])
            done[rows] = uncertainty <= tol * (1.0 + np.abs(midpoint))
            fallbacks += 2 * len(rows)

        if np.all(done):
            break

    score = np.sum(weights * contrast.psi(windows - x[:, None]), axis=1)
    return (
        x,
        iterations,
        float(np.max(delta)),
        int(np.count_nonzero(~done)),
        fallbacks,
        float(np.max(np.abs(score))),
    )
