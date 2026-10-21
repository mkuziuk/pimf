"""Safeguarded, independently stopped local score roots."""

import numpy as np


def _newton_fit_windows(windows, weights, contrast, max_iter, tol):
    """Return roots, max iterations/delta, unfinished count, fallbacks and score.

    Each row brackets a nonincreasing weighted score. Start at the weighted mean,
    restarting at a sample median if its slope is zero. The rtsafe safeguard bisects
    unless Newton stays inside the bracket and reduces the step to less than
    half the step before last. Before rejecting an interior step, expand local
    score probes to tighten a potentially outlier-wide bracket. Small steps
    need a positional bracket.
    Flat zero intervals use the midpoint of their two boundaries, each located
    by bisection within the same cap.
    """
    positive = weights > 0
    # A contiguous reduction axis keeps singleton chunks on the same sum path.
    windows = np.ascontiguousarray(windows[:, positive], dtype=float)
    weights = weights[positive] / weights[positive].sum()

    def score_at(rows, estimates):
        with np.errstate(over="ignore", invalid="ignore"):
            residuals = windows[rows] - estimates[:, None]
            return np.sum(weights * contrast.psi(residuals), axis=1)

    lo, hi = windows.min(axis=1), windows.max(axis=1)
    with np.errstate(over="ignore", invalid="ignore"):
        x = np.clip(np.sum(windows * weights, axis=1), lo, hi)
        previous_step = hi - lo
    older_step = previous_step.copy()
    delta = np.zeros(len(x))
    done = lo == hi
    flat = np.zeros(len(x), dtype=bool)
    zero_left, zero_right = x.copy(), x.copy()
    fallbacks = 0
    iterations = 0

    for _ in range(max_iter):
        if np.all(done):
            break
        iterations += 1
        rows = np.flatnonzero(~done & ~flat)
        if rows.size:
            score = score_at(rows, x[rows])
            finite = np.isfinite(score) & np.isfinite(x[rows])
            delta[rows[~finite]] = np.nan
            rows, score = rows[finite], score[finite]
            zeros = rows[score == 0]
            if zeros.size:
                epsilon = tol * (1.0 + np.abs(x[zeros]))
                with np.errstate(over="ignore", invalid="ignore"):
                    left, right = x[zeros] - epsilon, x[zeros] + epsilon
                left_score = score_at(zeros, left)
                right_score = score_at(zeros, right)
                finite = np.isfinite(left_score) & np.isfinite(right_score)
                plateau = (
                    finite & ((left_score == 0) | (right_score == 0)) & (lo[zeros] < hi[zeros])
                )
                flat[zeros[plateau]] = True
                zero_left[zeros[plateau]] = x[zeros[plateau]]
                zero_right[zeros[plateau]] = x[zeros[plateau]]
                done[zeros] = finite & ~plateau
                delta[zeros] = np.where(finite, 0.0, np.nan)

            moving = score != 0
            rows, score = rows[moving], score[moving]
            with np.errstate(over="ignore", invalid="ignore"):
                residuals = windows[rows] - x[rows, None]
                slope = np.sum(weights * contrast.psi_prime(residuals), axis=1)
            if iterations == 1 and np.any(slope == 0):
                stalled = slope == 0
                reset_rows = rows[stalled]
                middle = windows.shape[1] // 2
                x[reset_rows] = np.partition(windows[reset_rows], middle, axis=1)[:, middle]
                score[stalled] = score_at(reset_rows, x[reset_rows])
                with np.errstate(over="ignore", invalid="ignore"):
                    residuals = windows[reset_rows] - x[reset_rows, None]
                    slope[stalled] = np.sum(weights * contrast.psi_prime(residuals), axis=1)
            lo[rows] = np.where(score > 0, x[rows], lo[rows])
            hi[rows] = np.where(score < 0, x[rows], hi[rows])
            step = np.full(len(rows), np.inf)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                np.divide(score, slope, out=step, where=(slope > 0) & np.isfinite(slope))
                candidate = x[rows] + step
            bisect = (
                ~np.isfinite(candidate)
                | (candidate < lo[rows])
                | (candidate > hi[rows])
                | (np.abs(step) >= 0.5 * older_step[rows])
            )
            searching = bisect & np.isfinite(step) & (candidate > lo[rows]) & (candidate < hi[rows])
            probe_indices = np.flatnonzero(searching)
            probe_rows, radius = rows[probe_indices], np.abs(step[probe_indices])
            direction = np.sign(score[searching])
            for _ in range(max_iter):
                if not probe_rows.size:
                    break
                with np.errstate(over="ignore", invalid="ignore"):
                    probe = np.clip(
                        x[probe_rows] + direction * radius, lo[probe_rows], hi[probe_rows]
                    )
                probe_score = score_at(probe_rows, probe)
                lo[probe_rows] = np.where(probe_score > 0, probe, lo[probe_rows])
                hi[probe_rows] = np.where(probe_score < 0, probe, hi[probe_rows])
                zeros = probe_score == 0
                candidate[probe_indices[zeros]] = probe[zeros]
                bisect[probe_indices[zeros]] = False
                expand = np.isfinite(probe_score) & (direction * probe_score > 0)
                expand &= np.where(direction > 0, probe < hi[probe_rows], probe > lo[probe_rows])
                probe_indices = probe_indices[expand]
                probe_rows, direction = probe_rows[expand], direction[expand]
                with np.errstate(over="ignore"):
                    radius = 2 * radius[expand]
            midpoint = 0.5 * lo[rows] + 0.5 * hi[rows]
            candidate = np.where(bisect, midpoint, candidate)
            with np.errstate(over="ignore", invalid="ignore"):
                narrow = hi[rows] - lo[rows] <= tol * (
                    1.0 + np.maximum(np.abs(lo[rows]), np.abs(hi[rows]))
                )
            candidate = np.where(narrow, midpoint, candidate)
            fallbacks += int(np.count_nonzero(bisect & ~narrow))
            finished = narrow.copy()
            with np.errstate(over="ignore", invalid="ignore"):
                small = (np.abs(candidate - x[rows]) <= tol * (1.0 + np.abs(candidate))) & ~narrow
            if np.any(small):
                probe_rows = rows[small]
                epsilon = tol * (1.0 + np.abs(candidate[small]))
                with np.errstate(over="ignore", invalid="ignore"):
                    left, right = candidate[small] - epsilon, candidate[small] + epsilon
                left_score, right_score = score_at(probe_rows, left), score_at(probe_rows, right)
                finite = np.isfinite(left_score) & np.isfinite(right_score)
                zero_hit = finite & ((left_score == 0) | (right_score == 0))
                bracketed = finite & (left_score >= 0) & (right_score <= 0)
                finished[small] = bracketed & ~zero_hit
                lo[probe_rows] = np.where(
                    left_score > 0, np.maximum(lo[probe_rows], left), lo[probe_rows]
                )
                lo[probe_rows] = np.where(
                    right_score > 0, np.maximum(lo[probe_rows], right), lo[probe_rows]
                )
                hi[probe_rows] = np.where(
                    left_score < 0, np.minimum(hi[probe_rows], left), hi[probe_rows]
                )
                hi[probe_rows] = np.where(
                    right_score < 0, np.minimum(hi[probe_rows], right), hi[probe_rows]
                )
                next_x = np.where(left_score == 0, left, right)
                retry = ~bracketed & ~zero_hit
                next_x = np.where(retry, 0.5 * lo[probe_rows] + 0.5 * hi[probe_rows], next_x)
                candidate[small] = np.where(finished[small], candidate[small], next_x)
                fallbacks += int(np.count_nonzero(retry))
            with np.errstate(over="ignore", invalid="ignore"):
                delta[rows] = np.abs(candidate - x[rows])
            older_step[rows] = previous_step[rows]
            previous_step[rows] = delta[rows]
            x[rows] = candidate
            done[rows] = finished

        rows = np.flatnonzero(flat & ~done)
        if rows.size:
            left = 0.5 * lo[rows] + 0.5 * zero_left[rows]
            right = 0.5 * zero_right[rows] + 0.5 * hi[rows]
            left_score, right_score = score_at(rows, left), score_at(rows, right)
            finite = np.isfinite(left_score) & np.isfinite(right_score)
            delta[rows[~finite]] = np.nan
            rows, left, right = rows[finite], left[finite], right[finite]
            left_score, right_score = left_score[finite], right_score[finite]
            lo[rows] = np.where(left_score > 0, left, lo[rows])
            zero_left[rows] = np.where(left_score > 0, zero_left[rows], left)
            hi[rows] = np.where(right_score < 0, right, hi[rows])
            zero_right[rows] = np.where(right_score < 0, zero_right[rows], right)
            midpoint = 0.5 * zero_left[rows] + 0.5 * zero_right[rows]
            with np.errstate(over="ignore", invalid="ignore"):
                delta[rows] = np.abs(midpoint - x[rows])
            x[rows] = midpoint
            with np.errstate(over="ignore", invalid="ignore"):
                uncertainty = np.maximum(zero_left[rows] - lo[rows], hi[rows] - zero_right[rows])
            done[rows] = uncertainty <= tol * (
                1.0 + np.maximum(np.abs(zero_left[rows]), np.abs(zero_right[rows]))
            )
            fallbacks += 2 * len(rows)

        if np.all(done):
            break

    score = score_at(np.arange(len(x)), x)
    finite = np.isfinite(x) & np.isfinite(score)
    done &= finite
    delta[~finite] = np.nan
    return (
        x,
        iterations,
        float(np.max(delta)),
        int(np.count_nonzero(~done)),
        fallbacks,
        float(np.max(np.abs(score))),
    )
