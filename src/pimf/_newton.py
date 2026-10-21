"""Safeguarded, independently stopped local score roots."""

import numpy as np

from .contrasts import _SCORE, _SCORE_DELTA, SmoothAbs

_SIGN = np.int64(-(2**63))
# Grid tables padded with one zero-slope entry above the grid and, at index -1,
# one below it.
_PADDED_SCORE = np.r_[_SCORE, _SCORE[-1], _SCORE[0]]
_PADDED_DELTA = np.r_[_SCORE_DELTA, 0.0, 0.0]
for _table in (_PADDED_SCORE, _PADDED_DELTA):
    _table.setflags(write=False)
# Samples per evaluation block, so that its temporaries stay in cache.
_BLOCK_SAMPLES = 1 << 16


def _middle(lo, hi):
    """Bracket midpoint; IEEE-754 order when the bracket spans magnitudes."""
    # Ordered float64 ranks bound wide-bracket searches to at most 64 halvings.
    a, b = (np.where(v < 0, _SIGN - v, v) for v in (lo.view(np.int64), hi.view(np.int64)))
    m = (a >> 1) + (b >> 1) + (a & b & 1)
    ordered = np.where(m < 0, _SIGN - m, m).view(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        wide = hi - lo > 2.0 * (1.0 + np.minimum(np.abs(lo), np.abs(hi)))
        return np.where(wide | np.isinf(lo) | np.isinf(hi), ordered, 0.5 * lo + 0.5 * hi)


def _newton_fit_windows(windows, weights, contrast, max_iter, tol):
    """Return roots, max iterations/delta, unfinished count, fallbacks and score.

    Each row brackets a nonincreasing weighted score. Start at the weighted mean,
    or at a weighted median if the mean is not finite or its slope is zero.
    The rtsafe safeguard bisects unless Newton stays inside the bracket and
    reduces the step to less than half the step before last. Wide brackets
    bisect float64 ranks to bound the number of halvings across extreme
    outliers. Small steps need a positional bracket at candidate +- epsilon.
    For SmoothAbs lookup scores, probes whose sign the scores at the estimate or
    the candidate already fix are skipped. A row finishing at a small step
    reports the candidate's score.
    Flat zero intervals use the midpoint of their two boundaries, each located
    by bisection within the same cap.
    """
    # Slicing the positive span copies sliding windows much faster than a mask.
    first, last = np.flatnonzero(weights > 0)[[0, -1]]
    windows, weights = windows[:, first : last + 1], weights[first : last + 1]
    if not np.all(weights > 0):
        windows, weights = windows[:, weights > 0], weights[weights > 0]
    # A contiguous reduction axis keeps singleton chunks on the same sum path.
    windows = np.ascontiguousarray(windows, dtype=float)
    weights = weights / weights.sum()
    block = max(1, _BLOCK_SAMPLES // windows.shape[1])
    lookup = type(contrast) is SmoothAbs and contrast.lookup
    if lookup:
        # Dividing by an exact H / 256 gives SmoothAbs's 256 * (r / H) without rounding;
        # the zero-slope padding below needs a finite 8H.
        divisor = contrast.H / 256
        lookup = divisor * 256 == contrast.H and np.isfinite(8 * contrast.H)
    if lookup:
        with np.errstate(over="ignore"):
            slope_table = (256.0 * _PADDED_DELTA) / contrast.H
        # Probes evaluate up to two estimates per row.
        shape = (min(block, 2 * len(windows)), windows.shape[1])
        position, floor, index = np.empty(shape), np.empty(shape), np.empty(shape, dtype=np.intp)
        tiled_weights = np.tile(weights, (shape[0], 1))

    lo, hi = windows.min(axis=1), windows.max(axis=1)
    # Finite samples give NaN table positions only at NaN estimates.
    finite_samples = np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))

    def evaluate(rows, estimates, slope=False):
        """Weighted scores, and slopes if requested, of rows (None: all) at estimates."""
        scores = np.empty(len(estimates))
        slopes = np.empty(len(estimates)) if slope else None
        nan_positions = not finite_samples or np.isnan(estimates).any()
        for start in range(0, len(estimates), block):
            part = slice(start, start + block)
            estimate = estimates[part, None]
            if not lookup:
                sample = windows[part] if rows is None else windows[rows[part]]
                scores[part] = np.sum(weights * contrast.psi(sample - estimate), axis=1)
                if slope:
                    psi_prime = contrast.psi_prime(sample - estimate)
                    slopes[part] = np.sum(weights * psi_prime, axis=1)
                continue
            # Must match SmoothAbs.psi and psi_prime bit for bit; both share a position.
            n = len(estimate)
            p, f, i, w = position[:n], floor[:n], index[:n], tiled_weights[:n]
            if rows is None:
                np.subtract(windows[part], estimate, out=p)
            else:
                # Rows are valid indices; "clip" mode gathers without a buffer.
                np.take(windows, rows[part], axis=0, out=p, mode="clip")
                p -= estimate
            p /= divisor
            # As 8H is exact, |r| > 8H exactly when r / H rounds outside [-8, 8];
            # those positions land in the zero-slope padding cells.
            np.clip(p, -2049.0, 2049.0, out=p)
            p += 2048.0
            np.floor(p, out=f)
            p -= f
            if nan_positions:
                # NaNs need safe indices, but still propagate through the fractions.
                np.fmin(f, 4097.0, out=f)
            np.copyto(i, f, casting="unsafe")
            p *= _PADDED_DELTA[i]
            p += _PADDED_SCORE[i]
            p *= w
            scores[part] = p.sum(axis=1)
            if slope:
                g = slope_table[i]
                g *= w
                slopes[part] = g.sum(axis=1)
        return scores, slopes

    def weighted_median(rows):
        order = np.argsort(windows[rows], axis=1)
        middle = np.argmax(np.cumsum(weights[order], axis=1) >= 0.5, axis=1)
        return windows[rows, order[np.arange(len(rows)), middle]]

    with np.errstate(over="ignore", invalid="ignore"):
        x = np.clip(np.sum(windows * weights, axis=1), lo, hi)
        previous_step = hi - lo
    # Infinite samples make the mean infinite or NaN; NaN samples stay unfinished.
    reset_rows = np.flatnonzero(~np.isfinite(x) & ~np.isnan(lo) & ~np.isnan(hi))
    x[reset_rows] = weighted_median(reset_rows)
    older_step = previous_step.copy()
    delta = np.zeros(len(x))
    done = lo == hi
    flat = np.zeros(len(x), dtype=bool)
    zero_left, zero_right = x.copy(), x.copy()
    known, final_score = np.zeros(len(x), dtype=bool), np.zeros(len(x))
    fallbacks = 0
    iterations = 0

    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(max_iter):
            if np.all(done):
                break
            iterations += 1
            rows = np.flatnonzero(~done & ~flat)
            if rows.size:
                xr = x[rows]
                score, slope = evaluate(None if rows.size == len(x) else rows, xr, slope=True)
                keep = np.isfinite(score) & np.isfinite(xr)
                if not np.all(keep):
                    delta[rows[~keep]] = np.nan
                    rows, xr, score, slope = rows[keep], xr[keep], score[keep], slope[keep]
                keep = score != 0
                if not np.all(keep):
                    zeros, xz = rows[~keep], xr[~keep]
                    epsilon = tol * (1.0 + np.abs(xz))
                    left_score = evaluate(zeros, xz - epsilon)[0]
                    right_score = evaluate(zeros, xz + epsilon)[0]
                    finite = np.isfinite(left_score) & np.isfinite(right_score)
                    plateau = (
                        finite & ((left_score == 0) | (right_score == 0)) & (lo[zeros] < hi[zeros])
                    )
                    flat[zeros[plateau]] = True
                    zero_left[zeros[plateau]] = xz[plateau]
                    zero_right[zeros[plateau]] = xz[plateau]
                    done[zeros] = known[zeros] = finite & ~plateau
                    delta[zeros] = np.where(finite, 0.0, np.nan)
                    rows, xr, score, slope = rows[keep], xr[keep], score[keep], slope[keep]

                if iterations == 1 and np.any(slope == 0):
                    stalled = slope == 0
                    xr[stalled] = weighted_median(rows[stalled])
                    score[stalled], slope[stalled] = evaluate(rows[stalled], xr[stalled], True)
                lower = np.where(score > 0, xr, lo[rows])
                upper = np.where(score < 0, xr, hi[rows])
                step = np.full(len(rows), np.inf)
                np.divide(score, slope, out=step, where=(slope > 0) & np.isfinite(slope))
                candidate = xr + step
                bisect = (
                    ~np.isfinite(candidate)
                    | (candidate < lower)
                    | (candidate > upper)
                    | (np.abs(step) >= 0.5 * older_step[rows])
                )
                if np.any(bisect):
                    candidate[bisect] = _middle(lower[bisect], upper[bisect])
                narrow = upper - lower <= tol * (1.0 + np.maximum(np.abs(lower), np.abs(upper)))
                narrow &= np.isfinite(lower) & np.isfinite(upper)
                if np.any(narrow):
                    candidate[narrow] = 0.5 * lower[narrow] + 0.5 * upper[narrow]
                fallbacks += int(np.count_nonzero(bisect & ~narrow))
                finished = narrow.copy()
                small = (np.abs(candidate - xr) <= tol * (1.0 + np.abs(candidate))) & ~narrow
                if np.any(small):
                    probe_rows, xs, ss, cs = rows[small], xr[small], score[small], candidate[small]
                    lows, highs = lower[small], upper[small]
                    epsilon = tol * (1.0 + np.abs(cs))
                    left, right = cs - epsilon, cs + epsilon
                    at_candidate = evaluate(probe_rows, cs)[0]
                    # Lookup scores are nonincreasing as computed (the table differences
                    # are exact), so score(x) or score(candidate) fixes the sign of a
                    # finite probe beyond them. Other scores can wiggle by rounding.
                    left_score = np.where(at_candidate > 0, at_candidate, ss)
                    right_score = np.where(at_candidate < 0, at_candidate, ss)
                    known_left = (at_candidate > 0) | ((ss > 0) & (left <= xs))
                    known_right = (at_candidate < 0) | ((ss < 0) & (right >= xs))
                    probe_left = ~(lookup & known_left & np.isfinite(left))
                    probe_right = ~(lookup & known_right & np.isfinite(right))
                    probes = evaluate(
                        np.concatenate((probe_rows[probe_left], probe_rows[probe_right])),
                        np.concatenate((left[probe_left], right[probe_right])),
                    )[0]
                    split = np.count_nonzero(probe_left)
                    left_score[probe_left], right_score[probe_right] = (
                        probes[:split],
                        probes[split:],
                    )
                    finite = np.isfinite(left_score) & np.isfinite(right_score)
                    zero_hit = finite & ((left_score == 0) | (right_score == 0))
                    bracketed = finite & (left_score >= 0) & (right_score <= 0)
                    end = bracketed & ~zero_hit
                    lows = np.where(
                        right_score > 0,
                        np.maximum(lows, right),
                        np.where(left_score > 0, np.maximum(lows, left), lows),
                    )
                    highs = np.where(
                        left_score < 0,
                        np.minimum(highs, left),
                        np.where(right_score < 0, np.minimum(highs, right), highs),
                    )
                    next_x = np.where(left_score == 0, left, right)
                    retry = ~bracketed & ~zero_hit
                    if np.any(retry):
                        next_x[retry] = _middle(lows[retry], highs[retry])
                    lower[small], upper[small] = lows, highs
                    candidate[small] = np.where(end, cs, next_x)
                    finished[small] = end
                    known[probe_rows[end]] = True
                    final_score[probe_rows[end]] = at_candidate[end]
                    fallbacks += int(np.count_nonzero(retry))
                moved = np.abs(candidate - xr)
                delta[rows] = moved
                older_step[rows] = previous_step[rows]
                previous_step[rows] = moved
                x[rows], lo[rows], hi[rows], done[rows] = candidate, lower, upper, finished

            rows = np.flatnonzero(flat & ~done)
            if rows.size:
                left = _middle(lo[rows], zero_left[rows])
                right = _middle(zero_right[rows], hi[rows])
                left_score, right_score = evaluate(rows, left)[0], evaluate(rows, right)[0]
                finite = np.isfinite(left_score) & np.isfinite(right_score)
                delta[rows[~finite]] = np.nan
                rows, left, right = rows[finite], left[finite], right[finite]
                left_score, right_score = left_score[finite], right_score[finite]
                lo[rows] = np.where(left_score > 0, left, lo[rows])
                zero_left[rows] = np.where(left_score > 0, zero_left[rows], left)
                hi[rows] = np.where(right_score < 0, right, hi[rows])
                zero_right[rows] = np.where(right_score < 0, zero_right[rows], right)
                midpoint = 0.5 * zero_left[rows] + 0.5 * zero_right[rows]
                delta[rows] = np.abs(midpoint - x[rows])
                x[rows] = midpoint
                uncertainty = np.maximum(zero_left[rows] - lo[rows], hi[rows] - zero_right[rows])
                done[rows] = uncertainty <= tol * (
                    1.0 + np.maximum(np.abs(zero_left[rows]), np.abs(zero_right[rows]))
                )
                fallbacks += 2 * len(rows)

        rows = np.flatnonzero(~known)
        final_score[rows] = evaluate(None if rows.size == len(x) else rows, x[rows])[0]
    finite = np.isfinite(x) & np.isfinite(final_score)
    done &= finite
    delta[~finite] = np.nan
    return (
        x,
        iterations,
        float(np.max(delta)),
        int(np.count_nonzero(~done)),
        fallbacks,
        float(np.max(np.abs(final_score))),
    )
