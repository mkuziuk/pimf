"""Geometric window-size schedule for the IMF decomposition.

Faithful port of make_window_schedule from the IMF research notebooks:
odd, strictly decreasing sizes from about n/2 down to min_window_size,
shrinking by factor each stage.
"""

import numpy as np

from ._erf import SQRT_2


def odd_ceiling(value):
    """Smallest odd integer >= ceil(value), at least 1."""
    size = int(np.ceil(value))
    if size % 2 == 0:
        size += 1
    return max(1, size)


def nearest_odd(value):
    """Odd integer nearest to value (ties round down), at least 1."""
    rounded = int(np.round(value))
    if rounded % 2 == 1:
        return max(1, rounded)

    lower = max(1, rounded - 1)
    upper = rounded + 1
    if abs(value - lower) <= abs(upper - value):
        return lower
    return upper


def make_window_schedule(n, factor=SQRT_2, min_window_size=31):
    """Window sizes for a length-n signal: first = odd_ceiling(n / 2), then
    geometric shrink by factor, all odd, strictly decreasing, floored at
    min_window_size."""
    if n <= 0:
        raise ValueError("n must be positive")
    if factor <= 1:
        raise ValueError("factor must be larger than 1")

    first = odd_ceiling(n / 2)
    if first > n:
        first = n if n % 2 == 1 else n - 1

    min_size = nearest_odd(min_window_size)
    if min_size > first:
        return [first]

    sizes = [first]
    current = first

    while current > min_size:
        candidate = nearest_odd(current / factor)
        candidate = min(candidate, current - 2)
        if candidate % 2 == 0:
            candidate -= 1
        if candidate < min_size:
            candidate = min_size

        sizes.append(candidate)
        current = candidate

    return sizes
