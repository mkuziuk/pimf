import math

import numpy as np

from pimf._erf import erf_approx


def test_matches_math_erf():
    x = np.linspace(-6.0, 6.0, 4001)
    reference = np.array([math.erf(v) for v in x])
    assert np.max(np.abs(erf_approx(x) - reference)) < 1.5e-7


def test_zero_is_exact():
    assert erf_approx(0.0) == 0.0


def test_odd_symmetry_is_exact():
    x = np.linspace(0.0, 8.0, 1001)
    assert np.array_equal(erf_approx(-x), -erf_approx(x))


def test_saturates_in_tails():
    assert erf_approx(1e6) == 1.0
    assert erf_approx(-1e6) == -1.0
