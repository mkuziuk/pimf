import numpy as np
import pytest

from pimf import Quadratic, SmoothAbs
from pimf._erf import SQRT_2_OVER_PI


def test_smooth_abs_requires_positive_h():
    with pytest.raises(ValueError):
        SmoothAbs(0.0)
    with pytest.raises(ValueError):
        SmoothAbs(-1.0)


def test_smooth_abs_score_is_bounded():
    contrast = SmoothAbs(0.4)
    r = np.linspace(-1e6, 1e6, 10001)
    assert np.all(np.abs(contrast.psi(r)) <= 1.0)


def test_smooth_abs_symmetry():
    contrast = SmoothAbs(0.4)
    r = np.linspace(0.0, 10.0, 1001)
    assert np.array_equal(contrast.psi(-r), -contrast.psi(r))
    assert np.allclose(contrast(-r), contrast(r), atol=1e-15)


def test_smooth_abs_at_zero():
    h = 0.7
    assert SmoothAbs(h)(0.0) == SQRT_2_OVER_PI * h


def test_smooth_abs_approaches_abs_in_tails():
    h = 0.4
    r = 50.0 * h
    assert abs(SmoothAbs(h)(r) - r) / r < 1e-6


def test_smooth_abs_curvature():
    h = 0.4
    assert SmoothAbs(h).curvature() == SQRT_2_OVER_PI / h


def test_quadratic():
    contrast = Quadratic()
    r = np.linspace(-3.0, 3.0, 101)
    assert np.array_equal(contrast.psi(r), r)
    assert contrast.curvature() == 1.0
    assert np.array_equal(contrast(r), 0.5 * r**2)


def test_quadratic_solve_is_weighted_mean():
    rng = np.random.default_rng(0)
    windows = rng.normal(size=(7, 5))
    weights = np.array([0.0, 0.25, 0.5, 0.25, 0.0])
    assert np.array_equal(Quadratic().solve(windows, weights), windows @ weights)


@pytest.mark.parametrize("contrast", [Quadratic(), SmoothAbs(0.5)])
def test_psi_is_derivative_of_rho(contrast):
    # Template consistency check for any contrast implementation.
    r = np.linspace(-3.0, 3.0, 61)
    eps = 1e-6
    finite_difference = (contrast(r + eps) - contrast(r - eps)) / (2 * eps)
    assert np.max(np.abs(finite_difference - contrast.psi(r))) < 1e-4
