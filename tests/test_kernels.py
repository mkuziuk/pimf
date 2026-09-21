import numpy as np
import pytest

from pimf import Kernel, SquaredTriangle
from pimf.kernels import (
    Epanechnikov,
    Triangle,
    Uniform,
    epanechnikov,
    squared_triangle,
    triangle,
    uniform,
)


def research_epanechnikov_weights(window_size):
    """Inlined weight computation from the research notebooks (bit-parity target)."""
    radius = window_size // 2
    offsets = np.arange(-radius, radius + 1)
    u = offsets / radius
    weights = 0.75 * np.maximum(0, 1 - np.abs(u)) ** 2
    return weights / weights.sum()


@pytest.mark.parametrize("window_size", [3, 31, 501])
def test_weight_invariants(window_size):
    weights = SquaredTriangle().weights(window_size)
    assert len(weights) == window_size
    assert np.all(weights >= 0)
    assert np.isclose(weights.sum(), 1.0, atol=1e-15)
    assert weights[0] == 0.0
    assert weights[-1] == 0.0
    assert np.array_equal(weights, weights[::-1])


@pytest.mark.parametrize("window_size", [3, 13, 31, 151, 501])
def test_bit_parity_with_research_code(window_size):
    assert np.array_equal(
        SquaredTriangle().weights(window_size), research_epanechnikov_weights(window_size)
    )


def test_sum_of_squares_fixture():
    # Validated in the research audit: drives the exact stage-1 noise SD.
    assert abs(np.sum(SquaredTriangle().weights(501) ** 2) - 0.0036000384) < 1e-10


def test_even_window_raises():
    with pytest.raises(ValueError):
        SquaredTriangle().weights(10)


def test_window_of_one():
    assert np.array_equal(SquaredTriangle().weights(1), np.array([1.0]))


def test_kernel_is_callable():
    kernel = SquaredTriangle()
    assert np.array_equal(kernel(31), kernel.weights(31))


def test_custom_kernel_one_liner():
    class Triangle(Kernel):
        def profile(self, u):
            return 1.0 - np.abs(u)

    weights = Triangle().weights(5)
    assert np.all(weights >= 0)
    assert np.isclose(weights.sum(), 1.0)


def test_negative_profile_raises():
    class Bad(Kernel):
        def profile(self, u):
            return u  # negative on [-1, 0)

    with pytest.raises(ValueError):
        Bad().weights(5)


@pytest.mark.parametrize("window_size", [0, -1, 2, 3.5, np.nan, np.inf, True, "5"])
def test_invalid_window_size_raises(window_size):
    with pytest.raises(ValueError, match="positive odd integer"):
        squared_triangle.weights(window_size)


@pytest.mark.parametrize("window_size", [5, 5.0, np.int64(5)])
def test_integral_window_size(window_size):
    assert np.array_equal(squared_triangle.weights(window_size), research_epanechnikov_weights(5))


@pytest.mark.parametrize("bandwidth", [0, -1, np.nan, np.inf, True, "2"])
def test_invalid_bandwidth_raises(bandwidth):
    with pytest.raises(ValueError, match="bandwidth"):
        squared_triangle.weights(5, bandwidth=bandwidth)


@pytest.mark.parametrize("profile", [np.nan, np.inf, -0.1, 0.0])
def test_invalid_profile_values_raise(profile):
    class Bad(Kernel):
        def profile(self, u):
            return np.full_like(u, profile)

    with pytest.raises(ValueError):
        Bad().weights(5)


def test_profile_shape_must_match_offsets():
    class Bad(Kernel):
        def profile(self, u):
            return np.ones((len(u), 1))

    with pytest.raises(ValueError, match="matching u"):
        Bad().weights(5)


@pytest.mark.parametrize(
    ("kernel", "kernel_type"),
    [
        (squared_triangle, SquaredTriangle),
        (epanechnikov, Epanechnikov),
        (triangle, Triangle),
        (uniform, Uniform),
    ],
)
def test_predefined_kernels(kernel, kernel_type):
    assert isinstance(kernel, kernel_type)
    assert vars(kernel) == {}
    assert np.array_equal(kernel.weights(5), kernel_type().weights(5))
    assert np.array_equal(kernel.weights(1), [1.0])
    assert np.array_equal(kernel.weights(1, bandwidth=0.25), [1.0])


@pytest.mark.parametrize("kernel", [squared_triangle, epanechnikov, triangle, uniform])
def test_fractional_bandwidth_preserves_sample_radius(kernel):
    u = np.arange(-3, 4) / 2.5
    expected = np.asarray(kernel.profile(u), dtype=float)
    expected[np.abs(u) > 1] = 0.0
    expected /= expected.sum()
    assert np.array_equal(kernel.weights(7, bandwidth=2.5), expected)
    assert np.array_equal(kernel(7, bandwidth=2.5), expected)


def test_custom_profile_only_receives_supported_offsets():
    class SupportedTriangle(Kernel):
        def profile(self, u):
            assert np.all(np.abs(u) <= 1)
            return 1.0 - np.abs(u)

    assert np.array_equal(
        SupportedTriangle().weights(7, bandwidth=2.5), triangle.weights(7, bandwidth=2.5)
    )


def test_predefined_profile_formulas():
    u = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    np.testing.assert_allclose(
        epanechnikov.profile(u), [0, 0, 0.5625, 0.75, 0.5625, 0, 0], atol=0, rtol=1e-15
    )
    np.testing.assert_allclose(triangle.profile(u), [0, 0, 0.5, 1, 0.5, 0, 0], atol=0, rtol=1e-15)
    np.testing.assert_allclose(
        uniform.profile(u), [0, 0.5, 0.5, 0.5, 0.5, 0.5, 0], atol=0, rtol=1e-15
    )
