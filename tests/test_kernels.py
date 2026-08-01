import numpy as np
import pytest

from pimf import Kernel, SquaredTriangle


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
