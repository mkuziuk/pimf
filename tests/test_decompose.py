import numpy as np
import pytest

from pimf import Quadratic, imf, linear_imf, robust_imf
from pimf.decompose import _gd_fit_windows
from pimf.kernels import SquaredTriangle


class QuadraticViaGD(Quadratic):
    """Quadratic contrast with the closed form disabled, to exercise the GD path."""

    solve = None


@pytest.fixture
def noisy_signal():
    rng = np.random.default_rng(42)
    t = np.linspace(0.0, 1.0, 512)
    return np.sin(2 * np.pi * t) + 0.3 * np.sin(14 * np.pi * t) + rng.normal(0.0, 0.2, 512)


def test_reconstruction_linear(noisy_signal):
    result = linear_imf(noisy_signal)
    assert np.max(np.abs(result.reconstruction - noisy_signal)) < 1e-10


def test_reconstruction_robust(noisy_signal):
    result = robust_imf(noisy_signal, h=0.4)
    assert np.max(np.abs(result.reconstruction - noisy_signal)) < 1e-10


def test_shapes_and_stage_info(noisy_signal):
    result = robust_imf(noisy_signal, h=0.4, window_sizes=[151, 75, 31])
    assert result.imfs.shape == (3, len(noisy_signal))
    assert result.residual.shape == (len(noisy_signal),)
    assert result.window_sizes == [151, 75, 31]
    assert np.all(np.isfinite(result.imfs))
    assert np.all(np.isfinite(result.residual))
    for k, info in enumerate(result.stages, start=1):
        assert info.stage == k
        assert 1 <= info.iterations <= 60
        assert np.isfinite(info.final_max_delta)


def test_linear_stage_info(noisy_signal):
    result = linear_imf(noisy_signal, window_sizes=[63, 31])
    for info in result.stages:
        assert info.iterations == 1
        assert np.isnan(info.final_max_delta)


def test_linearity_of_linear_imf():
    rng = np.random.default_rng(7)
    x = rng.normal(size=256)
    y = rng.normal(size=256)
    left = linear_imf(y).imfs - linear_imf(x).imfs
    right = linear_imf(y - x).imfs
    assert np.max(np.abs(left - right)) < 1e-12


def test_gd_quadratic_matches_closed_form(noisy_signal):
    # tol=1e-12: the GD stopping rule leaves ~tol-scale error, so the default
    # 1e-6 would only match the closed form to ~1e-7.
    closed = linear_imf(noisy_signal, window_sizes=[63, 31])
    via_gd = imf(noisy_signal, window_sizes=[63, 31], contrast=QuadraticViaGD(), tol=1e-12)
    assert np.max(np.abs(closed.imfs - via_gd.imfs)) < 1e-9
    assert np.max(np.abs(closed.residual - via_gd.residual)) < 1e-9


def test_robust_fit_symmetric_window_returns_center():
    from pimf import SmoothAbs

    windows = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]])
    weights = SquaredTriangle().weights(5)
    x, _, _ = _gd_fit_windows(windows, weights, SmoothAbs(0.5), max_iter=60, tol=1e-6)
    assert x[0] == 2.0


def test_stage_one_noise_sd():
    # Exact per-point SD under wrap is sigma * sqrt(sum w^2); audit fixture 0.0240001.
    sigma = 0.4
    weights = SquaredTriangle().weights(501)
    exact_sd = sigma * np.sqrt(np.sum(weights**2))
    assert abs(exact_sd - 0.0240001) < 1e-7

    rng = np.random.default_rng(123)
    smoothed = []
    for _ in range(50):
        noise = rng.normal(0.0, sigma, 1000)
        smoothed.append(linear_imf(noise, window_sizes=[501]).imfs[0])
    empirical_sd = np.std(np.concatenate(smoothed))
    assert abs(empirical_sd - exact_sd) / exact_sd < 0.1


def test_invalid_inputs_raise():
    y = np.zeros(64)
    with pytest.raises(ValueError):
        imf(y, window_sizes=[10])
    with pytest.raises(ValueError):
        imf(np.zeros((4, 4)))
    with pytest.raises(ValueError):
        imf(np.array([]))


@pytest.mark.parametrize("boundary", ["wrap", "reflect", "edge"])
def test_boundary_reaches_the_padding(boundary):
    # Reconstruction telescopes for any smoother output, so it cannot detect
    # boundary plumbing bugs. Instead pin the stage-1 edge value against the
    # weighted window computed from independently padded data.
    y = np.linspace(0.0, 1.0, 64)  # a ramp: maximally asymmetric at the edges
    component = linear_imf(y, window_sizes=[31], boundary=boundary).imfs[0]
    weights = SquaredTriangle().weights(31)
    padded = np.pad(y, 15, mode=boundary)
    assert component[0] == pytest.approx(weights @ padded[:31], abs=1e-15)
    assert component[-1] == pytest.approx(weights @ padded[-31:], abs=1e-15)


def test_boundaries_differ_at_the_edges():
    y = np.linspace(0.0, 1.0, 64)
    edge_values = {
        boundary: linear_imf(y, window_sizes=[31], boundary=boundary).imfs[0][:5]
        for boundary in ("wrap", "reflect", "edge")
    }
    assert not np.allclose(edge_values["wrap"], edge_values["reflect"], atol=1e-6)
    assert not np.allclose(edge_values["wrap"], edge_values["edge"], atol=1e-6)
    assert not np.allclose(edge_values["reflect"], edge_values["edge"], atol=1e-6)


def test_max_iter_is_respected(noisy_signal):
    result = robust_imf(noisy_signal, h=0.4, window_sizes=[63, 31], max_iter=1)
    for info in result.stages:
        assert info.iterations == 1


def test_custom_kernel_flows_through_imf(noisy_signal):
    from numpy.lib.stride_tricks import sliding_window_view

    from pimf import Kernel

    class Uniform(Kernel):
        def profile(self, u):
            return np.ones_like(u)

    result = linear_imf(noisy_signal, window_sizes=[7], kernel=Uniform())
    weights = Uniform().weights(7)
    expected = sliding_window_view(np.pad(noisy_signal, 3, mode="wrap"), 7) @ weights
    assert np.array_equal(result.imfs[0], expected)

    default = linear_imf(noisy_signal, window_sizes=[7]).imfs[0]
    assert not np.allclose(result.imfs[0], default, atol=1e-6)
