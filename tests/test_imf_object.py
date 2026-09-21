import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view

from pimf import Kernel, Quadratic, SmoothAbs
from pimf.decompose import IMF
from pimf.kernels import epanechnikov, squared_triangle, triangle, uniform


def test_uniform_retains_endpoints_at_a_rounded_grid_boundary():
    result = IMF(kernel=uniform).decompose(np.zeros(64), k_max=3)
    assert result.bandwidths[2] == pytest.approx(0.125, rel=0, abs=1e-15)
    assert result.window_sizes[2] == 17
    weights = uniform.weights(17, bandwidth=result.bandwidths[2] * 64)
    np.testing.assert_allclose(weights, np.full(17, 1 / 17), rtol=0, atol=1e-15)


def test_boundary_snapping_does_not_restart_the_geometric_schedule():
    result = IMF(kernel=uniform).decompose(np.zeros(64), a=1 + 1e-11, k_max=3)
    assert all(
        left > right for left, right in zip(result.bandwidths, result.bandwidths[1:], strict=False)
    )


def test_default_fractional_schedule():
    result = IMF().decompose(np.zeros(1000))
    assert result.window_sizes == [501, 355, 251, 179, 127, 91, 65, 47]
    np.testing.assert_allclose(
        result.bandwidths, 0.25 / np.sqrt(2.0) ** np.arange(8), rtol=1e-15, atol=0
    )
    assert result.imfs.shape == (8, 1000)


def test_fractional_bandwidth_controls_weights_without_integer_rounding():
    y = np.arange(23.0) ** 2
    result = IMF().decompose(y, h1=0.2, a=2, k_max=2)
    residual = y.copy()
    for component, radius in zip(result.imfs, [4.6, 2.3], strict=True):
        support = int(np.ceil(radius))
        u = np.arange(-support, support + 1) / radius
        weights = 0.75 * np.maximum(0.0, 1.0 - np.abs(u)) ** 2
        weights /= weights.sum()
        windows = sliding_window_view(np.pad(residual, support, mode="wrap"), len(weights))
        expected = windows @ weights
        np.testing.assert_allclose(component, expected, rtol=1e-14, atol=1e-13)
        residual -= expected
    np.testing.assert_allclose(result.residual, residual, rtol=1e-14, atol=1e-13)


def test_first_bandwidth_is_half_width_of_unit_period():
    default = IMF().decompose(np.zeros(100), k_max=1)
    narrower = IMF().decompose(np.zeros(100), h1=0.1, k_max=1)
    assert default.window_sizes == [51]
    assert narrower.window_sizes == [21]
    assert narrower.bandwidths == [0.1]


def test_stage_limit_and_inclusive_minimum_bandwidth():
    y = np.zeros(128)
    limited = IMF().decompose(y, a=2, k_max=2, h_min=0.03125)
    threshold = IMF().decompose(y, a=2, k_max=None, h_min=0.0625)
    assert limited.bandwidths == [0.25, 0.125]
    assert threshold.bandwidths == [0.25, 0.125, 0.0625]
    assert IMF().decompose(y, h_min=0.25).bandwidths == [0.25]


def test_inclusive_minimum_survives_geometric_roundoff():
    result = IMF().decompose(np.zeros(64), h_min=0.125)
    assert len(result.bandwidths) == 3
    assert result.bandwidths[-1] == pytest.approx(0.125, abs=1e-15)


@pytest.mark.parametrize("contrast", [Quadratic(), SmoothAbs(H=0.2)])
def test_explicit_windows_define_the_schedule(contrast):
    y = np.random.default_rng(777).normal(size=64)
    windows = [31, 15, 7]
    result = IMF(contrast).decompose(y, window_sizes=windows)
    assert result.window_sizes == windows
    assert result.bandwidths == [15 / 64, 7 / 64, 3 / 64]
    assert result.imfs.shape == (len(windows), len(y))


@pytest.mark.parametrize("kernel", [squared_triangle, epanechnikov, triangle, uniform])
def test_predefined_kernel_instances_control_the_fit(kernel):
    y = np.arange(16.0) ** 2
    result = IMF(kernel=kernel).decompose(y, window_sizes=[7])
    expected = sliding_window_view(np.pad(y, 3, mode="wrap"), 7) @ kernel.weights(7)
    assert np.array_equal(result.imfs[0], expected)


def test_custom_profile_subclass_supports_fractional_bandwidths():
    class CustomTriangle(Kernel):
        def profile(self, u):
            return 1.0 - np.abs(u)

    y = np.arange(19.0) ** 2
    custom = IMF(kernel=CustomTriangle()).decompose(y, k_max=2)
    expected = IMF(kernel=triangle).decompose(y, k_max=2)
    assert np.array_equal(custom.imfs, expected.imfs)


def test_configuration_reuse_returns_independent_results():
    model = IMF(SmoothAbs(H=0.2), kernel=triangle)
    y = np.random.default_rng(12).normal(size=32)
    original = y.copy()
    first = model.decompose(y, window_sizes=[9, 5])
    second = model.decompose(y, window_sizes=[9, 5])
    assert np.array_equal(first.imfs, second.imfs)
    first.imfs[:] = 999
    first.residual[:] = 999
    first.window_sizes.clear()
    first.bandwidths.clear()
    first.stages.clear()
    repeated = model.decompose(y, window_sizes=[9, 5])
    assert np.array_equal(second.imfs, repeated.imfs)
    assert np.array_equal(second.residual, repeated.residual)
    assert second.window_sizes == [9, 5]
    assert len(second.bandwidths) == len(second.stages) == 2
    assert np.array_equal(y, original)


@pytest.mark.parametrize("contrast", [Quadratic(), SmoothAbs(H=0.2)])
@pytest.mark.parametrize("y", [[7.0], [0.0, 100.0], [0.0, 100.0, 0.0]])
def test_center_only_small_signals_are_identity(contrast, y):
    result = IMF(contrast).decompose(y)
    assert result.imfs.shape == (1, len(y))
    np.testing.assert_allclose(result.imfs[0], y, atol=1e-12, rtol=0)
    np.testing.assert_allclose(result.residual, 0, atol=1e-12, rtol=0)
    assert result.stages[0].converged


@pytest.mark.parametrize("y", [[], [[1, 2]], [0, np.nan], [0, np.inf], 1.0])
def test_invalid_signals_raise(y):
    with pytest.raises(ValueError):
        IMF().decompose(y)


@pytest.mark.parametrize(
    "config",
    [
        {"max_iter": 0},
        {"max_iter": 1.5},
        {"max_iter": True},
        {"tol": 0},
        {"tol": np.nan},
        {"tol": np.inf},
        {"tol": True},
    ],
)
def test_invalid_solver_configuration_raises(config):
    with pytest.raises(ValueError):
        IMF(**config)


@pytest.mark.parametrize(
    "schedule",
    [
        {"h1": 0},
        {"h1": 0.6},
        {"h1": np.nan},
        {"a": 1},
        {"a": np.inf},
        {"k_max": 0},
        {"k_max": 1.5},
        {"k_max": True},
        {"k_max": None},
        {"h_min": 0},
        {"h_min": 0.3},
        {"h_min": np.nan},
        {"window_sizes": []},
        {"window_sizes": [0]},
        {"window_sizes": [2]},
        {"window_sizes": [3.5]},
        {"window_sizes": [True]},
        {"window_sizes": [np.inf]},
        {"window_sizes": 7},
        {"window_sizes": [5], "h1": 0.1},
        {"window_sizes": [5], "k_max": 2},
    ],
)
def test_invalid_schedule_raises(schedule):
    with pytest.raises(ValueError):
        IMF().decompose(np.zeros(32), **schedule)


def test_convergence_metadata_distinguishes_capped_and_converged_fits():
    y = np.array([0.0, 0.0, 10.0, 2.0, 0.0, 1.0, 0.0])
    capped = IMF(SmoothAbs(H=0.2), max_iter=1, tol=1e-15).decompose(y, window_sizes=[5])
    stage = capped.stages[0]
    assert stage.stage == 1
    assert stage.window_size == 5
    assert stage.bandwidth == 2 / len(y)
    assert stage.iterations == 1
    assert stage.final_max_delta > 1e-15 * (1 + np.max(np.abs(capped.imfs[0])))
    assert not stage.converged
    converged = IMF(SmoothAbs(H=0.2)).decompose(np.ones(32), window_sizes=[5])
    assert converged.stages[0].converged
    assert converged.stages[0].iterations == 1
    assert converged.stages[0].final_max_delta == 0
    closed = IMF().decompose(y, window_sizes=[5])
    assert closed.stages[0].converged
    assert closed.stages[0].iterations == 1
    assert np.isnan(closed.stages[0].final_max_delta)
