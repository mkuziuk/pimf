import warnings
from pathlib import Path

import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view

from pimf import IMF, Contrast, Quadratic, SmoothAbs, squared_triangle
from pimf._newton import _newton_fit_windows
from pimf.decompose import _gd_fit_windows


@pytest.fixture(scope="module")
def reference():
    with np.load(Path(__file__).parent / "data" / "newton_reference.npz") as data:
        return dict(data)


def bisect_roots(windows, weights, contrast):
    """Independent score bisection; return the midpoint of the zero set."""
    positive = weights > 0
    windows, weights = windows[:, positive], weights[positive]
    weights = weights / weights.sum()
    edges = []
    for right_edge in (False, True):
        lo, hi = windows.min(axis=1), windows.max(axis=1)
        for _ in range(100):
            mid = 0.5 * lo + 0.5 * hi
            score = np.sum(weights * contrast.psi(windows - mid[:, None]), axis=1)
            positive_score = score >= 0 if right_edge else score > 0
            lo = np.where(positive_score, mid, lo)
            hi = np.where(positive_score, hi, mid)
        edges.append(0.5 * lo + 0.5 * hi)
    return 0.5 * edges[0] + 0.5 * edges[1]


@pytest.mark.parametrize("H", [0.01, 1.0, 10.0])
def test_lookup_score_slope_and_H_scaling(reference, H):
    z = reference["slope_z"]
    midpoints = 0.5 * (z[:-1] + z[1:]) * H
    contrast = SmoothAbs(H)
    expected = reference["slope_lookup"][:-1] / H
    np.testing.assert_allclose(contrast.psi_prime(midpoints), expected, rtol=1e-15, atol=1e-15)
    eps = H / 2048
    difference = (contrast.psi(midpoints + eps) - contrast.psi(midpoints - eps)) / (2 * eps)
    np.testing.assert_allclose(contrast.psi_prime(midpoints), difference, rtol=0, atol=2e-12 / H)


def test_lookup_slope_at_knots_and_outside_grid(reference):
    contrast = SmoothAbs(1)
    np.testing.assert_allclose(
        contrast.psi_prime(reference["slope_z"]), reference["slope_lookup"], rtol=0, atol=0
    )
    with np.errstate(all="raise"):
        np.testing.assert_allclose(
            contrast.psi_prime([-np.inf, -9, 9, np.inf, np.nan]),
            [0, 0, 0, 0, np.nan],
            rtol=0,
            atol=0,
        )
    with np.errstate(all="raise"):
        assert np.isnan(contrast.psi_prime(np.nan))
        np.testing.assert_allclose(
            contrast.psi_prime([[np.nan, 0], [-np.inf, np.inf]]),
            [[np.nan, reference["slope_lookup"][2048]], [0, 0]],
            rtol=0,
            atol=0,
        )
    assert np.shape(contrast.psi_prime(0)) == ()
    assert contrast.psi_prime(np.empty((2, 0))).shape == (2, 0)


@pytest.mark.parametrize("H", [0.01, 1.0, 10.0])
def test_direct_score_slope_and_H_scaling(reference, H):
    contrast = SmoothAbs(H, lookup=False)
    np.testing.assert_allclose(
        contrast.psi_prime(reference["slope_z"] * H),
        reference["slope_direct"] / H,
        rtol=1e-14,
        atol=1e-15,
    )


def test_base_and_quadratic_score_slopes():
    with pytest.raises(NotImplementedError):
        Contrast().psi_prime(np.zeros(3))
    np.testing.assert_allclose(Quadratic().psi_prime([-2, 0, 4]), [1, 1, 1], rtol=0, atol=0)


def test_rounded_newton_candidate_at_bracket_endpoint_can_finish():
    class ScoreOnly(Contrast):
        def psi(self, r):
            return r

        def psi_prime(self, r):
            return np.ones_like(r)

    windows, weights = np.array([[0.1, 0.2, 0.4]]), np.array([0.2, 0.3, 0.5])
    roots, iterations, _, unfinished, fallbacks, _ = _newton_fit_windows(
        windows, weights, ScoreOnly(), 1, 1e-12
    )
    assert unfinished == 0
    assert iterations == 1
    assert fallbacks == 0
    np.testing.assert_allclose(roots, windows @ weights, rtol=0, atol=1e-15)


@pytest.mark.parametrize("H", [0.01, 1.0, 10.0])
@pytest.mark.parametrize("size", [3, 5, 7, 65, 257, 1025])
@pytest.mark.parametrize("lookup", [False, True])
def test_roots_match_score_bisection_and_research_fixture(reference, H, size, lookup):
    prefix = f"h{[0.01, 1.0, 10.0].index(H)}_w{size}_"
    windows, weights = reference[prefix + "windows"], reference[prefix + "weights"]
    original = windows.copy()
    contrast = SmoothAbs(H, lookup=lookup)
    roots, iterations, delta, unfinished, _, _ = _newton_fit_windows(
        windows, weights, contrast, 100, 1e-12
    )
    assert unfinished == 0
    assert 1 <= iterations <= 100
    assert np.isfinite(delta)
    np.testing.assert_allclose(
        roots, bisect_roots(windows, weights, contrast), rtol=2e-12, atol=2e-12
    )
    np.testing.assert_allclose(roots, reference[prefix + f"roots_{lookup}"], rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(windows, original, rtol=0, atol=0)


@pytest.mark.parametrize("H", [0.01, 1.0, 10.0])
@pytest.mark.parametrize("size", [3, 7, 65, 1025])
@pytest.mark.parametrize("lookup", [False, True])
def test_agreement_with_tightly_converged_gd(reference, H, size, lookup):
    prefix = f"h{[0.01, 1.0, 10.0].index(H)}_w{size}_"
    windows, weights = reference[prefix + "windows"][:1], reference[prefix + "weights"]
    contrast = SmoothAbs(H, lookup=lookup)
    gd, _, delta, unconverged_windows = _gd_fit_windows(windows, weights, contrast, 20000, 1e-12)
    assert delta <= 1e-12 * (1 + np.max(np.abs(gd)))
    assert unconverged_windows == 0
    newton, _, _, unfinished, _, _ = _newton_fit_windows(windows, weights, contrast, 100, 1e-12)
    assert unfinished == 0
    np.testing.assert_allclose(newton, gd, rtol=2e-11, atol=2e-11)


def test_flat_zero_interval_uses_its_midpoint(reference):
    windows, weights = reference["flat_windows"], reference["flat_weights"]
    contrast = SmoothAbs(1)
    roots, _, _, unfinished, fallbacks, score = _newton_fit_windows(
        windows, weights, contrast, 100, 1e-12
    )
    assert unfinished == 0
    assert fallbacks > 0
    assert score == pytest.approx(0, rel=0, abs=1e-15)
    assert abs(roots[0] - (windows @ weights)[0]) > 1
    np.testing.assert_allclose(roots, reference["flat_root"], rtol=1e-12, atol=1e-11)
    np.testing.assert_allclose(
        roots, bisect_roots(windows, weights, contrast), rtol=1e-12, atol=1e-11
    )
    permuted = _newton_fit_windows(windows[:, ::-1], weights[::-1], contrast, 100, 1e-12)[0]
    np.testing.assert_allclose(permuted, roots, rtol=1e-12, atol=1e-11)


def test_zero_weight_samples_do_not_change_bracket_or_root(reference):
    windows = np.c_[[-1e100], reference["flat_windows"], [1e100]]
    weights = np.r_[0.0, reference["flat_weights"], 0.0]
    root, iterations, _, unfinished, _, _ = _newton_fit_windows(
        windows, weights, SmoothAbs(1), 100, 1e-12
    )
    assert unfinished == 0
    assert iterations < 100
    np.testing.assert_allclose(root, reference["flat_root"], rtol=1e-12, atol=1e-11)


def test_issue_3_spike(reference):
    y = np.zeros(64)
    y[32] = 100
    result = IMF(SmoothAbs(1), solver="newton").decompose(y, h1=2 * np.sqrt(2) / len(y), k_max=1)
    with pytest.warns(RuntimeWarning, match=r"Stage 1 \(window size 7\)"):
        gd = IMF(SmoothAbs(1)).decompose(y, h1=2 * np.sqrt(2) / len(y), k_max=1)
    stage = result.stages[0]
    assert stage.converged and stage.unconverged_windows == 0
    assert stage.iterations <= 15
    assert stage.bisection_fallbacks >= 4
    assert stage.final_max_delta <= 1e-12 * (1 + np.max(np.abs(result.imfs[0])))
    assert stage.max_abs_score < 1e-12
    assert not gd.stages[0].converged
    assert abs(gd.imfs[0, 32] - reference["spike_root"][0]) > 0.2
    np.testing.assert_allclose(
        result.imfs[0, 32], reference["spike_root"][0], rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(result.reconstruction, y, rtol=0, atol=1e-13)


def test_issue_3_bursty_full_schedule(reference):
    y, H = reference["bursty_y"], float(reference["bursty_H"])
    schedule = {"k_max": None, "h_min": 1.5 / len(y)}
    result = IMF(SmoothAbs(H), solver="newton").decompose(y, **schedule)
    gd = IMF(SmoothAbs(H), max_iter=20000, tol=1e-12).decompose(y, **schedule)
    assert all(stage.converged and stage.unconverged_windows == 0 for stage in result.stages)
    assert all(stage.converged for stage in gd.stages)
    assert result.window_sizes == reference["bursty_window_sizes"].tolist()
    np.testing.assert_allclose(result.imfs, reference["bursty_imfs"], rtol=0, atol=2e-11)
    np.testing.assert_allclose(result.residual, reference["bursty_residual"], rtol=0, atol=2e-11)
    np.testing.assert_allclose(result.imfs, gd.imfs, rtol=0, atol=2e-9)
    np.testing.assert_allclose(result.reconstruction, y, rtol=0, atol=1e-13)


@pytest.mark.parametrize("lookup", [False, True])
@pytest.mark.parametrize("workers", [2, 3, 8])
def test_workers_do_not_change_roots_or_diagnostics(reference, lookup, workers):
    y = reference["bursty_y"][::8]
    config = {
        "contrast": SmoothAbs(float(reference["bursty_H"]), lookup=lookup),
        "solver": "newton",
    }
    serial = IMF(**config).decompose(y, window_sizes=[129, 65, 7])
    parallel = IMF(**config, workers=workers).decompose(y, window_sizes=[129, 65, 7])
    np.testing.assert_allclose(parallel.imfs, serial.imfs, rtol=0, atol=0)
    np.testing.assert_allclose(parallel.residual, serial.residual, rtol=0, atol=0)
    assert parallel.stages == serial.stages


def test_default_gd_numerics_are_unchanged(reference):
    y, contrast = reference["bursty_y"][::8], SmoothAbs(0.4)
    weights = squared_triangle.weights(65)
    windows = sliding_window_view(np.pad(y, 32, mode="wrap"), 65)
    expected, iterations, delta, unfinished = _gd_fit_windows(windows, weights, contrast, 60, 1e-6)
    result = IMF(contrast, on_unconverged="ignore").decompose(y, window_sizes=[65])
    assert np.array_equal(result.imfs[0], expected)
    assert result.stages[0].iterations == iterations
    assert result.stages[0].final_max_delta == delta
    assert result.stages[0].unconverged_windows == unfinished
    assert result.stages[0].converged == (unfinished == 0)


@pytest.mark.parametrize("solver", [None, "", "Newton", "other", False, 1, [], np.nan])
def test_invalid_solver_raises(solver):
    with pytest.raises(ValueError, match="solver"):
        IMF(solver=solver)


@pytest.mark.parametrize("policy", [None, "", False, "error"])
def test_invalid_unconverged_policy_raises(policy):
    with pytest.raises(ValueError, match="on_unconverged"):
        IMF(on_unconverged=policy)


def test_missing_score_derivative_has_clear_error():
    class LegacyContrast(Contrast):
        def psi(self, r):
            return np.tanh(r)

        def curvature(self):
            return 1.0

    with pytest.raises(ValueError, match=r"newton.*requires contrast.psi_prime"):
        IMF(LegacyContrast(), solver="newton")
    with pytest.raises(ValueError, match=r"newton.*requires contrast.psi_prime"):
        IMF(object(), solver="newton")
    assert IMF(LegacyContrast()).decompose(np.ones(8), window_sizes=[3]).stages[0].converged


def test_closed_form_precedes_solver():
    class DirectContrast(Contrast):
        def solve(self, windows, weights):
            return windows @ weights

    y = np.arange(100.0)
    for contrast in (Quadratic(), DirectContrast()):
        newton = IMF(contrast, solver="newton").decompose(y, window_sizes=[17, 7])
        gd = IMF(contrast).decompose(y, window_sizes=[17, 7])
        assert np.array_equal(newton.imfs, gd.imfs)
        assert all(stage.converged and stage.iterations == 1 for stage in newton.stages)
        assert all(stage.unconverged_windows == 0 for stage in newton.stages)


@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("policy", ["warn", "raise", "ignore"])
def test_capped_newton_policy_and_diagnostics(workers, policy):
    y = np.zeros(256)
    y[128] = 100
    method = IMF(SmoothAbs(1), solver="newton", max_iter=1, workers=workers, on_unconverged=policy)
    if policy == "raise":
        with pytest.raises(
            RuntimeError,
            match=r"Stage 1 \(window size 7\): 5 windows did not converge after 1 iterations",
        ):
            method.decompose(y, h1=2 * np.sqrt(2) / len(y), k_max=1)
        return
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = method.decompose(y, h1=2 * np.sqrt(2) / len(y), k_max=1)
    assert len(caught) == (1 if policy == "warn" else 0)
    if caught:
        assert caught[0].category is RuntimeWarning
        assert caught[0].filename == __file__
        assert str(caught[0].message) == (
            "Stage 1 (window size 7): 5 windows did not converge after 1 iterations"
        )
    stage = result.stages[0]
    assert not stage.converged
    assert stage.iterations == 1
    assert stage.unconverged_windows == 5
    assert np.isfinite(stage.final_max_delta)
    assert stage.max_abs_score > 0


def test_solver_specific_defaults_and_explicit_overrides():
    gd = IMF()
    newton = IMF(solver="newton")
    custom = IMF(solver="newton", max_iter=17, tol=1e-9)
    assert (gd.max_iter, gd.tol) == (60, 1e-6)
    assert (newton.max_iter, newton.tol) == (100, 1e-12)
    assert (custom.max_iter, custom.tol) == (17, 1e-9)
