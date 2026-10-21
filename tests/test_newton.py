import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view

from pimf import IMF, Contrast, Quadratic, SmoothAbs, squared_triangle, triangle, uniform
from pimf._newton import _newton_fit_windows
from pimf.contrasts import _SCORE, _SCORE_DELTA
from pimf.decompose import _gd_fit_windows


@pytest.fixture(scope="module")
def reference():
    with np.load(Path(__file__).parent / "data" / "newton_reference.npz") as data:
        return dict(data)


def bisect_roots(windows, weights, contrast, iterations=100):
    """Independent score bisection; return the midpoint of the zero set."""
    positive = weights > 0
    windows = np.ascontiguousarray(windows[:, positive], dtype=float)
    weights = weights[positive]
    weights = weights / weights.sum()
    edges = []
    for right_edge in (False, True):
        lo, hi = windows.min(axis=1), windows.max(axis=1)
        for _ in range(iterations):
            mid = 0.5 * lo + 0.5 * hi
            if np.all((mid == lo) | (mid == hi)):
                break
            with np.errstate(over="ignore", invalid="ignore"):
                score = np.sum(weights * contrast.psi(windows - mid[:, None]), axis=1)
            positive_score = score >= 0 if right_edge else score > 0
            lo = np.where(positive_score, mid, lo)
            hi = np.where(positive_score, hi, mid)
        edges.append(0.5 * lo + 0.5 * hi)
    return 0.5 * edges[0] + 0.5 * edges[1]


# Saved slopes are 256 * differences of scores that match across platforms to 2e-15.
SLOPE_ATOL = 256 * 2 * 2e-15


@pytest.mark.parametrize("H", [0.01, 1.0, 10.0])
def test_lookup_score_slope_and_H_scaling(reference, H):
    z = reference["slope_z"]
    midpoints = 0.5 * (z[:-1] + z[1:]) * H
    contrast = SmoothAbs(H)
    expected = reference["slope_lookup"][:-1] / H
    np.testing.assert_allclose(contrast.psi_prime(midpoints), expected, rtol=0, atol=SLOPE_ATOL / H)
    eps = H / 2048
    difference = (contrast.psi(midpoints + eps) - contrast.psi(midpoints - eps)) / (2 * eps)
    np.testing.assert_allclose(contrast.psi_prime(midpoints), difference, rtol=0, atol=2e-12 / H)


def test_lookup_slope_at_knots_and_outside_grid(reference):
    contrast = SmoothAbs(1)
    np.testing.assert_allclose(
        contrast.psi_prime(reference["slope_z"]), reference["slope_lookup"], rtol=0, atol=SLOPE_ATOL
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
            atol=SLOPE_ATOL,
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
    gd, _, delta, unconverged_windows, _, _ = _gd_fit_windows(
        windows, weights, contrast, 20000, 1e-12
    )
    assert delta <= 1e-12 * (1 + np.max(np.abs(gd)))
    assert unconverged_windows == 0
    newton, _, _, unfinished, _, _ = _newton_fit_windows(windows, weights, contrast, 100, 1e-12)
    assert unfinished == 0
    np.testing.assert_allclose(newton, gd, rtol=2e-11, atol=2e-11)


def test_flat_zero_interval_uses_its_midpoint(reference):
    windows, weights = reference["flat_windows"], reference["flat_weights"]
    contrast = SmoothAbs(1)
    tol = 1e-12
    roots, _, _, unfinished, fallbacks, score = _newton_fit_windows(
        windows, weights, contrast, 100, tol
    )
    assert unfinished == 0
    assert fallbacks > 0
    assert score == pytest.approx(0, rel=0, abs=1e-15)
    assert abs(roots[0] - (windows @ weights)[0]) > 1
    tolerance = tol * (1 + np.max(np.abs(reference["flat_boundaries"])))
    np.testing.assert_allclose(roots, reference["flat_root"], rtol=0, atol=tolerance)
    np.testing.assert_allclose(
        roots, bisect_roots(windows, weights, contrast), rtol=0, atol=tolerance
    )
    permuted = _newton_fit_windows(windows[:, ::-1], weights[::-1], contrast, 100, 1e-12)[0]
    np.testing.assert_allclose(permuted, roots, rtol=0, atol=tolerance)


def test_zero_weight_samples_do_not_change_bracket_or_root(reference):
    windows = np.c_[[-1e100], reference["flat_windows"], [1e100]]
    weights = np.r_[0.0, reference["flat_weights"], 0.0]
    tol = 1e-12
    root, iterations, _, unfinished, _, _ = _newton_fit_windows(
        windows, weights, SmoothAbs(1), 100, tol
    )
    assert unfinished == 0
    assert iterations < 100
    tolerance = tol * (1 + np.max(np.abs(reference["flat_boundaries"])))
    np.testing.assert_allclose(root, reference["flat_root"], rtol=0, atol=tolerance)


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
            match=r"Stage 1 \(window size 7\): 5 windows did not converge after 1 iteration\.",
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
            "Stage 1 (window size 7): 5 windows did not converge after 1 iteration. "
            'Increase max_iter or set on_unconverged="ignore".'
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


@pytest.mark.parametrize("lookup", [False, True])
def test_tiny_H_needs_a_positional_bracket(reference, lookup):
    y = reference["tiny_H_window"][0]
    contrast = SmoothAbs(1e-14, lookup=lookup)
    result = IMF(contrast, kernel=uniform, solver="newton", on_unconverged="raise").decompose(
        y, window_sizes=[5]
    )
    stage = result.stages[0]
    assert stage.converged and stage.unconverged_windows == 0
    np.testing.assert_allclose(
        result.imfs[0], np.repeat(reference[f"tiny_H_root_{lookup}"], len(y)), rtol=0, atol=2e-12
    )
    assert stage.max_abs_score > 0.1


@pytest.mark.parametrize("lookup", [False, True])
@pytest.mark.parametrize("sign", [1, -1])
def test_plateau_boundaries_and_mirror_symmetry(reference, lookup, sign):
    windows = reference["adversarial_windows"]
    weights = reference["adversarial_weights"]
    contrast = SmoothAbs(1, lookup=lookup)
    roots, _, _, unfinished, _, _ = _newton_fit_windows(
        (sign * windows).astype(int), weights, contrast, 100, 1e-12
    )
    assert unfinished == 0
    np.testing.assert_allclose(
        roots, sign * reference[f"adversarial_roots_{lookup}"], rtol=2e-12, atol=2e-11
    )
    mirrored = _newton_fit_windows(-sign * windows, weights, contrast, 100, 1e-12)[0]
    np.testing.assert_allclose(mirrored, -roots, rtol=2e-12, atol=2e-11)


@pytest.mark.parametrize("lookup", [False, True])
@pytest.mark.parametrize("sign", [1, -1])
def test_triangle_plateau_at_grid_edge_and_direct_saturation(reference, lookup, sign):
    contrast = SmoothAbs(1, lookup=lookup)
    for y, expected in zip(
        reference["adversarial_triangle_windows"],
        reference[f"adversarial_triangle_roots_{lookup}"],
        strict=True,
    ):
        result = IMF(contrast, kernel=triangle, solver="newton", on_unconverged="raise").decompose(
            sign * y, window_sizes=[5]
        )
        np.testing.assert_allclose(result.imfs[0, 2], sign * expected, rtol=2e-12, atol=2e-11)
        assert result.stages[0].converged


@pytest.mark.parametrize("lookup", [False, True])
@pytest.mark.parametrize("index", [0, 1, 2])
def test_wide_plateau_stops_relative_to_its_edges(reference, lookup, index):
    windows = reference["wide_windows"][[index]]
    roots, iterations, _, unfinished, _, _ = _newton_fit_windows(
        windows, reference["adversarial_weights"], SmoothAbs(1, lookup=lookup), 100, 1e-12
    )
    assert unfinished == 0
    assert iterations < 100
    np.testing.assert_allclose(
        roots, reference[f"wide_roots_{lookup}"][[index]], rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("policy", ["warn", "raise", "ignore"])
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_newton_score_policy_and_parallel_diagnostics(workers, reverse, policy, value):
    class NonfiniteScore(Contrast):
        def psi(self, r):
            return np.where(r > 5, value, r)

        def psi_prime(self, r):
            return np.ones_like(r)

    y = np.zeros(129)
    y[-2] = 10
    if reverse:
        y = y[::-1]
    method = IMF(
        NonfiniteScore(),
        kernel=uniform,
        solver="newton",
        workers=workers,
        max_iter=2,
        on_unconverged=policy,
    )
    if policy == "raise":
        with pytest.raises(RuntimeError, match="windows did not converge"):
            method.decompose(y, window_sizes=[3])
        return
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = method.decompose(y, window_sizes=[3])
    assert len(caught) == (1 if policy == "warn" else 0)
    stage = result.stages[0]
    assert not stage.converged
    assert stage.unconverged_windows == 3
    assert np.isnan(stage.final_max_delta)
    if np.isnan(value):
        assert np.isnan(stage.max_abs_score)
    else:
        assert np.isinf(stage.max_abs_score)


def test_nonfinite_newton_estimate_is_not_done():
    with np.errstate(all="ignore"):
        roots, _, delta, unfinished, _, score = _newton_fit_windows(
            np.array([[np.nan, 0, 1]]), np.ones(3), SmoothAbs(1), 2, 1e-12
        )
    assert np.isnan(roots[0])
    assert unfinished == 1
    assert np.isnan(delta) and np.isnan(score)


@pytest.mark.parametrize(
    ("window", "expected"),
    [
        ([0.1, -0.2, 0, np.inf, 0.3, -0.1, 0.2], 0.10780207441813276),
        ([0.1, -0.2, 0, np.inf, 0.3, -np.inf, 0.2], None),
        ([-np.inf, np.inf, 0.0], 0.0),
    ],
)
def test_infinite_samples_give_finite_roots(window, expected):
    window = np.array([window])
    roots, _, _, unfinished, _, score = _newton_fit_windows(
        window, np.ones(window.shape[1]), SmoothAbs(0.2), 100, 1e-12
    )
    assert unfinished == 0
    assert np.isfinite(roots[0]) and np.isfinite(score)
    if expected is not None:
        np.testing.assert_allclose(roots[0], expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("lookup", [True, False])
def test_overflowed_residuals_keep_newton_finite(lookup):
    big = np.finfo(float).max
    y = np.r_[np.full(20, -big), big, np.full(20, -big), np.zeros(10)]
    with np.errstate(over="ignore"):
        result = IMF(
            SmoothAbs(0.2, lookup=lookup), solver="newton", on_unconverged="raise"
        ).decompose(y, k_max=4)
    assert np.all(np.isfinite(result.imfs))
    assert all(stage.converged for stage in result.stages)


@pytest.mark.parametrize("workers", [1, 3])
def test_identical_gd_runs_have_equal_stages(workers):
    y = np.random.default_rng(6).normal(size=129)
    method = IMF(SmoothAbs(0.4), workers=workers, on_unconverged="ignore")
    a = method.decompose(y, window_sizes=[65])
    b = method.decompose(y, window_sizes=[65])
    assert a.stages == b.stages
    assert np.isfinite(a.stages[0].max_abs_score)


@pytest.mark.parametrize("max_iter", [1, 5, 60])
def test_gd_score_uses_last_evaluation_without_extra_pass(max_iter):
    class RecordingScore(SmoothAbs):
        def __init__(self):
            super().__init__(0.4)
            self.scores = []

        def psi(self, r):
            score = super().psi(r)
            self.scores.append(score)
            return score

    contrast = RecordingScore()
    y = np.random.default_rng(6).normal(size=129)
    result = IMF(contrast, max_iter=max_iter, on_unconverged="ignore").decompose(
        y, window_sizes=[65]
    )
    stage = result.stages[0]
    assert len(contrast.scores) == stage.iterations
    weights = squared_triangle.weights(65)
    scores = np.sum((weights / weights.sum()) * contrast.scores[-1], axis=1)
    np.testing.assert_allclose(stage.max_abs_score, np.max(np.abs(scores)), rtol=0, atol=1e-15)


@pytest.mark.parametrize("lookup", [False, True])
def test_newton_singleton_last_chunk_is_bit_identical(lookup):
    y = np.random.default_rng(6).normal(size=129)
    config = dict(contrast=SmoothAbs(0.4, lookup=lookup), solver="newton")
    serial = IMF(**config).decompose(y, window_sizes=[65, 17, 7])
    parallel = IMF(**config, workers=8).decompose(y, window_sizes=[65, 17, 7])
    assert np.array_equal(parallel.imfs, serial.imfs)
    assert np.array_equal(parallel.residual, serial.residual)
    assert parallel.stages == serial.stages


@pytest.mark.parametrize("lookup", [False, True])
def test_small_H_score_slope_avoids_intermediate_overflow(lookup):
    H = 1e-306
    contrast = SmoothAbs(H, lookup=lookup)
    with np.errstate(all="raise"):
        slopes = contrast.psi_prime(np.array([0, 8 * H]))
    assert np.all(np.isfinite(slopes))
    expected = SmoothAbs(1, lookup=lookup).psi_prime(np.array([0, 8])) / H
    np.testing.assert_allclose(slopes, expected, rtol=1e-14, atol=0)
    with np.errstate(all="raise"):
        residuals = np.array([8e-310, 9e-310])
        tail = SmoothAbs(1e-310, lookup=lookup).psi_prime(residuals)
    assert np.all(np.isfinite(tail))
    expected = SmoothAbs(1, lookup=lookup).psi_prime(residuals / 1e-310) / 1e-310
    np.testing.assert_allclose(tail, expected, rtol=1e-13, atol=0)


def test_derivative_probe_preserves_attribute_errors():
    class BrokenSlope(Contrast):
        def psi_prime(self, r):
            return self.missing_attribute

    with pytest.raises(AttributeError, match="missing_attribute"):
        IMF(BrokenSlope(), solver="newton")


@pytest.mark.parametrize("lookup", [False, True])
def test_extreme_outliers_match_research_score_roots(reference, lookup):
    for prefix, weights in (("extreme", np.ones(7)), ("extreme_spike", reference["spike_weights"])):
        windows = reference["extreme_windows" if prefix == "extreme" else "extreme_spikes"]
        roots, iterations, _, unfinished, _, _ = _newton_fit_windows(
            windows, weights, SmoothAbs(0.2, lookup=lookup), 100, 1e-12
        )
        assert unfinished == 0
        assert iterations < 100
        np.testing.assert_allclose(
            roots, reference[f"{prefix}_roots_{lookup}"], rtol=2e-12, atol=2e-12
        )


@pytest.mark.parametrize("big", [1e30, 9.97e36, 3.4e38])
@pytest.mark.parametrize("kernel", [uniform, squared_triangle])
@pytest.mark.parametrize("lookup", [False, True])
def test_extreme_outlier_decompositions_finish(big, kernel, lookup):
    method = IMF(
        SmoothAbs(0.2, lookup=lookup), kernel=kernel, solver="newton", on_unconverged="raise"
    )
    y = np.random.default_rng(42).normal(0, 0.1, 512)
    y[128] = big
    result = method.decompose(y, window_sizes=[7])
    assert all(stage.converged and stage.iterations < 100 for stage in result.stages)
    for schedule in ({}, {"k_max": None, "h_min": 1.5 / 512}):
        y = np.sin(2 * np.pi * np.arange(512) / 512)
        y[128] = big
        result = method.decompose(y, **schedule)
        assert all(stage.converged and stage.iterations < 100 for stage in result.stages)
        assert np.all(np.isfinite(result.imfs))
        assert np.all(np.isfinite(result.residual))


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("size", [1, 7])
def test_degenerate_extreme_windows_finish_without_probes(sign, size):
    class ConstantScore(SmoothAbs):
        def psi(self, r):
            assert np.all(r == 0), "constant windows must not probe outside the bracket"
            return super().psi(r)

    y = np.full(size, sign * np.finfo(float).max)
    result = IMF(ConstantScore(1), solver="newton", on_unconverged="raise").decompose(
        y, window_sizes=[size]
    )
    np.testing.assert_allclose(result.imfs[0], y, rtol=0, atol=0)
    np.testing.assert_allclose(result.residual, 0.0, rtol=0, atol=0)
    stage = result.stages[0]
    assert stage.converged and stage.unconverged_windows == 0
    assert stage.iterations == stage.bisection_fallbacks == 0
    assert stage.final_max_delta == stage.max_abs_score == 0


@pytest.mark.parametrize("lookup", [False, True])
def test_extreme_bracket_range_and_probe_arithmetic_are_quiet(lookup):
    limit = np.finfo(float).max
    windows = np.array([[-limit, -limit / 2, limit], [-limit, limit / 2, limit]])
    roots, _, delta, unfinished, _, score = _newton_fit_windows(
        windows, np.ones(3), SmoothAbs(1, lookup=lookup), 100, 1e-12
    )
    assert unfinished == 0
    assert np.isfinite(delta) and np.isfinite(score)
    np.testing.assert_allclose(roots / limit, [-0.5, 0.5], rtol=2e-12, atol=0)


@pytest.mark.parametrize("lookup", [False, True])
def test_outlier_restart_and_midpoints_preserve_worker_parity(lookup):
    y = np.sin(2 * np.pi * np.arange(129) / 129)
    y[32] = 3.4e38
    config = dict(contrast=SmoothAbs(0.2, lookup=lookup), solver="newton", on_unconverged="raise")
    serial = IMF(**config).decompose(y, k_max=None, h_min=1.5 / len(y))
    parallel = IMF(**config, workers=8).decompose(y, k_max=None, h_min=1.5 / len(y))
    assert np.array_equal(parallel.imfs, serial.imfs)
    assert np.array_equal(parallel.residual, serial.residual)
    assert parallel.stages == serial.stages


@pytest.mark.parametrize("lookup", [False, True])
def test_weighted_median_restart_avoids_low_weight_outlier_cluster(reference, lookup):
    contrast = SmoothAbs(0.2, lookup=lookup)
    for y, expected in zip(
        reference["cluster_windows"], reference[f"cluster_roots_{lookup}"], strict=True
    ):
        result = IMF(contrast, solver="newton", on_unconverged="raise").decompose(
            y, window_sizes=[5]
        )
        assert result.stages[0].converged and result.stages[0].iterations < 10
        np.testing.assert_allclose(result.imfs[0, 2], expected, rtol=0, atol=2e-12)


@pytest.mark.parametrize("lookup", [False, True])
def test_zero_slope_outlier_bracket_closes(reference, lookup):
    windows, weights = reference["stalled_window"], reference["stalled_weights"]
    contrast = SmoothAbs(0.01, lookup=lookup)
    roots, iterations, _, unfinished, _, _ = _newton_fit_windows(
        windows, weights, contrast, 100, 1e-12
    )
    assert unfinished == 0 and iterations < 100
    np.testing.assert_allclose(roots, reference[f"stalled_root_{lookup}"], rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        roots, bisect_roots(windows, weights, contrast, iterations=256), rtol=2e-12, atol=2e-12
    )


def test_square_wave_fill_values_do_not_spread_to_non_fill_samples(reference):
    y = reference["fill_y"]
    result = IMF(SmoothAbs(0.02), solver="newton", on_unconverged="raise").decompose(
        y, k_max=None, h_min=1.5 / len(y)
    )
    assert all(stage.converged and stage.iterations < 100 for stage in result.stages)
    non_fill = np.abs(y) < 1e30
    assert np.max(np.abs(result.imfs[:, non_fill])) < 2
    np.testing.assert_allclose(
        result.imfs[:, non_fill], reference["fill_imfs"][:, non_fill], rtol=0, atol=2e-10
    )


def test_huber_integer_example_with_gross_outlier(reference):
    class Huber(Contrast):
        def psi(self, r):
            return np.clip(r, -0.3, 0.3)

        def psi_prime(self, r):
            return (np.abs(r) < 0.3).astype(float)

    result = IMF(Huber(), solver="newton", on_unconverged="raise").decompose(reference["huber_y"])
    assert all(stage.converged and stage.iterations < 100 for stage in result.stages)
    np.testing.assert_allclose(
        result.imfs[0], reference["huber_first_component"], rtol=0, atol=1e-11
    )


@pytest.mark.parametrize("size", [7, 15, 33])
@pytest.mark.parametrize("kernel", ["uniform", "squared_triangle"])
@pytest.mark.parametrize("H", [0.003, 0.01])
@pytest.mark.parametrize("lookup", [False, True])
def test_outlier_magnitude_sweep_matches_independent_bisection(reference, size, kernel, H, lookup):
    windows = reference[f"sweep_{size}_windows"]
    weights = reference[f"sweep_{size}_{kernel}_weights"]
    contrast = SmoothAbs(H, lookup=lookup)
    roots, iterations, _, unfinished, _, _ = _newton_fit_windows(
        windows, weights, contrast, 100, 1e-12
    )
    assert unfinished == 0 and iterations < 100
    np.testing.assert_allclose(
        roots, reference[f"sweep_{size}_{kernel}_{H}_{lookup}"], rtol=2e-12, atol=2e-12
    )
    np.testing.assert_allclose(
        roots, bisect_roots(windows, weights, contrast, iterations=1200), rtol=2e-12, atol=2e-12
    )


def test_lookup_table_steps_are_exact_and_nonnegative():
    # Probe skipping relies on lookup scores being nonincreasing as computed.
    assert np.all(_SCORE_DELTA >= 0)
    exact = [
        Fraction(a) + Fraction(d) == Fraction(b)
        for a, d, b in zip(_SCORE[:-1], _SCORE_DELTA[:-1], _SCORE[1:], strict=True)
    ]
    assert all(exact)


class GenericSmoothAbs(SmoothAbs):
    """SmoothAbs evaluated through the generic psi and psi_prime path."""


def assert_same_fit(fused, generic):
    np.testing.assert_array_equal(fused[0], generic[0])
    np.testing.assert_array_equal(fused[1:], generic[1:])


@pytest.mark.parametrize("H", [0.01, 1.0, 10.0])
@pytest.mark.parametrize("size", [3, 65, 1025])
def test_fused_lookup_is_bit_identical_to_generic_path(reference, H, size):
    prefix = f"h{[0.01, 1.0, 10.0].index(H)}_w{size}_"
    windows, weights = reference[prefix + "windows"], reference[prefix + "weights"]
    for max_iter in (1, 3, 100):
        assert_same_fit(
            _newton_fit_windows(windows, weights, SmoothAbs(H), max_iter, 1e-12),
            _newton_fit_windows(windows, weights, GenericSmoothAbs(H), max_iter, 1e-12),
        )


def test_fused_lookup_matches_generic_path_on_nonfinite_and_extreme_samples():
    big = np.finfo(float).max
    windows = np.array(
        [
            [0.1, -0.2, np.inf, 0.3, -np.inf],
            [np.nan, 0.0, 1.0, 2.0, 3.0],
            [big, -big, 0.0, 0.1, 0.2],
            [big, big, big, 0.0, big],
            [np.inf, np.inf, np.inf, 0.0, 1.0],
            [-8.0, -7.999, 0.0, 7.999, 8.0],
        ]
    )
    weights = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    fused = _newton_fit_windows(windows, weights, SmoothAbs(1), 100, 1e-12)
    assert_same_fit(fused, _newton_fit_windows(windows, weights, GenericSmoothAbs(1), 100, 1e-12))
    assert fused[3] == 2


@pytest.mark.parametrize("H", [1e-300, 1e-307, 1e4, 1e300])
def test_extreme_H_lookup_matches_generic_path(H):
    windows = np.array(
        [[0.0, 1e-300, -1e-300, 3e-300, 2.0], [1.0, 2.0, 3.0, 4.0, 5.0], [-1, 0, 1, 0, 0]]
    )
    for max_iter in (3, 100):
        assert_same_fit(
            _newton_fit_windows(windows, np.ones(5), SmoothAbs(H), max_iter, 1e-12),
            _newton_fit_windows(windows, np.ones(5), GenericSmoothAbs(H), max_iter, 1e-12),
        )


@pytest.mark.parametrize("lookup", [False, True])
def test_longdouble_weights_bisect_like_float64_weights(lookup):
    class LongDoubleKernel:
        def weights(self, size, *, bandwidth=None):
            return squared_triangle.weights(size, bandwidth=bandwidth).astype(np.longdouble)

    y = np.random.default_rng(0).normal(0, 0.1, 64)
    y[20] = 100.0
    contrast = SmoothAbs(1.0, lookup=lookup)
    wide = IMF(contrast, LongDoubleKernel(), solver="newton", on_unconverged="raise").decompose(y)
    plain = IMF(contrast, solver="newton", on_unconverged="raise").decompose(y)
    assert wide.stages[0].bisection_fallbacks > 0
    np.testing.assert_allclose(wide.imfs, plain.imfs, rtol=0, atol=1e-12)


def test_huge_H_slope_table_does_not_raise_underflow():
    with np.errstate(under="raise"):
        result = IMF(SmoothAbs(1e300), solver="newton").decompose(np.ones(3), window_sizes=[3])
    np.testing.assert_array_equal(result.imfs, [[1.0, 1.0, 1.0]])


def test_large_H_plateau_root_matches_generic_path():
    windows, weights = np.array([[-1.0, 0.0, 1.0]]), np.ones(3)
    for max_iter in (3, 100):
        fused = _newton_fit_windows(windows, weights, SmoothAbs(1e4), max_iter, 1e-12)
        generic = _newton_fit_windows(windows, weights, GenericSmoothAbs(1e4), max_iter, 1e-12)
        assert_same_fit(fused, generic)
        assert fused[3] == 0


def test_residuals_just_beyond_8H_have_zero_slope():
    y = [0.0, 0.0, np.nextafter(36.0, np.inf)]
    for contrast in (SmoothAbs(1.5), GenericSmoothAbs(1.5)):
        result = IMF(
            contrast, kernel=uniform, solver="newton", max_iter=4, on_unconverged="raise"
        ).decompose(y, window_sizes=[3])
        assert result.stages[0].converged and result.stages[0].bisection_fallbacks == 0
    windows = sliding_window_view(np.pad(y, 1, mode="wrap"), 3)
    assert_same_fit(
        _newton_fit_windows(windows, np.ones(3), SmoothAbs(1.5), 4, 1e-12),
        _newton_fit_windows(windows, np.ones(3), GenericSmoothAbs(1.5), 4, 1e-12),
    )


def test_direct_scores_evaluate_both_probes():
    # erf_approx can wiggle by rounding across a probe interval when H is large.
    windows = np.array([[-731.7543616953625, 998.8702915620365]])
    weights = np.array([1.0757267432751028, 0.7886655183662474])
    contrast = SmoothAbs(1e4, lookup=False)
    for max_iter in (4, 100):
        roots, iterations, _, unfinished, _, _ = _newton_fit_windows(
            windows, weights, contrast, max_iter, 1e-12
        )
        assert unfinished == 0 and iterations == 4
        np.testing.assert_allclose(roots, 1.98e-14, rtol=0, atol=1e-13)


def test_overflowing_probes_and_8H_match_generic_path():
    big = np.finfo(float).max
    cases = [
        (np.array([[0, 0, big, big, big, np.inf]]), np.array([0.7, 0.3, 0.1, 0.1, 0.9, 1.0]), 0.2),
        (np.array([[-big, big, 0.0]]), np.array([1.0, 2.0, 1.0]), 2.5e307),
        (np.array([[-np.inf, -big, -1.7976931346825464e308]]), np.array([1.0, 2.0, 2.0]), 1e296),
    ]
    for windows, weights, H in cases:
        assert_same_fit(
            _newton_fit_windows(windows, weights, SmoothAbs(H), 100, 1e-12),
            _newton_fit_windows(windows, weights, GenericSmoothAbs(H), 100, 1e-12),
        )


def test_custom_float32_scores_keep_float32_steps():
    class Cubic32(Contrast):
        def psi(self, r):
            return r.astype(np.float32) ** 3

        def psi_prime(self, r):
            return 3 * r.astype(np.float32) ** 2

    windows, weights = np.array([[0.0, 1.0, 3.0]]), np.array([1, 2, 3], dtype=np.float32)
    roots = _newton_fit_windows(windows, weights, Cubic32(), 1, 1e-12)[0]
    weights = weights / weights.sum()
    x = np.sum(windows * weights, axis=1)
    score = np.sum(weights * Cubic32().psi(windows - x[:, None]), axis=1)
    slope = np.sum(weights * Cubic32().psi_prime(windows - x[:, None]), axis=1)
    assert score.dtype == slope.dtype == np.float32
    np.testing.assert_array_equal(roots, x + np.divide(score, slope, out=np.empty(1)))


def test_custom_scores_reduce_over_the_full_batch():
    class FortranSmoothAbs(SmoothAbs):
        def psi(self, r):
            return np.asfortranarray(super().psi(r))

        def psi_prime(self, r):
            return np.asfortranarray(super().psi_prime(r))

    windows, weights = np.random.default_rng(0).normal(size=(2, 65)), np.ones(65) / 65
    contrast = FortranSmoothAbs(0.4)
    roots, _, _, _, _, score = _newton_fit_windows(windows, weights, contrast, 100, 1e-12)
    scores = np.sum(weights * contrast.psi(windows - roots[:, None]), axis=1)
    assert score == np.max(np.abs(scores))


@pytest.mark.parametrize("lookup", [False, True])
def test_nan_and_zero_weights_inside_the_span_are_dropped(lookup):
    windows = np.random.default_rng(2).normal(size=(4, 5))
    contrast = SmoothAbs(0.5, lookup=lookup)
    expected = _newton_fit_windows(windows[:, [0, 3, 4]], np.ones(3), contrast, 100, 1e-12)
    for weights in ([1, np.nan, 0, 1, 1], [1, 0, np.nan, 1, 1]):
        result = _newton_fit_windows(windows, np.array(weights, float), contrast, 100, 1e-12)
        assert_same_fit(result, expected)


@pytest.mark.parametrize("size", [65, 9001])
def test_singleton_rows_match_batched_rows(size):
    windows = np.random.default_rng(7).normal(size=(5, size))
    weights = squared_triangle.weights(size)
    batch = _newton_fit_windows(windows, weights, SmoothAbs(0.4), 100, 1e-12)
    single = [
        _newton_fit_windows(row[None], weights, SmoothAbs(0.4), 100, 1e-12) for row in windows
    ]
    assert np.array_equal(batch[0], [fit[0][0] for fit in single])
    assert batch[5] == max(fit[5] for fit in single)
