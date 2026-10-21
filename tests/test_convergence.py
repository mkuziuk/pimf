import re
import warnings

import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view

from pimf import IMF, Quadratic, SmoothAbs


@pytest.fixture
def spike():
    y = np.zeros(64)
    y[len(y) // 2] = 100.0
    return y


def test_issue_3_spike_warns_by_default(spike):
    with pytest.warns(RuntimeWarning, match=r"Stage 1 \(window size 7\)") as caught:
        result = IMF(SmoothAbs(1.0)).decompose(spike, h1=2 * np.sqrt(2) / len(spike), k_max=1)
    stage = result.stages[0]
    assert len(caught) == 1
    assert caught[0].filename == __file__
    assert str(caught[0].message) == (
        f"Stage 1 (window size 7): {stage.unconverged_windows} windows "
        "did not converge after 60 iterations"
    )
    assert stage.iterations == 60
    assert not stage.converged
    assert stage.unconverged_windows > 0
    np.testing.assert_allclose(result.imfs[0, len(spike) // 2], 2.43134204, rtol=0, atol=5e-9)


@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("policy", ["warn", "raise", "ignore"])
def test_unconverged_policy_and_total_count_across_chunks(spike, workers, policy):
    single = IMF(SmoothAbs(1.0), on_unconverged="ignore").decompose(
        spike, h1=2 * np.sqrt(2) / len(spike), k_max=1
    )
    expected_count = 4 * single.stages[0].unconverged_windows
    y = np.tile(spike, 4)
    method = IMF(SmoothAbs(1.0), workers=workers, on_unconverged=policy)
    message = (
        f"Stage 1 (window size 7): {expected_count} windows did not converge after 60 iterations"
    )
    if policy == "raise":
        with pytest.raises(RuntimeError, match=re.escape(message)):
            method.decompose(y, h1=2 * np.sqrt(2) / len(y), k_max=1)
        return
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = method.decompose(y, h1=2 * np.sqrt(2) / len(y), k_max=1)
    assert result.stages[0].unconverged_windows == expected_count
    assert not result.stages[0].converged
    assert len(caught) == (1 if policy == "warn" else 0)
    if caught:
        assert caught[0].category is RuntimeWarning
        assert str(caught[0].message) == message
    np.testing.assert_allclose(result.imfs[0], np.tile(single.imfs[0], 4), rtol=0, atol=1e-12)


@pytest.mark.parametrize("large_first", [False, True])
@pytest.mark.parametrize("max_iter", [1, 60])
@pytest.mark.parametrize("policy", ["warn", "raise", "ignore"])
def test_row_diagnostics_use_research_stopping_threshold(large_first, max_iter, policy):
    large = np.full(128, 1e9)
    small = np.tile([0.0, 0.0, 10.0, 2.0, 0.0, 1.0, 0.0, 0.0], 16)
    y = np.concatenate((large, small) if large_first else (small, large))
    result = IMF(
        SmoothAbs(0.2), boundary="edge", max_iter=max_iter, on_unconverged=policy
    ).decompose(y, window_sizes=[5])
    stage = result.stages[0]
    initial = np.median(sliding_window_view(np.pad(y, 2, mode="edge"), 5), axis=1)
    final = result.imfs[0]
    delta = np.abs(final - initial)
    threshold = 1e-6 * (1 + np.max(np.abs(final)))
    assert stage.iterations == 1
    assert stage.final_max_delta <= threshold
    assert np.any(delta > 1e-6 * (1 + np.abs(final)))
    assert stage.unconverged_windows == np.count_nonzero(delta > threshold) == 0
    assert stage.converged


def test_issue_3_bursty_signal_warns_at_defaults():
    n = 2048
    rng = np.random.default_rng(20261020 + n)
    t = np.arange(n)
    x = (1 + 8 * np.exp(-0.5 * ((t - 0.45 * n) / (0.06 * n)) ** 2)) * np.sin(2 * np.pi * 0.018 * t)
    x += rng.normal(0, 0.08, n)
    y = (x - np.median(x)) / (1.4826 * np.median(np.abs(x - np.median(x))))
    H = 2 * np.median(np.abs(np.diff(y))) / (0.6745 * np.sqrt(2))
    with pytest.warns(RuntimeWarning) as caught:
        result = IMF(SmoothAbs(H)).decompose(y, k_max=None, h_min=1.5 / n)
    unfinished = [stage for stage in result.stages if not stage.converged]
    assert len(result.stages) == 17
    assert len(caught) == len(unfinished)
    assert all(stage.iterations == 60 for stage in unfinished)
    for warning, stage in zip(caught, unfinished, strict=True):
        assert 0 < stage.unconverged_windows <= n
        assert str(warning.message) == (
            f"Stage {stage.stage} (window size {stage.window_size}): "
            f"{stage.unconverged_windows} windows did not converge "
            f"after {stage.iterations} iterations"
        )
    np.testing.assert_allclose(result.reconstruction, y, rtol=0, atol=1e-12)


@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("policy", ["warn", "raise", "ignore"])
@pytest.mark.parametrize("contrast", [Quadratic(), SmoothAbs(0.4)])
def test_converged_runs_are_silent(contrast, policy, workers):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = IMF(contrast, workers=workers, on_unconverged=policy).decompose(
            np.ones(256), window_sizes=[63, 31]
        )
    assert not caught
    assert all(stage.converged and stage.unconverged_windows == 0 for stage in result.stages)


@pytest.mark.parametrize("policy", ["", "WARN", "error", None, True, 1, [], {}, np.array(["warn"])])
def test_invalid_unconverged_policy_raises_value_error(policy):
    with pytest.raises(ValueError, match="on_unconverged"):
        IMF(on_unconverged=policy)
