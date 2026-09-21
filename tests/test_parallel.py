from pathlib import Path
from queue import SimpleQueue
from threading import current_thread, main_thread

import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view

from pimf import IMF, Quadratic, SmoothAbs


@pytest.mark.parametrize("workers", [0, -1, 1.5, 2.0, True, np.bool_(True), None, np.nan, "2"])
def test_invalid_worker_count_raises(workers):
    with pytest.raises(ValueError, match="workers"):
        IMF(workers=workers)


@pytest.mark.parametrize("boundary", ["wrap", "reflect", "edge"])
@pytest.mark.parametrize("workers", [2, 3, 8, np.int64(4)])
def test_parallel_linear_components_match_serial_across_chunk_edges(boundary, workers):
    y = np.arange(197.0) / 10 + np.random.default_rng(777).normal(size=197)
    windows = [51, 23, 7]
    serial = IMF(boundary=boundary).decompose(y, window_sizes=windows)
    parallel = IMF(boundary=boundary, workers=workers).decompose(y, window_sizes=windows)
    assert np.array_equal(parallel.imfs, serial.imfs)
    assert np.array_equal(parallel.residual, serial.residual)
    assert parallel.window_sizes == serial.window_sizes
    assert parallel.bandwidths == serial.bandwidths
    assert all(stage.iterations == 1 and stage.converged for stage in parallel.stages)
    assert all(np.isnan(stage.final_max_delta) for stage in parallel.stages)


@pytest.fixture(scope="module")
def research_case():
    with np.load(Path(__file__).parent / "data" / "solver_reference.npz") as fixture:
        y = fixture["your_gd_1000_y"]
        reference = fixture["your_gd_1000_reference"]
    windows = [501, 355, 251, 177, 125, 89, 63, 45, 31]
    serial = IMF(SmoothAbs(H=0.4)).decompose(y, window_sizes=windows)
    return y, reference, windows, serial


@pytest.mark.parametrize("workers", [3, 7])
def test_parallel_robust_agrees_with_serial_and_research_reference(research_case, workers):
    y, reference, windows, serial = research_case
    result = IMF(SmoothAbs(H=0.4), workers=workers).decompose(y, window_sizes=windows)
    np.testing.assert_allclose(result.imfs, reference, rtol=0, atol=2e-6)
    np.testing.assert_allclose(result.imfs, serial.imfs, rtol=0, atol=2e-6)
    np.testing.assert_allclose(result.reconstruction, y, rtol=0, atol=1e-12)
    assert all(stage.converged for stage in result.stages)


@pytest.mark.parametrize("contrast", [Quadratic(), SmoothAbs(H=0.2)])
@pytest.mark.parametrize("n", [1, 3, 17, 63, 65])
def test_worker_count_larger_than_signal_preserves_short_signals(contrast, n):
    y = np.linspace(-1, 1, n)
    result = IMF(contrast, workers=128).decompose(y, k_max=2)
    assert result.imfs.shape[1] == n
    assert result.residual.shape == (n,)
    np.testing.assert_allclose(result.reconstruction, y, rtol=0, atol=1e-12)
    assert all(np.all(np.isfinite(component)) for component in result.imfs)
    assert len(result.stages) == result.imfs.shape[0]
    if n < 4:
        np.testing.assert_allclose(result.imfs[0], y, rtol=0, atol=1e-12)
        assert result.stages[0].converged


@pytest.mark.parametrize("path", ["score", "direct"])
def test_custom_contrast_exception_reaches_the_caller(path):
    threads = SimpleQueue()

    class BrokenScore(SmoothAbs):
        def psi(self, r):
            threads.put(current_thread())
            raise RuntimeError("custom score failed")

    class BrokenDirect(Quadratic):
        def solve(self, windows, weights):
            threads.put(current_thread())
            raise RuntimeError("custom direct solver failed")

    contrast = BrokenScore(H=0.2) if path == "score" else BrokenDirect()
    with pytest.raises(RuntimeError, match="custom .* failed"):
        IMF(contrast, workers=4).decompose(np.arange(256.0), window_sizes=[5])
    assert not threads.empty()
    while not threads.empty():
        thread = threads.get_nowait()
        assert thread is not main_thread()
        assert not thread.is_alive()


@pytest.mark.parametrize("large_first", [False, True])
@pytest.mark.parametrize("max_iter", [1, 2])
def test_convergence_uses_each_chunks_own_signal_scale(large_first, max_iter):
    large = np.full(128, 1e9)
    small = np.tile([0.0, 0.0, 10.0, 2.0, 0.0, 1.0, 0.0, 0.0], 16)
    y = np.concatenate((large, small) if large_first else (small, large))
    result = IMF(
        SmoothAbs(H=0.2), workers=2, boundary="edge", max_iter=max_iter, tol=1e-6
    ).decompose(y, window_sizes=[5])
    stage = result.stages[0]
    component = result.imfs[0]
    small_component = component[128:] if large_first else component[:128]
    assert stage.iterations == max_iter
    assert stage.final_max_delta > 1e-6 * (1 + np.max(np.abs(small_component)))
    assert stage.final_max_delta <= 1e-6 * (1 + np.max(np.abs(component)))
    assert not stage.converged
    if max_iter == 1:
        initial = np.median(sliding_window_view(np.pad(y, 2, mode="edge"), 5), axis=1)
        assert stage.final_max_delta == pytest.approx(
            np.max(np.abs(component - initial)), rel=0, abs=1e-15
        )
