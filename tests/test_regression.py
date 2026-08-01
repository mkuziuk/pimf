"""End-to-end regression against the research example.

Setup mirrors experiments/robust-gradient-descent/robust_gradient_descent_imf.ipynb:
n=1000, sigma=0.1, contamination p=0.2 scale=2.0 (one-sided), seed 777,
windows [151, 107, 75, 53, 37, 27, 19, 13], robust h = 2 * sigma = 0.2
(the winner of the notebook's H grid).

The metric expectations follow the notebook's recorded results: the robust
decomposition wins on component-wise error (the contamination it rejects ends
up in its residual, so a total-MSE-with-residual comparison favors linear —
by design, not by defect).
"""

import numpy as np
import pytest
from conftest import gen_signal, generate_observation

from pimf import linear_imf, robust_imf

WINDOW_SIZES = [151, 107, 75, 53, 37, 27, 19, 13]
SIGMA = 0.1
H = 0.2


@pytest.fixture(scope="module")
def signals():
    t = np.linspace(0.0, 1.0, 1000)
    x_clean = gen_signal(t)
    rng = np.random.default_rng(777)
    y_noisy = generate_observation(
        x_clean, sigma=SIGMA, contamination_prob=0.2, contamination_scale=2.0, rng=rng
    )
    return x_clean, y_noisy


@pytest.fixture(scope="module")
def decompositions(signals):
    x_clean, y_noisy = signals
    return {
        "linear_noisy": linear_imf(y_noisy, window_sizes=WINDOW_SIZES),
        "linear_clean": linear_imf(x_clean, window_sizes=WINDOW_SIZES),
        "robust_noisy": robust_imf(y_noisy, H, window_sizes=WINDOW_SIZES),
        "robust_clean": robust_imf(x_clean, H, window_sizes=WINDOW_SIZES),
    }


def test_reconstructions(signals, decompositions):
    x_clean, y_noisy = signals
    assert np.max(np.abs(decompositions["linear_noisy"].reconstruction - y_noisy)) < 1e-10
    assert np.max(np.abs(decompositions["robust_noisy"].reconstruction - y_noisy)) < 1e-10
    assert np.max(np.abs(decompositions["robust_clean"].reconstruction - x_clean)) < 1e-10


def test_robust_beats_linear_on_components(decompositions):
    # Noisy-vs-clean error per component, excluding the residual: the robust
    # fit rejects contamination into its residual, the linear fit spreads it
    # over the components. Cross-checked: robust ~0.0014 vs linear ~0.023.
    def component_mse(noisy, clean):
        return float(np.mean((noisy.imfs - clean.imfs) ** 2))

    robust = component_mse(decompositions["robust_noisy"], decompositions["robust_clean"])
    linear = component_mse(decompositions["linear_noisy"], decompositions["linear_clean"])
    assert robust < linear


def test_robust_beats_linear_on_mean_mae(decompositions):
    # Mean MAE over components plus residual — the aggregate where the
    # notebook's benchmark records the robust win (0.058 vs 0.128 here).
    def mean_mae(noisy, clean):
        rows = [np.mean(np.abs(a - b)) for a, b in zip(noisy.imfs, clean.imfs, strict=True)]
        rows.append(np.mean(np.abs(noisy.residual - clean.residual)))
        return float(np.mean(rows))

    robust = mean_mae(decompositions["robust_noisy"], decompositions["robust_clean"])
    linear = mean_mae(decompositions["linear_noisy"], decompositions["linear_clean"])
    assert robust < linear


def test_golden_spot_values(decompositions):
    # Frozen from the implementation after it was verified bit-identical to
    # the reference notebook on this exact case (cross-check script).
    result = decompositions["robust_noisy"]
    golden_imfs = {
        (0, 0): 0.07303580151881464,
        (0, 500): 0.11423344217234899,
        (3, 250): -0.011978870909632695,
        (5, 777): 0.034920007932709995,
        (7, 999): 0.08526826398194841,
    }
    for (stage, index), value in golden_imfs.items():
        assert abs(result.imfs[stage, index] - value) < 1e-12
    assert abs(result.residual[123] - (-0.031524706119654425)) < 1e-12


def test_stage_iterations_match_reference(decompositions):
    # Cross-checked against the reference implementation (bit-identical run):
    # late stages legitimately hit the max_iter cap on this contaminated case.
    iterations = [info.iterations for info in decompositions["robust_noisy"].stages]
    assert iterations == [39, 18, 18, 19, 24, 30, 60, 60]
