from pathlib import Path

import numpy as np
import pytest

from pimf import IMF, SmoothAbs, make_window_schedule
from pimf.kernels import epanechnikov, squared_triangle


@pytest.mark.parametrize(
    ("case", "H", "kernel", "schedule"),
    [
        ("your_gd_1000", 0.4, squared_triangle, {"window_sizes": make_window_schedule(1000)}),
        ("quantlet_zero_2000", 1.0, epanechnikov, {"h1": 0.2, "k_max": 8}),
    ],
)
def test_research_scalar_solver_reference(case, H, kernel, schedule):
    with np.load(Path(__file__).parent / "data" / "solver_reference.npz") as fixture:
        y = fixture[case + "_y"]
        expected = fixture[case + "_reference"]
    result = IMF(SmoothAbs(H), kernel).decompose(y, **schedule)
    np.testing.assert_allclose(result.imfs, expected, rtol=0, atol=2e-6)
    np.testing.assert_allclose(result.reconstruction, y, rtol=0, atol=1e-12)
    assert all(stage.converged for stage in result.stages)
