"""Run with the imf checkout path to regenerate lookup_reference.npz."""

import ast
import json
import sys
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

notebook = Path(sys.argv[1]) / "experiments/gd-irmf/gd_irmf.ipynb"
functions = {
    "erf_approx",
    "epanechnikov_weights",
    "make_lookup_grid",
    "_interp_like",
    "lookup_score",
    "lookup_exp",
    "smooth_abs_rho_lookup",
    "robust_gd_fit_windows",
}
namespace = {"np": np, "SQRT_2": np.sqrt(2.0), "SQRT_2_OVER_PI": np.sqrt(2.0 / np.pi)}
for cell in json.loads(notebook.read_text())["cells"]:
    if cell["cell_type"] != "code":
        continue
    for node in ast.parse("".join(cell["source"])).body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            exec(
                compile(ast.Module(body=[node], type_ignores=[]), str(notebook), "exec"), namespace
            )
assert functions <= namespace.keys()
grid = namespace["make_lookup_grid"]()
namespace["lookup_grid"] = grid
z = np.unique(
    np.r_[
        np.linspace(-10, 10, 20001),
        grid["z"],
        (grid["z"][:-1] + grid["z"][1:]) / 2,
        np.nextafter([-8.0, 8.0], [-np.inf, np.inf]),
    ]
)
score = namespace["lookup_score"](z, 1.0, grid)
rho = namespace["smooth_abs_rho_lookup"](z, 1.0, grid)
direct_score = namespace["erf_approx"](z / np.sqrt(2.0))
direct_rho = z * direct_score + np.sqrt(2.0 / np.pi) * np.exp(-0.5 * z**2)
np.testing.assert_allclose(score, direct_score, rtol=0, atol=1.1e-6)
np.testing.assert_allclose(rho, direct_rho, rtol=0, atol=1.6e-6)

with np.load(Path(__file__).with_name("solver_reference.npz")) as fixture:
    y = fixture["your_gd_1000_y"]
    scalar_reference = fixture["your_gd_1000_reference"]
windows = [501, 355, 251, 177, 125, 89, 63, 45, 31]
residual = y.copy()
components, iterations = [], []
for size in windows:
    weights = namespace["epanechnikov_weights"](size)
    values = sliding_window_view(np.pad(residual, size // 2, mode="wrap"), size)
    component, trace = namespace["robust_gd_fit_windows"](
        values, weights, 0.4, grid=grid, return_trace=True
    )
    components.append(component)
    iterations.append(len(trace))
    residual -= component
np.testing.assert_allclose(components, scalar_reference, rtol=0, atol=2e-6)
np.savez_compressed(
    Path(__file__).with_name("lookup_reference.npz"),
    z=z,
    score=score,
    rho=rho,
    direct_score=direct_score,
    direct_rho=direct_rho,
    y=y,
    window_sizes=windows,
    imfs=components,
    residual=residual,
    iterations=iterations,
)
