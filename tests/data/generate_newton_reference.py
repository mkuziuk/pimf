"""Generate roots from the research notebook's score tables, without pimf.

Run: uv run python tests/data/generate_newton_reference.py /path/to/imf
"""

import ast
import json
import sys
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

notebook = Path(sys.argv[1]) / "experiments/gd-irmf/gd_irmf.ipynb"
functions = {"erf_approx", "make_lookup_grid"}
namespace = {"np": np, "SQRT_2": np.sqrt(2.0)}
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


def roots(windows, weights, H, lookup=True):
    """Bisect both edges of the zero set, including flat score intervals."""
    positive = weights > 0
    windows, weights = windows[:, positive], weights[positive]
    weights = weights / weights.sum()
    boundaries = []
    for right_edge in (False, True):
        lo, hi = windows.min(axis=1), windows.max(axis=1)
        for _ in range(80):
            mid = 0.5 * lo + 0.5 * hi
            residuals = windows - mid[:, None]
            score = (
                np.interp(residuals / H, grid["z"], grid["score"])
                if lookup
                else namespace["erf_approx"](residuals / (np.sqrt(2.0) * H))
            )
            score = np.sum(weights * score, axis=1)
            positive_score = score >= 0 if right_edge else score > 0
            lo = np.where(positive_score, mid, lo)
            hi = np.where(positive_score, hi, mid)
        boundaries.append(0.5 * lo + 0.5 * hi)
    return 0.5 * boundaries[0] + 0.5 * boundaries[1]


fixture = {}
fixture["slope_z"] = grid["z"]
fixture["slope_lookup"] = np.r_[256 * np.diff(grid["score"]), 0.0]
fixture["slope_direct"] = np.sqrt(2 / np.pi) * grid["exp"]
rng = np.random.default_rng(20261021)
for h_index, H in enumerate((0.01, 1.0, 10.0)):
    for size in (3, 5, 7, 65, 257, 1025):
        weights = rng.uniform(0.1, 1, size)
        if size > 3:
            weights[[0, -1]] = 0
        weights /= weights.sum()
        windows = rng.normal(size=(6, size)) * H
        windows[1] *= 10
        windows[2] = 3.5 * H
        windows[3] = 0
        windows[3, size // 2] = 100 * H
        windows[4] += np.where(np.arange(size) < size // 2, -25 * H, 25 * H)
        windows[5] += 30 * H
        prefix = f"h{h_index}_w{size}_"
        fixture[prefix + "windows"] = windows
        fixture[prefix + "weights"] = weights
        for lookup in (False, True):
            fixture[prefix + f"roots_{lookup}"] = roots(windows, weights, H, lookup)

flat_windows = np.array([[-100.0, -90.0, 100.0]])
flat_weights = np.array([0.25, 0.25, 0.5])
fixture["flat_windows"] = flat_windows
fixture["flat_weights"] = flat_weights
fixture["flat_root"] = roots(flat_windows, flat_weights, 1.0)
spike_weights = 0.75 * np.maximum(0.0, 1 - np.abs(np.arange(-3, 4) / (2 * np.sqrt(2)))) ** 2
spike_weights /= spike_weights.sum()
fixture["spike_weights"] = spike_weights
fixture["spike_window"] = np.array([[0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0]])
fixture["spike_root"] = roots(fixture["spike_window"], spike_weights, 1.0)

n = 2048
t = np.arange(n)
rng = np.random.default_rng(20261020 + n)
x = (1 + 8 * np.exp(-0.5 * ((t - 0.45 * n) / (0.06 * n)) ** 2)) * np.sin(2 * np.pi * 0.018 * t)
x += rng.normal(0, 0.08, n)
y = (x - np.median(x)) / (1.4826 * np.median(np.abs(x - np.median(x))))
H = 2 * np.median(np.abs(np.diff(y))) / (0.6745 * np.sqrt(2))
fixture["bursty_y"], fixture["bursty_H"] = y, H
residual = y.copy()
components, sizes, bandwidths = [], [], []
h = 0.25
while h >= 1.5 / n:
    radius = h * n
    if abs(radius - round(radius)) < 1e-9:
        radius = round(radius)
    support = int(np.ceil(radius))
    size = 2 * support + 1
    weights = 0.75 * np.maximum(0.0, 1 - np.abs(np.arange(-support, support + 1) / radius)) ** 2
    weights /= weights.sum()
    windows = sliding_window_view(np.pad(residual, support, mode="wrap"), size)
    component = roots(windows, weights, H)
    components.append(component)
    sizes.append(size)
    bandwidths.append(h)
    residual -= component
    h /= np.sqrt(2)
fixture["bursty_imfs"] = components
fixture["bursty_residual"] = residual
fixture["bursty_window_sizes"] = sizes
fixture["bursty_bandwidths"] = bandwidths
np.savez_compressed(Path(__file__).with_name("newton_reference.npz"), **fixture)
