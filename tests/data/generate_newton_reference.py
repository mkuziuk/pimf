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


def roots(windows, weights, H, lookup=True, iterations=80, return_edges=False):
    """Bisect both edges of the zero set, including flat score intervals."""
    positive = weights > 0
    windows, weights = windows[:, positive].astype(float), weights[positive]
    weights = weights / weights.sum()
    boundaries = []
    for right_edge in (False, True):
        lo, hi = windows.min(axis=1), windows.max(axis=1)
        for _ in range(iterations):
            mid = 0.5 * lo + 0.5 * hi
            with np.errstate(over="ignore", invalid="ignore"):
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
    return np.array(boundaries) if return_edges else 0.5 * boundaries[0] + 0.5 * boundaries[1]


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
fixture["flat_boundaries"] = roots(flat_windows, flat_weights, 1.0, return_edges=True)
spike_weights = 0.75 * np.maximum(0.0, 1 - np.abs(np.arange(-3, 4) / (2 * np.sqrt(2)))) ** 2
spike_weights /= spike_weights.sum()
fixture["spike_weights"] = spike_weights
fixture["spike_window"] = np.array([[0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0]])
fixture["spike_root"] = roots(fixture["spike_window"], spike_weights, 1.0)


fixture["adversarial_windows"] = np.array([[-192.0, -8.0, 100.0], [-20.0, -10.0, 20.0]])
fixture["adversarial_weights"] = np.array([0.25, 0.25, 0.5])
fixture["adversarial_triangle_windows"] = np.array(
    [[0.0, -168.0, 100.0, 0.0, 0.0], [0.0, -160.0, 100.0, 0.0, 0.0]]
)
fixture["adversarial_triangle_weights"] = np.array([0.0, 0.25, 0.5, 0.25, 0.0])
fixture["tiny_H_window"] = np.array([[-3.0, 0.0, 1.0, 1.0, 1.0]])
fixture["wide_windows"] = np.array([[-A, -A, A] for A in (1e2, 1e4, 1e8)])
for lookup in (False, True):
    for prefix in ("adversarial", "adversarial_triangle"):
        fixture[prefix + f"_roots_{lookup}"] = roots(
            fixture[prefix + "_windows"], fixture[prefix + "_weights"], 1.0, lookup
        )
    fixture[f"tiny_H_root_{lookup}"] = roots(fixture["tiny_H_window"], np.ones(5), 1e-14, lookup)
    fixture[f"wide_roots_{lookup}"] = roots(
        fixture["wide_windows"], fixture["adversarial_weights"], 1.0, lookup
    )

fixture["extreme_outliers"] = np.array([1e15, 1e30, 9.97e36, 3.4e38])
fixture["extreme_windows"] = np.tile([0.1, -0.2, 0.0, 0.0, 0.3, -0.1, 0.2], (4, 1))
fixture["extreme_windows"][:, 3] = fixture["extreme_outliers"]
fixture["extreme_spikes"] = np.zeros((4, 7))
fixture["extreme_spikes"][:, 3] = fixture["extreme_outliers"]
for lookup in (False, True):
    fixture[f"extreme_roots_{lookup}"] = roots(
        fixture["extreme_windows"], np.ones(7), 0.2, lookup, iterations=256
    )
    fixture[f"extreme_spike_roots_{lookup}"] = roots(
        fixture["extreme_spikes"], spike_weights, 0.2, lookup, iterations=256
    )

fixture["stalled_window"] = np.array([[-0.0199, 1.6906, 0.6221, -1.5291, -1e40, -0.3950, -0.8795]])
fixture["stalled_weights"] = np.maximum(0.0, 1 - np.abs(np.arange(-3, 4) / 3)) ** 2
fixture["cluster_windows"] = np.array([[0, big, 0, big, 0] for big in (1e15, 3.4e38)])
fixture["cluster_weights"] = np.array([0, 1, 4, 1, 0], dtype=float) / 6
for lookup in (False, True):
    fixture[f"stalled_root_{lookup}"] = roots(
        fixture["stalled_window"], fixture["stalled_weights"], 0.01, lookup, iterations=256
    )
    fixture[f"cluster_roots_{lookup}"] = roots(
        fixture["cluster_windows"], fixture["cluster_weights"], 0.2, lookup, iterations=256
    )

rng = np.random.default_rng(20261023)
for size in (7, 15, 33):
    magnitudes = np.tile([1e10, 1e30, 1e100, 1e300, np.finfo(float).max], 2)
    magnitudes[5:] *= -1
    windows = rng.normal(size=(len(magnitudes), size))
    windows[np.arange(len(magnitudes)), rng.integers(1, size - 1, len(magnitudes))] = magnitudes
    fixture[f"sweep_{size}_windows"] = windows
    for kernel in ("uniform", "squared_triangle"):
        weights = (
            np.ones(size)
            if kernel == "uniform"
            else np.maximum(0.0, 1 - np.abs(np.arange(size) - size // 2) / (size // 2)) ** 2
        )
        fixture[f"sweep_{size}_{kernel}_weights"] = weights
        for H in (0.003, 0.01):
            for lookup in (False, True):
                fixture[f"sweep_{size}_{kernel}_{H}_{lookup}"] = roots(
                    windows, weights, H, lookup, iterations=1200
                )

rng = np.random.default_rng(3)
n = 512
t = np.arange(n)
rng.normal(0, 0.3, n)
y = np.round(4 * np.sin(2 * np.pi * t / n * 3) + rng.normal(0, 0.5, n))
y[200] = 1e30
fixture["huber_y"] = y
windows = sliding_window_view(np.pad(y, n // 4, mode="wrap"), n // 2 + 1)
weights = np.maximum(0.0, 1 - np.abs(np.arange(-n // 4, n // 4 + 1) / (n // 4))) ** 2
weights /= weights.sum()
boundaries = []
for right_edge in (False, True):
    lo, hi = windows.min(axis=1), windows.max(axis=1)
    for _ in range(256):
        mid = 0.5 * lo + 0.5 * hi
        score = np.sum(weights * np.clip(windows - mid[:, None], -0.3, 0.3), axis=1)
        positive_score = score >= 0 if right_edge else score > 0
        lo = np.where(positive_score, mid, lo)
        hi = np.where(positive_score, hi, mid)
    boundaries.append(0.5 * lo + 0.5 * hi)
fixture["huber_first_component"] = 0.5 * boundaries[0] + 0.5 * boundaries[1]

n = 2048
t = np.arange(n)
rng = np.random.default_rng(20261020 + n)
x = (1 + 8 * np.exp(-0.5 * ((t - 0.45 * n) / (0.06 * n)) ** 2)) * np.sin(2 * np.pi * 0.018 * t)
x += rng.normal(0, 0.08, n)
y = (x - np.median(x)) / (1.4826 * np.median(np.abs(x - np.median(x))))
H = 2 * np.median(np.abs(np.diff(y))) / (0.6745 * np.sqrt(2))
fixture["bursty_y"], fixture["bursty_H"] = y, H
rng = np.random.default_rng(11)
fill_y = np.sign(np.sin(2 * np.pi * t / 300 + 0.1)) + rng.normal(0, 0.01, n)
fill_y[rng.choice(n, 3, replace=False)] = 9.96921e36
fixture["fill_y"] = fill_y
for prefix, signal, smoothing_H, iterations in (("bursty", y, H, 80), ("fill", fill_y, 0.02, 256)):
    residual = signal.copy()
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
        component = roots(windows, weights, smoothing_H, iterations=iterations)
        components.append(component)
        sizes.append(size)
        bandwidths.append(h)
        residual -= component
        h /= np.sqrt(2)
    fixture[prefix + "_imfs"] = components
    fixture[prefix + "_residual"] = residual
    fixture[prefix + "_window_sizes"] = sizes
    fixture[prefix + "_bandwidths"] = bandwidths
np.savez_compressed(Path(__file__).with_name("newton_reference.npz"), **fixture)
