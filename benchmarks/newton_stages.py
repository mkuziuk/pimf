"""Time each Newton stage of a full decomposition of the bursty signal.

Run: uv run python benchmarks/newton_stages.py --output /tmp/newton_stages.json

Needs only NumPy and pimf >= 0.5.0, so it can also time an installed release:
uv run --isolated --no-project --with pimf==0.5.0 python benchmarks/newton_stages.py
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from pimf import IMF, SmoothAbs, __version__, squared_triangle
from pimf._newton import _newton_fit_windows

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--length", type=int, default=2048)
parser.add_argument("--repeats", type=int, default=5)
parser.add_argument("--output", type=Path, default=Path("newton_stages.json"))
args = parser.parse_args()
if min(args.length, args.repeats) < 1:
    parser.error("length and repeats must be positive")

n = args.length
rng = np.random.default_rng(20261020 + n)
t = np.arange(n)
x = (1 + 8 * np.exp(-0.5 * ((t - 0.45 * n) / (0.06 * n)) ** 2)) * np.sin(2 * np.pi * 0.018 * t)
x += rng.normal(0, 0.08, n)
y = (x - np.median(x)) / (1.4826 * np.median(np.abs(x - np.median(x))))
H = 2 * np.median(np.abs(np.diff(y))) / (0.6745 * np.sqrt(2))
contrast = SmoothAbs(H)
result = IMF(contrast, solver="newton", on_unconverged="ignore").decompose(
    y, k_max=None, h_min=1.5 / n
)

stages = []
residual = y.copy()
for info, component in zip(result.stages, result.imfs, strict=True):
    weights = squared_triangle.weights(info.window_size, bandwidth=info.bandwidth * n)
    windows = sliding_window_view(
        np.pad(residual, info.window_size // 2, mode="wrap"), info.window_size
    )
    _newton_fit_windows(windows, weights, contrast, 100, 1e-12)
    seconds = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        _newton_fit_windows(windows, weights, contrast, 100, 1e-12)
        seconds.append(time.perf_counter() - start)
    stages.append(
        {
            "stage": info.stage,
            "window_size": info.window_size,
            "iterations": info.iterations,
            "bisection_fallbacks": info.bisection_fallbacks,
            "median_seconds": float(np.median(seconds)),
            "samples_seconds": seconds,
        }
    )
    print(
        f"stage {info.stage:2d}, window {info.window_size:5d}: "
        f"{stages[-1]['median_seconds'] * 1e3:8.2f} ms, {info.iterations} iterations",
        flush=True,
    )
    residual = residual - component
total = sum(stage["median_seconds"] for stage in stages)
print(f"total: {total * 1e3:.1f} ms")
report = {
    "environment": {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pimf": __version__,
    },
    "protocol": {
        "L": n,
        "H": float(H),
        "repeats": args.repeats,
        "warmups": 1,
        "newton": {"max_iter": 100, "tol": 1e-12},
        "schedule": {"h1": 0.25, "a": "sqrt(2)", "k_max": None, "h_min": "1.5/L"},
        "timing": "one stage solve on the stage's residual windows; excludes padding",
    },
    "total_median_seconds": total,
    "stages": stages,
}
args.output.write_text(json.dumps(report, indent=2) + "\n")
