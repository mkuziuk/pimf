"""Compare safeguarded Newton with default GD on a bursty signal.

Run: uv run python benchmarks/newton.py --output /tmp/newton_results.json
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from pimf import IMF, SmoothAbs, __version__, squared_triangle
from pimf._newton import _newton_fit_windows

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048, 4096])
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--output", type=Path, default=Path("newton_results.json"))
args = parser.parse_args()
if min(args.sizes + [args.repeats]) < 1:
    parser.error("sizes and repeats must be positive")

spike = _newton_fit_windows(
    np.array([[0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0]]),
    squared_triangle.weights(7, bandwidth=2 * np.sqrt(2)),
    SmoothAbs(1),
    100,
    1e-12,
)
report = {
    "environment": {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pimf": __version__,
    },
    "protocol": {
        "repeats": args.repeats,
        "warmups": 1,
        "order": "alternating",
        "workers": 1,
        "on_unconverged": "ignore",
        "gd": {"max_iter": 60, "tol": 1e-6},
        "newton": {"max_iter": 100, "tol": 1e-12},
        "schedule": {"h1": 0.25, "a": "sqrt(2)", "k_max": None, "h_min": "1.5/L"},
        "timing": "full decomposition; excludes imports and input generation",
    },
    "spike_window": {
        "root": float(spike[0][0]),
        "iterations": spike[1],
        "final_max_delta": spike[2],
        "unconverged_windows": spike[3],
        "bisection_fallbacks": spike[4],
        "max_abs_score": spike[5],
    },
    "cases": [],
}
for n in args.sizes:
    rng = np.random.default_rng(20261020 + n)
    t = np.arange(n)
    x = (1 + 8 * np.exp(-0.5 * ((t - 0.45 * n) / (0.06 * n)) ** 2)) * np.sin(2 * np.pi * 0.018 * t)
    x += rng.normal(0, 0.08, n)
    y = (x - np.median(x)) / (1.4826 * np.median(np.abs(x - np.median(x))))
    H = 2 * np.median(np.abs(np.diff(y))) / (0.6745 * np.sqrt(2))
    methods = {
        solver: IMF(SmoothAbs(H), solver=solver, on_unconverged="ignore")
        for solver in ("gd", "newton")
    }
    schedule = {"k_max": None, "h_min": 1.5 / n}
    outputs = {name: method.decompose(y, **schedule) for name, method in methods.items()}
    assert all(stage.converged for stage in outputs["newton"].stages)
    samples = {name: [] for name in methods}
    cpu_samples = {name: [] for name in methods}
    names = list(methods)
    for repeat in range(args.repeats):
        for name in names if repeat % 2 == 0 else names[::-1]:
            start, cpu_start = time.perf_counter(), time.process_time()
            methods[name].decompose(y, **schedule)
            cpu_samples[name].append(time.process_time() - cpu_start)
            samples[name].append(time.perf_counter() - start)
    medians = {name: float(np.median(values)) for name, values in samples.items()}
    stages = outputs["newton"].stages
    stack_gd = np.vstack([outputs["gd"].imfs, outputs["gd"].residual])
    stack_newton = np.vstack([outputs["newton"].imfs, outputs["newton"].residual])
    case = {
        "L": n,
        "H": float(H),
        "median_seconds": medians,
        "median_cpu_seconds": {name: float(np.median(v)) for name, v in cpu_samples.items()},
        "speedup": medians["gd"] / medians["newton"],
        "samples_seconds": samples,
        "samples_cpu_seconds": cpu_samples,
        "gd_unconverged_stages": sum(not stage.converged for stage in outputs["gd"].stages),
        "newton_unconverged_windows": sum(stage.unconverged_windows for stage in stages),
        "newton_max_iterations": max(stage.iterations for stage in stages),
        "newton_bisection_fallbacks": sum(stage.bisection_fallbacks for stage in stages),
        "newton_fallbacks_per_window": sum(stage.bisection_fallbacks for stage in stages)
        / (n * len(stages)),
        "newton_max_abs_score": max(stage.max_abs_score for stage in stages),
        "stack_relative_l2_difference": float(
            np.linalg.norm(stack_gd - stack_newton) / np.linalg.norm(stack_newton)
        ),
        "stages": [
            {
                "window_size": stage.window_size,
                "iterations": stage.iterations,
                "final_max_delta": stage.final_max_delta,
                "unconverged_windows": stage.unconverged_windows,
                "bisection_fallbacks": stage.bisection_fallbacks,
                "max_abs_score": stage.max_abs_score,
            }
            for stage in stages
        ],
    }
    report["cases"].append(case)
    print(
        f"L={n}: gd {medians['gd']:.4f}s, newton {medians['newton']:.4f}s, "
        f"{case['speedup']:.2f}x; max iterations {case['newton_max_iterations']}, "
        f"fallbacks/window {case['newton_fallbacks_per_window']:.3f}",
        flush=True,
    )
args.output.write_text(json.dumps(report, indent=2) + "\n")
