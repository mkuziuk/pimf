"""Compare SmoothAbs evaluation modes on the research signal.

Run with: uv run python benchmarks/contrast_lookup.py --output results.json
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from pimf import IMF, SmoothAbs, __version__

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 2000, 4000, 8000])
parser.add_argument("--workers", nargs="+", type=int, default=[1, 4])
parser.add_argument("--repeats", type=int, default=5)
parser.add_argument("--output", type=Path, default=Path("contrast_lookup_results.json"))
args = parser.parse_args()
if min(args.sizes + args.workers + [args.repeats]) < 1:
    parser.error("sizes, workers and repeats must be positive")

results = {
    "environment": {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pimf": __version__,
    },
    "protocol": {
        "repeats": args.repeats,
        "warmups": 1,
        "order": "alternating",
        "H": 0.4,
        "seed": 777,
        "max_iter": 60,
        "tol": 1e-6,
        "boundary": "wrap",
        "kernel": "squared_triangle",
        "timing": "full decomposition; excludes input generation and imports/table construction",
    },
    "cases": [],
}
base_windows = np.array([501, 355, 251, 177, 125, 89, 63, 45, 31])
for n in args.sizes:
    t = np.arange(n) / n
    slow = 0.6 * np.sin(2 * np.pi * t)
    medium = 0.25 * np.sin(12 * np.pi * t)
    bump = 0.8 * np.exp(-((t - 0.55) ** 2) / (2 * 0.015**2))
    trend = 0.5 * (t - 0.5)
    clean = slow + medium + bump + trend
    rng = np.random.default_rng(777)
    gaussian = rng.normal(0, 0.2, n)
    mask = rng.random(n) < 0.2
    contamination = mask * rng.exponential(0.2, n)
    contamination *= rng.choice([-1, 1], n)
    y = clean + gaussian + contamination
    if n == 1000:
        fixture = Path(__file__).parents[1] / "tests/data/lookup_reference.npz"
        with np.load(fixture) as data:
            np.testing.assert_allclose(y, data["y"], rtol=0, atol=1e-15)
    windows = (2 * np.rint((base_windows // 2) * n / 1000).astype(int) + 1).tolist()
    for workers in args.workers:
        print(f"n={n}, workers={workers}", flush=True)
        methods = {
            "direct": IMF(SmoothAbs(0.4, lookup=False), workers=workers),
            "lookup": IMF(SmoothAbs(0.4), workers=workers),
        }
        outputs = {
            name: method.decompose(y, window_sizes=windows) for name, method in methods.items()
        }
        np.testing.assert_allclose(
            outputs["direct"].imfs, outputs["lookup"].imfs, rtol=0, atol=2e-6
        )
        samples = {name: [] for name in methods}
        names = list(methods)
        for repeat in range(args.repeats):
            for name in names if repeat % 2 == 0 else names[::-1]:
                start = time.perf_counter()
                methods[name].decompose(y, window_sizes=windows)
                elapsed = time.perf_counter() - start
                samples[name].append(elapsed)
                print(f"  {name}: {elapsed:.6f} s", flush=True)
        case = {
            "n": n,
            "workers": workers,
            "windows": windows,
            "median_s": {name: float(np.median(values)) for name, values in samples.items()},
            "samples_s": samples,
            "max_abs_difference": max(
                float(np.max(np.abs(outputs["direct"].imfs - outputs["lookup"].imfs))),
                float(np.max(np.abs(outputs["direct"].residual - outputs["lookup"].residual))),
            ),
            "iterations": {
                name: [stage.iterations for stage in result.stages]
                for name, result in outputs.items()
            },
        }
        results["cases"].append(case)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
