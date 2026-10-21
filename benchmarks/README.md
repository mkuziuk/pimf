# Contrast benchmark

Compare the direct and lookup evaluation modes with:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  uv run python benchmarks/contrast_lookup.py --output /tmp/contrast_lookup_results.json
```

Use `--sizes`, `--workers` and `--repeats` to change the workload. The JSON output
contains individual timings, medians, window sizes, iteration counts and output
differences. Include measured results and the machine configuration in the pull
request description.

## Newton benchmark

Compare the solvers on the bursty signal from issue #3:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  uv run python benchmarks/newton.py --output /tmp/newton_results.json
```

The script uses one worker and the full schedule from `h1=0.25` down to
`h_min=1.5/L`, with `a=sqrt(2)`. It compares GD at `max_iter=60, tol=1e-6`
with Newton at `max_iter=100, tol=1e-12`. After one warmup, three runs alternate
solver order. Input generation and imports are outside the timed region.
JSON output records all timings, environment details and Newton stage diagnostics.
GD includes its final score diagnostic evaluation; Newton includes the score
probes that verify positional accuracy before accepting a small step.
Both solvers use `on_unconverged="ignore"` to keep warnings for intentionally
capped GD stages out of the timed region; convergence diagnostics are retained.

Measured on an Apple M4 Max, macOS 26.6.2, Python 3.13.13 and NumPy 2.5.1,
with the thread limits above, median wall times were:

- L=1024: GD 0.4101 s, Newton 0.0930 s, a 4.41x speedup.
- L=2048: GD 1.5620 s, Newton 0.3052 s, a 5.12x speedup.
- L=4096: GD 6.3111 s, Newton 1.2129 s, a 5.20x speedup.

Every Newton window converged in at most six iterations, with no bisection
fallbacks on these three signals. Their largest absolute returned scores
were at most 1.13e-15. Default GD capped 7 of 15, 9 of 17 and 11 of 19 stages,
respectively. At L=2048, its component-and-residual stack differed from
Newton by 14.04% in relative L2, reproducing issue #3.

The separate spike window from issue #3 reaches 2.682593636676779 in ten
iterations with four bisection fallbacks. Across all 64 windows of that stage,
there are 20 fallbacks and no unfinished windows. These figures describe this
workload and machine; compare timings on your own signals.
