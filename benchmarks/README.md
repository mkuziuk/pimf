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

Compare GD and Newton on a noisy signal with a burst of high amplitude:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  uv run python benchmarks/newton.py --output /tmp/newton_results.json
```

Use `--sizes` and `--repeats` to change the workload. The script uses one
worker, `SmoothAbs` lookup evaluation and the squared-triangular kernel, with
all stages from `h1=0.25` down to `h_min=1.5/L` at `a=sqrt(2)`. It compares
GD at `max_iter=60, tol=1e-6` with Newton at `max_iter=100, tol=1e-12`.
After one warmup, three runs alternate solver order. Imports and input
creation are outside the timed region. JSON output records individual wall
and CPU timings, environment details and stage diagnostics.

Both solvers use `on_unconverged="ignore"` to keep warning handling outside
the timed region. Convergence diagnostics are retained. GD reports the score
from its last evaluated iterate, before the final update, without an extra
score pass. Newton evaluates scores and slopes and may probe a positional
bracket or bisect. Its starting point is the weighted mean; a zero initial
slope triggers a weighted-median restart.

Measured with pimf 0.5.1 on a shared Apple M4 Max, macOS 26.6.2, Python
3.13.13 and NumPy 2.5.1, with the thread limits above, median wall times over
two runs of the script were:

- L=1024: GD 0.4750 s, Newton 0.0522 s, a 9.09x speedup.
- L=2048: GD 2.0464 s, Newton 0.1780 s, a 11.50x speedup.
- L=4096: GD 9.2097 s, Newton 0.6294 s, a 14.63x speedup.

Other jobs were running, so compare ratios rather than absolute times. Runs of
pimf 0.5.0 alternating with these took 0.1127 s, 0.4336 s and 2.0083 s for
Newton, so 0.5.1 is 2.16x, 2.44x and 3.19x faster there, with bit-identical
results. Every Newton window converged in at most six iterations, with no
bisection fallbacks on these three signals. The largest absolute returned
score was at most 1.13e-15. Default GD capped 7 of 15, 9 of 17 and 11 of 19
stages, respectively. At L=2048, the component-and-residual stack differed
from Newton by 14.04% in relative L2. Compare timings on your own signals;
iteration counts alone do not measure solver cost or root accuracy.

The separate seven-sample spike window in the JSON output uses
`squared_triangle.weights(7, bandwidth=2*sqrt(2))` and `SmoothAbs(1)`.
It reaches 2.6825936366767804 in 11 iterations with five bisection fallbacks.
Across all 64 windows of that stage, there are seven fallbacks and no
unfinished windows.

## Per-stage Newton benchmark

Time each Newton stage of the L=2048 bursty decomposition with:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  uv run python benchmarks/newton_stages.py --output /tmp/newton_stages.json
```

Use `--length` and `--repeats` to change the workload. The script needs only
NumPy and pimf 0.5.0 or later, so the same command can time the previous
release:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  uv run --isolated --no-project --with pimf==0.5.0 \
  python benchmarks/newton_stages.py --output /tmp/newton_stages_050.json
```

Under the conditions above, the 17 stage solves took 169 ms in total with
0.5.1 and 430 ms with 0.5.0, averaging two alternating runs of five repeats.
Wide stages gained the most: the 1025-sample stage went from 134 ms to 47 ms.
On the narrowest, five-sample stage, NumPy call overhead dominates and the time
went from 1.16 ms to 0.73 ms.

## Outlier windows

The seven-sample window `[0.1, -0.2, 0, big, 0.3, -0.1, 0.2]` with uniform
weights `1/7` and `SmoothAbs(0.2)` returns approximately 0.10780207441813
for `big` in `{1e15, 1e30, 9.97e36, 3.4e38}`. Research-score fixtures use
independent bisection to resolve the roots. For `[0, big, 0, big, 0]`, the
squared-triangular weights are `[0, 1, 4, 1, 0]/6`; the lookup root is
approximately 0.13489813577792 at both 1e15 and 3.4e38.

Wide-bracket fallbacks bisect the ordered float64 values, bounding these
searches to at most 64 ordered halvings. Arithmetic midpoints refine nearby
bounds. Small-H windows can therefore need dozens of iterations even when
ordinary noisy windows finish in a few. Both kinds of step and the search
for flat zero-score boundaries count toward `max_iter`. Check
`unconverged_windows` when lowering that cap. Constant `±finfo.max` windows
return their input, zero residual and zero iterations.

Default GD preserves the research components, residuals, iteration counts,
final updates and convergence flags. Its small-update stopping test does
not bound distance to a score root. Use Newton when root position matters,
and inspect the returned diagnostics for either solver.
