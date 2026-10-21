# pimf

pimf is a Python library for intrinsic multiscale filtering of one-dimensional
signals, built on NumPy. It extracts components from coarse to fine scales using
linear or robust local fits, with optional parallel execution. You can choose
predefined or custom kernels and contrasts, and control window sizes and the
number of components.

[Documentation and examples](https://mkuziuk.github.io/pimf/)

`IMF` warns with a `RuntimeWarning` when a stage hits `max_iter` with
unconverged windows. For gradient descent, this means windows are still moving
above the stopping threshold, `tol * (1 + max(abs(x_chunk)))`, at the final
estimates. Non-finite updates also count as unconverged. Set
`on_unconverged="raise"` to raise `RuntimeError`, or `on_unconverged="ignore"`
to silence it. A small GD update and `converged=True` do not bound distance to
the score root.

The default emits one warning per unconverged stage, so long schedules and
repeated decompositions can produce many warnings. For batch runs, use
`on_unconverged="ignore"` and inspect `StageInfo.converged` and
`unconverged_windows`, or filter these warnings with Python's `warnings` module.

For robust fits where GD takes too long to reach the weighted score root,
opt into safeguarded Newton:

```python
from pimf import IMF, SmoothAbs

method = IMF(SmoothAbs(H=0.2), solver="newton", on_unconverged="raise")
```

Newton uses pure NumPy and defaults to `max_iter=100` and `tol=1e-12`;
explicit values override them. The default `solver="gd"` preserves research
numerical results and defaults of 60 iterations and `1e-6`. Newton targets the
same score equation. Differences beyond stopping roundoff occur where GD has
not reached the root, even if its small-update test reports convergence. On
flat zero-score intervals in either lookup mode, Newton locates both boundaries
and returns their midpoint; GD can choose another zero.

A small Newton step alone cannot stop a window. The score must bracket a root
at `x ± tol * (1 + abs(x))`, or the bracket must be narrow enough at its edge
scale. Flat interval boundaries use `tol * (1 + max(abs(edges)))`. When
`H << tol * abs(x)`, Newton resolves the root in position to about
`tol * (1 + abs(x))`, but `max_abs_score` can remain large because the score
is steep. This is expected. Newton stops windows independently and returns
bit-identical values and diagnostics across worker counts for the built-in
contrasts. GD chunks stop independently, so changing workers can change GD
results near its tolerance.

Each Newton iteration evaluates the score and its slope, with extra score
probes or bisections when needed. On the issue #3 bursty signals with 1024,
2048 and 4096 samples, it was 4.41x, 5.12x and 5.20x faster than default GD
with one worker. See the [solver benchmark](benchmarks/README.md) for measured
costs, machine details and reproduction.

Custom convex contrasts need a vectorized `psi_prime(r)` for Newton, returning
the slope of `psi(r)` in raw residual units. `SmoothAbs.psi_prime` includes
`1/H`; lookup mode uses the piecewise-linear score's slope, and `lookup=False`
uses the Gaussian slope while preserving the research `erf_approx` score.
A contrast with `solve()` keeps its closed form for either solver.

Each result contains per-stage diagnostics:

| `StageInfo` field | Meaning |
| --- | --- |
| `stage` | Stage number, starting at one. |
| `window_size` | Odd support size in samples. |
| `bandwidth` | Normalized window half-width. |
| `iterations` | Maximum iteration count across windows or GD chunks. |
| `final_max_delta` | Largest final absolute update, propagating NaN. |
| `converged` | Whether `unconverged_windows` is zero. |
| `unconverged_windows` | GD rows failing their chunk's final update threshold, or Newton rows still active at the cap. Non-finite Newton scores or estimates remain unconverged. Counts sum across workers. |
| `bisection_fallbacks` | Total Newton bisections, including both flat-boundary searches; zero for GD and direct fits. |
| `max_abs_score` | Largest absolute normalized weighted score at the returned estimates, for GD and Newton. Maxima across workers propagate NaN. |

GD computes `max_abs_score` with one extra score evaluation after fitting;
its updates, stopping iterations and final deltas are unchanged. Direct fits
report one iteration, zero unconverged windows and NaN for `final_max_delta`
and `max_abs_score`. Newton follows the same `on_unconverged` policy as GD.

`SmoothAbs(H)` uses the research 4097-point uniform-grid lookup introduced in
0.4.0. Use `SmoothAbs(H, lookup=False)` for direct evaluation. Reducing solver
tolerance does not reduce lookup approximation error.

[Release notes](CHANGELOG.md)
