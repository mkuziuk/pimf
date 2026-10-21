# pimf

pimf is a Python library for intrinsic multiscale filtering of one-dimensional
signals, built on NumPy. It extracts components from coarse to fine scales using
linear or robust local fits, with optional parallel execution. You can choose
predefined or custom kernels and contrasts, and control window sizes and the
number of components.

[Documentation and examples](https://mkuziuk.github.io/pimf/)

In version 0.4.0, `SmoothAbs(H)` uses a precomputed uniform-grid lookup for
faster robust fits. Use `SmoothAbs(H, lookup=False)` to reproduce the previous
direct-evaluation path. Lookup results are numerically close, but need not be
bit-for-bit identical to earlier releases.

`IMF` warns with a `RuntimeWarning` when a stage hits `max_iter` with
unconverged windows. Set `on_unconverged="raise"` to raise `RuntimeError`, or
`on_unconverged="ignore"` to silence it. Each `StageInfo` reports
`unconverged_windows`, summed across workers, and is `converged` only when that
count is zero. For GD, each row uses the same threshold as its solver chunk,
`tol * (1 + max(abs(x)))`, evaluated at the final estimates. GD preserves the
research numerical results and stopping iterations. Its `converged=True` does
not bound distance to the score root. Non-finite updates also count as unconverged.
For batch runs, use `on_unconverged="ignore"` and inspect `StageInfo.converged`
and `unconverged_windows`, or filter these warnings with Python's `warnings`
module. The default emits one warning per unconverged stage, so long schedules
and repeated decompositions can produce many warnings.

For robust fits that converge to the weighted score root, opt into safeguarded
Newton:

```python
from pimf import IMF, SmoothAbs

method = IMF(SmoothAbs(H=0.2), solver="newton")
```

Newton uses pure NumPy, stops each window independently, and gives the same
results for every `workers` count. Its defaults are `max_iter=100` and
`tol=1e-12`; explicit values override them. The default `solver="gd"` retains
its existing numerical results and defaults of 60 iterations and `1e-6`.
Newton targets the same score equation. Differences beyond stopping roundoff
occur where GD has not reached the root, even if its small-update test reports
convergence. On flat zero-score intervals Newton chooses the midpoint, which
can also differ from GD's choice of zero.

Custom contrasts need `psi_prime(r)` for Newton, in the same raw residual units
as `psi(r)`. A contrast with `solve()` keeps its closed form for either solver.
Newton counts windows still active at its iteration cap and follows the same
`on_unconverged` policy as GD. Its stage diagnostics also report
`bisection_fallbacks` and `max_abs_score`.
See the [solver benchmark](benchmarks/README.md) for timings and reproduction,
and the [release notes](CHANGELOG.md) for all unreleased changes.
