# Changelog

## Unreleased

- Add `IMF(on_unconverged="warn" | "raise" | "ignore")`, defaulting to a
  `RuntimeWarning` when a stage hits `max_iter` with unconverged windows. The
  message includes the stage, window size and total unconverged window count.
  The policy also applies with multiple workers.
- Add `StageInfo.unconverged_windows`. For GD, it counts final row updates
  above the solver chunk's stopping threshold, `tol * (1 + max(abs(x)))`.
  GD preserves research numerical results and stopping iterations.
  `converged=True` means this count is zero; for GD, it does not bound distance
  to the score root. Non-finite updates count as unconverged, and parallel
  diagnostics propagate NaN.
- Batch users can silence per-stage warnings with `on_unconverged="ignore"`
  and inspect `StageInfo.converged` / `unconverged_windows`, or use a Python
  `warnings` filter. Long schedules and repeated runs can emit many warnings.
- Add opt-in `IMF(..., solver="newton")` with pure-NumPy safeguarded Newton
  steps, bisection fallbacks, independent window stopping, and deterministic
  midpoints for flat zero-score intervals. Results do not depend on `workers`.
- Add `Contrast.psi_prime(r)`, with implementations for `Quadratic` and
  `SmoothAbs`. Lookup slopes follow the piecewise-linear score exactly;
  direct evaluation uses the Gaussian slope in raw residual units.
- Newton defaults to 100 iterations and `tol=1e-12`. The default GD path keeps
  its numerical results, 60 iterations and `tol=1e-6`. Closed-form `solve()`
  hooks take precedence over either solver.
- Newton uses the same unconverged policy and counts windows still active
  at the iteration cap. Stage diagnostics also report `bisection_fallbacks`
  and `max_abs_score`; GD and direct fits leave these at zero and NaN.
- Add research-score root fixtures, spike and bursty regressions, and a
  reproducible benchmark against default GD.

### Upgrading from 0.4.0

`IMF.on_unconverged` is a new configuration attribute. Default GD now emits
`RuntimeWarning` on capped stages, so suites using `-W error` or
`filterwarnings=error` may fail. Set `on_unconverged="ignore"` or add a warning
filter. `StageInfo` gained the defaulted `unconverged_windows` field: existing
positional construction still works, but `dataclasses.astuple()` returns one
more item.

## 0.4.0

- `SmoothAbs(H)` now uses precomputed score and exponential tables with uniform
  arithmetic indexing. The tables use the research `erf_approx` formula and
  4,097 standardized residuals in [-8, 8].
- `SmoothAbs(H, lookup=False)` preserves the previous direct-evaluation path.
  The constructor remains compatible with existing `SmoothAbs(H)` calls.
- Lookup results approximate the direct path and can change component values
  slightly. Research-reference fixtures cover the score, loss, components and
  residual. The interpolated score is only approximately the derivative of
  the interpolated loss; reducing solver tolerance does not reduce table error.
- Documentation explains the new default and the direct-evaluation option.
