# Changelog

## Unreleased

- Add `IMF(on_unconverged="warn" | "raise" | "ignore")`, defaulting to a
  `RuntimeWarning` when a stage hits `max_iter` with windows still moving. The
  message includes the stage, window size and total unconverged window count.
  The policy also applies with multiple workers.
- Add `StageInfo.unconverged_windows`, counting final row updates above the
  solver chunk's stopping threshold, `tol * (1 + max(abs(x)))`. Preserve research
  numerical results and stopping iterations. `converged=True` means this count
  is zero; it does not bound distance to the score root. Non-finite updates
  count as unconverged, and parallel diagnostics propagate NaN.
- Batch users can silence per-stage warnings with `on_unconverged="ignore"`
  and inspect `StageInfo.converged` / `unconverged_windows`, or use a Python
  `warnings` filter. Long schedules and repeated runs can emit many warnings.

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
