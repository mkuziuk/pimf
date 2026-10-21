# Changelog

## Unreleased

- Add `IMF(on_unconverged="warn" | "raise" | "ignore")`, defaulting to one
  `RuntimeWarning` per capped stage with unfinished windows. Messages include
  the stage, window size, unfinished count and a remedy. The policy also
  applies with multiple workers.
- Add opt-in `solver="newton"` for robust local score roots, with independent
  window stopping and safeguarded bisection. Stalled weighted means restart
  at a weighted median. Outlier-wide brackets bisect float64 ranks, bounding
  these searches to at most 64 ordered halvings. Small steps require a
  positional root bracket. Flat zero-score intervals return the midpoint of
  boundaries resolved to tolerance at their edge scale. Non-finite scores or
  estimates remain unconverged. Built-in contrasts give bit-identical Newton
  values and diagnostics across worker counts.
- Add vectorized `Contrast.psi_prime(r)` for custom Newton contrasts, with
  implementations for `Quadratic` and `SmoothAbs`. Slopes use raw residual
  units, including `1/H`; lookup slopes follow the piecewise-linear score.
  Direct evaluation retains the research `erf_approx` score and uses the
  Gaussian slope. Closed-form `solve()` hooks still take precedence.
- Add `StageInfo.unconverged_windows`, `bisection_fallbacks` and `max_abs_score`.
  GD counts updates above the final chunk threshold, `tol * (1 + max(abs(x)))`,
  including non-finite updates. Its score diagnostic uses the last evaluated
  iterate, before the final update, without another score pass. Newton reports
  the score at its returned estimates. Parallel maxima propagate NaN and
  counts sum across workers. Direct fits report NaN scores and deltas.
- Add a reproducible solver benchmark; see `benchmarks/README.md` for measured
  times, evaluation costs and the protocol.

### Upgrading from 0.4.0

The signature defaults `IMF(max_iter=None, tol=None)` now resolve per solver:
GD retains `60` and `1e-6`, with bit-identical research numerics; Newton uses
`100` and `1e-12`. Explicit values override these defaults. `IMF.solver` and
`IMF.on_unconverged` are new configuration attributes.

Default GD now emits `RuntimeWarning` on capped stages, so suites using
`-W error` or `filterwarnings=error` may fail. Set `on_unconverged="ignore"`
or add a warning filter. `StageInfo` gained three defaulted fields: existing
positional construction still works, but `dataclasses.astuple()` now returns
nine items instead of six. GD's small-update convergence flag does not bound
distance to the score root. Newton's positional tolerance does not remove
lookup table approximation error.

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
