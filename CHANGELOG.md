# Changelog

## Unreleased

- Add `IMF(on_unconverged="warn" | "raise" | "ignore")`, defaulting to one
  `RuntimeWarning` per stage that hits `max_iter` with unconverged windows.
  For GD, windows are still moving above the final chunk threshold,
  `tol * (1 + max(abs(x)))`. The count ignores NaNs in the maximum without
  changing the research stop rule. Non-finite updates count as unconverged.
  Messages name the stage, window size and total unconverged window count.
- Batch users can silence warnings with `on_unconverged="ignore"` and inspect
  `StageInfo.converged` / `unconverged_windows`, or use a Python `warnings`
  filter. Long schedules and repeated runs can emit many warnings.
- Add opt-in `solver="newton"` for robust fits where GD takes too long to
  reach the root. Pure-NumPy safeguarded Newton uses bisection fallbacks and
  independent window stopping. Small steps require a positional root bracket;
  flat zero-score intervals in either lookup mode return the midpoint of
  boundaries located to tolerance at their edge scale. Non-finite scores or
  estimates remain unconverged. Saturated initial means restart at a sample
  median, and local score probes tighten outlier-wide brackets before fallback
  bisections. Constant windows finish without probes, including extreme finite
  values. Built-in contrasts give bit-identical values and diagnostics across
  worker counts, including singleton chunks.
- Newton defaults to `max_iter=100` and `tol=1e-12`. Default GD retains 60
  iterations, `tol=1e-6` and bit-identical research numerics. Both target the
  same score equation. Differences beyond stopping roundoff occur where GD
  has not reached the root, even if its small-update test reports convergence,
  or when it chooses another point on a flat zero interval. GD convergence
  does not bound distance to the root. When `H << tol * abs(x)`, Newton can
  meet positional tolerance with a large score because the score is steep.
- Add vectorized `Contrast.psi_prime(r)` for custom Newton contrasts, with
  implementations for `Quadratic` and `SmoothAbs`. Slopes use raw residual
  units, including `1/H`; lookup slopes follow the piecewise-linear score,
  while direct evaluation uses the Gaussian slope and research `erf_approx`
  score. `psi_prime` no longer overflows for very small H. Closed-form `solve()`
  hooks still take precedence over either solver.
- Document every `StageInfo` field: `stage`, `window_size`, `bandwidth`,
  `iterations`, `final_max_delta`, `converged`, and new `unconverged_windows`,
  `bisection_fallbacks` and `max_abs_score`. Both iterative solvers report the
  largest absolute weighted score. GD uses the last evaluated iterate, before
  the final update, without an extra score pass; Newton uses the returned
  estimates. GD updates are unchanged. Direct fits keep NaN scores and deltas.
  Parallel maxima propagate NaN; counts sum across workers. Identical GD runs
  retain equal stage diagnostics.
- Add research-score fixtures for adversarial plateaus and tiny H, numerical
  regressions and a reproducible solver benchmark. Each Newton iteration
  needs score and slope evaluations plus any probes or bisections. On the
  issue #3 bursty signals at L=1024/2048/4096, measured speedups over default
  GD were 4.40x/5.10x/5.01x; see `benchmarks/README.md` for costs and protocol.

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
