# Changelog

## Unreleased

- Add `IMF(on_unconverged="warn" | "raise" | "ignore")`, defaulting to a
  `RuntimeWarning` with the stage, window size and total unconverged window count.
  The policy also applies with multiple workers.
- Add `StageInfo.unconverged_windows` and test each row's final update against its
  own tolerance when reporting convergence. Preserve research numerical results
  and stopping iterations; rows are not frozen independently. The policy also
  covers rows that fail their tolerance when the research stop rule exits early.

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
