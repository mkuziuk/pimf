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

`IMF` warns with a `RuntimeWarning` when a stage hits `max_iter` with windows
still moving above the stopping threshold. Set `on_unconverged="raise"` to raise
`RuntimeError`, or `on_unconverged="ignore"` to silence it. Each `StageInfo` reports
`unconverged_windows`, summed across workers, and is `converged` only when that
count is zero. Each row uses the same threshold as its solver chunk,
`tol * (1 + max(abs(x)))`, evaluated at the final estimates. Research numerical
results and stopping iterations are preserved. `converged=True` does not bound
distance to the score root.

[Release notes](CHANGELOG.md)
