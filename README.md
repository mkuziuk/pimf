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

`IMF` warns with a `RuntimeWarning` when a stage has windows that fail their
individual update tolerance. Set `on_unconverged="raise"` to raise `RuntimeError`,
or `on_unconverged="ignore"` to silence it. Each `StageInfo` reports
`unconverged_windows`, summed across workers, and is `converged` only when that
count is zero. The research iteration is preserved, so a stage can report
unconverged rows even before `max_iter`. Small updates do not guarantee a small
distance to the fitted root.

[Release notes](CHANGELOG.md)
