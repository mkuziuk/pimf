# pimf — agent guide

## What this is

`pimf` packages the intrinsic multiscale filtering (IMF) research from
`/Users/mikhail/Projects/imf` into a small NumPy-only library. The research
notebooks are the specification: when behavior is in question, the notebooks
win.

The decomposition: `r_1 = y`; at stage k a local M-estimator (kernel weights +
contrast function) smooths the current residual into component `S_k`; then
`r_{k+1} = r_k - S_k`. The linear IMF is the quadratic contrast (closed-form
weighted mean); the robust IMF is the smoothed-absolute contrast solved by
clipped gradient descent.

## Code style: simple and robust

- Always prefer the simplest solution that works. If two designs solve the
  problem, pick the one with less machinery.
- Small modules, plain functions, tiny classes. No metaprogramming, no
  registries, no factories, no speculative abstraction.
- If a helper is used once, inline it.
- Raise `ValueError` early on invalid input; otherwise trust NumPy.
- Comments only for constraints the code cannot express (e.g. why a step size
  is stable). No narration.

## Hard rules

- NumPy is the only runtime dependency. Never add scipy, pandas, numba, etc.
- Numerics must reproduce the research notebooks. Keep `erf_approx`
  (Abramowitz–Stegun 7.1.26); never swap in `scipy.special.erf` or
  `math.erf`; no lookup grids.
- Window sizes are always odd; raise `ValueError` otherwise.
- Boundary default is `"wrap"` (the setting the linear-operator theory
  assumes); it is passed straight to `np.pad`, so `"reflect"` and `"edge"`
  also work.
- Vectorize with NumPy. No threading or multiprocessing: the research audit
  showed chunked stopping changes results at tolerance scale across worker
  counts. Deterministic beats marginally faster.
- Don't add features speculatively. v1 scope is: linear + robust IMF,
  contrasts, kernels, window schedule. No plotting, no median baselines, no
  FFT operator analysis.
- The default kernel profile is `0.75 * (1 - |u|)^2` — the square of the
  triangular kernel. The research repo calls it "Epanechnikov"; that is a
  misnomer (classical Epanechnikov is `(3/4)(1 - u^2)`), so here the class is
  named `SquaredTriangle`. Keep the formula and this naming note in sync.

## Testing

- pytest; run `python -m pytest`.
- Every numerical claim is backed by a fixture validated in the research repo.
- Use explicit tolerances. Exact equality (`np.array_equal`) only where
  bit-for-bit parity with the research code is the contract (erf, kernel
  weights, the linear path, the schedule).

## Tooling

- `ruff check .` and `ruff format .` must pass.
- Python >= 3.10.
