# pimf — agent guide

## What this is

`pimf` packages the intrinsic multiscale filtering (IMF) research from
https://github.com/mkuziuk/imf into a small NumPy-only library.

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

## Testing

- pytest; run `python -m pytest`.
- Every numerical claim is backed by a fixture validated in the research repo.
- Use explicit tolerances. Exact equality (`np.array_equal`) only where
  bit-for-bit parity with the research code is the contract (erf, kernel
  weights, the linear path, the schedule).

## Tooling

- `ruff check .` and `ruff format .` must pass.
- Python >= 3.10.
