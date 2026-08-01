# pimf

Intrinsic multiscale filtering (IMF) for one-dimensional signals: a **linear**
decomposition (local weighted mean) and a **robust** variant that replaces the
mean with a smooth robust location fit solved by gradient descent.

Both are the same algorithm. Starting from `r_1 = y`, each stage smooths the
current residual with a local M-estimator and passes on what is left:

```
S_k = argmin_x  sum_u  w_{t,u} * rho(r_k(u) - x)        (per position t)
r_{k+1} = r_k - S_k
```

The signal decomposes exactly: `y = S_1 + ... + S_K + r_{K+1}`. The only
difference between the variants is the contrast `rho`:

- **Quadratic** `rho(r) = r^2/2` — closed form, the kernel-weighted local mean
  (the linear IMF).
- **SmoothAbs** `rho_h(r) = r*erf(r/(sqrt(2)h)) + sqrt(2/pi)*h*exp(-r^2/(2h^2))`
  — a smoothed absolute value with bounded score `psi_h(r) = erf(r/(sqrt(2)h))`,
  solved by clipped gradient descent (the robust IMF). Large contaminated
  observations have bounded influence.

The library packages the algorithms validated in the IMF research project
(`Projects/imf`) and reproduces its numerics — the robust decomposition is
bit-identical to the reference notebook on the research example.

## Install

```bash
pip install pimf
```

Requires Python >= 3.10. NumPy is the only dependency.

For development, clone the repo and install in editable mode:

```bash
pip install -e .
```

## Quickstart

```python
import numpy as np
import pimf

t = np.linspace(0.0, 1.0, 1000)
y = (
    np.sin(2 * np.pi * t)
    + 0.25 * np.sin(12 * np.pi * t)
    + np.random.default_rng(0).normal(0, 0.1, 1000)
)

linear = pimf.linear_imf(y)  # quadratic contrast, closed form
robust = pimf.robust_imf(y, h=0.2)  # smooth-abs contrast, h ~ 2 * noise sigma

robust.imfs  # (K, n) array of components, coarsest first
robust.residual  # (n,) final residual
robust.stages  # per-stage diagnostics (window size, GD iterations, ...)
robust.reconstruction  # imfs.sum(axis=0) + residual == y to ~1e-15
```

The general entry point is `pimf.imf(y, window_sizes=..., contrast=..., kernel=...,
boundary=...)`; `linear_imf` and `robust_imf` are one-line wrappers around it.

## Concepts

**Window schedule.** `make_window_schedule(n)` generates the per-stage window
sizes: odd, strictly decreasing, starting at about `n/2` and shrinking
geometrically by `sqrt(2)` down to a floor of 31. Pass `window_sizes=` to
override. Windows must be odd.

**Kernel.** The window weights come from a kernel profile `k(u)` on `[-1, 1]`.
The default `SquaredTriangle` uses `k(u) = 0.75 * (1 - |u|)^2` — the square of
the triangular kernel. (The research project and IMF.pdf call this kernel
"Epanechnikov"; that is a misnomer — the classical Epanechnikov kernel is
`(3/4)(1 - u^2)` — so this library names it for what it is.) Endpoint weights
are exactly zero, and weights are normalized to sum to one.

**Contrast.** A contrast supplies the loss `rho` (via `__call__`), its
derivative `psi` (the score), and `curvature()` — an upper bound on `rho''`
that sets the stable gradient step `0.95 / curvature()`. A contrast with a
closed-form minimizer can provide `solve(windows, weights)`, which the driver
uses instead of gradient descent (that is what makes `Quadratic` the fast
linear path). For `SmoothAbs(h)`, `h ~ 2 * sigma` of the Gaussian noise is the
research-validated choice: smaller `h` behaves like a running median, larger
`h` like the local mean.

**Boundary.** `boundary="wrap"` (circular) is the default and the setting the
linear-operator theory of the research project assumes; it is passed straight
to `np.pad`, so `"reflect"` and `"edge"` also work.

## Extending

Custom contrast — one small class:

```python
import numpy as np
import pimf


class Huber(pimf.Contrast):
    def __init__(self, delta):
        self.delta = delta

    def __call__(self, r):
        a = np.abs(r)
        return np.where(a <= self.delta, 0.5 * r**2, self.delta * (a - 0.5 * self.delta))

    def psi(self, r):
        return np.clip(r, -self.delta, self.delta)

    def curvature(self):
        return 1.0


result = pimf.imf(y, contrast=Huber(0.3))
```

Custom kernel — one line:

```python
class Triangle(pimf.Kernel):
    def profile(self, u):
        return 1.0 - np.abs(u)


result = pimf.imf(y, kernel=Triangle())
```

## Numerical guarantees

- Exact reconstruction: `imfs.sum(axis=0) + residual` matches the input to
  ~1e-15 (float rounding only).
- Deterministic: no threading, no hidden state; the same input always gives
  the same output.
- Research parity (verified by cross-check against the reference notebook on
  the seed-777 example): kernel weights, the erf approximation
  (Abramowitz–Stegun 7.1.26 — deliberately kept instead of SciPy), signal
  generation, and the full robust decomposition are bit-identical; the linear
  path matches the loop-based notebook to ~2e-15 (float summation order — it
  is bit-identical to the vectorized `windows @ weights` notebooks).

## Development

```bash
uv sync                  # or: pip install -e . && pip install pytest ruff
python -m pytest
ruff check . && ruff format --check .
```

See [AGENTS.md](AGENTS.md) for the project's code style and hard rules.

## License

MIT — see [LICENSE](LICENSE).
