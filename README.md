# pimf

Intrinsic multiscale filtering for one-dimensional signals. Configure a kernel
and contrast once, then decompose signals into coarse-to-fine components and a
residual. NumPy is the only runtime dependency. Requires Python 3.10 or later.

[Documentation](https://mkuziuk.github.io/pimf/) ·
[API reference](https://mkuziuk.github.io/pimf/#api)

## Install

The object API below is available in the 0.2 development branch:

```bash
pip install "pimf @ git+https://github.com/mkuziuk/pimf@codex/imf-object-api"
```

## Quickstart

```python
import numpy as np
from pimf import IMF
from pimf.contrasts import SmoothAbs
from pimf.kernels import epanechnikov

t = np.arange(1000) / 1000
y = np.sin(2 * np.pi * t) + np.random.default_rng(0).normal(0, 0.1, len(t))

method = IMF(contrast=SmoothAbs(H=0.2), kernel=epanechnikov)
result = method.decompose(y, h1=0.25, a=np.sqrt(2), k_max=8)

result.imfs  # components, coarsest first; shape (K, n)
result.residual  # final residual; shape (n,)
result.reconstruction  # imfs.sum(axis=0) + residual
result.stages  # bandwidths, window sizes, iterations, convergence
```

`IMF()` uses the quadratic contrast, squared-triangular kernel and circular
boundaries. `h1` is the first window's half-width as a fraction of the unit time
domain, so `0.25` covers about half the samples. Subsequent bandwidths shrink
by `a`. Set `h_min` to stop at a minimum bandwidth, or supply explicit odd
`window_sizes` to reproduce an integer-window experiment.

Predefined kernels are `squared_triangle`, `epanechnikov`, `triangle` and
`uniform`. Custom `Kernel` and `Contrast` subclasses remain supported.

## Development

```bash
uv sync
uv run python -m pytest
uv run ruff check .
uv run ruff format --check .
uv build
```

Documentation is plain HTML and CSS in `docs/`, served directly by GitHub Pages.
Preview it with `python -m http.server 8000 --directory docs`.

MIT licensed. See [LICENSE](LICENSE).
