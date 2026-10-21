# pimf

pimf is a NumPy-only Python library for intrinsic multiscale filtering of
one-dimensional signals. It extracts components from coarse to fine scales
using linear or robust local fits, with optional parallel execution.

```sh
pip install pimf
```

```python
import numpy as np
from pimf import IMF, SmoothAbs

y = np.sin(2 * np.pi * np.arange(256) / 256)
result = IMF(SmoothAbs(H=0.2)).decompose(y, k_max=6)
components = result.imfs
residual = result.residual
```

Gradient descent preserves the research numerics. For robust fits that need
more accurate score roots, use `IMF(SmoothAbs(H=0.2), solver="newton")`.
Unfinished stages emit a warning by default; inspect `result.stages` for details.

See the [documentation and examples](https://mkuziuk.github.io/pimf/) for
kernels, contrasts, solver settings, convergence diagnostics and benchmarks.
[Release notes](https://github.com/mkuziuk/pimf/blob/main/CHANGELOG.md)
