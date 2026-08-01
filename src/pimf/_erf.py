"""Error function approximation shared by the contrast functions.

Abramowitz & Stegun formula 7.1.26 (max abs error ~1.5e-7), kept instead of
scipy.special.erf so results match the IMF research notebooks bit-for-bit.
"""

import numpy as np

SQRT_2 = np.sqrt(2.0)
SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)


def erf_approx(x):
    """Vectorized Abramowitz-Stegun approximation to erf(x)."""
    x = np.asarray(x, dtype=float)
    sign = np.sign(x)
    ax = np.abs(x)

    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    z = 1.0 / (1.0 + p * ax)
    poly = ((((a5 * z + a4) * z + a3) * z + a2) * z + a1) * z
    return sign * (1.0 - poly * np.exp(-(ax**2)))
