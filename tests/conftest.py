"""Test fixtures ported from the IMF research notebooks.

The rng draw order in generate_observation (normal, random, exponential)
matches the notebooks so seeded observations reproduce bit-for-bit.
"""

import numpy as np


def gen_signal(t):
    """Deterministic 4-component test signal from the research notebooks."""
    slow = 0.6 * np.sin(2 * np.pi * t)
    medium = 0.25 * np.sin(12 * np.pi * t)
    bump = 0.8 * np.exp(-((t - 0.55) ** 2) / (2 * 0.015**2))
    trend = 0.5 * (t - 0.5)
    return slow + medium + bump + trend


def generate_observation(x, sigma, contamination_prob, contamination_scale, rng):
    """Gaussian noise plus one-sided exponential contamination."""
    gaussian_noise = rng.normal(loc=0.0, scale=sigma, size=len(x))
    contamination_mask = rng.random(len(x)) < contamination_prob
    exponential_noise = rng.exponential(scale=contamination_scale, size=len(x))
    contamination = contamination_mask * exponential_noise
    return x + gaussian_noise + contamination
