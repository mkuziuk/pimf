"""Intrinsic multiscale filtering (IMF) for 1-D signals."""

from .contrasts import Contrast, Quadratic, SmoothAbs
from .decompose import IMFResult, StageInfo, imf, linear_imf, robust_imf
from .kernels import Kernel, SquaredTriangle
from .schedule import make_window_schedule

__version__ = "0.1.0"

__all__ = [
    "Contrast",
    "IMFResult",
    "Kernel",
    "Quadratic",
    "SmoothAbs",
    "SquaredTriangle",
    "StageInfo",
    "__version__",
    "imf",
    "linear_imf",
    "make_window_schedule",
    "robust_imf",
]
