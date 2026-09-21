"""Intrinsic multiscale filtering (IMF) for 1-D signals."""

from .contrasts import Contrast, Quadratic, SmoothAbs
from .decompose import IMF, IMFResult, StageInfo
from .kernels import (
    Epanechnikov,
    Kernel,
    SquaredTriangle,
    Triangle,
    Uniform,
    epanechnikov,
    squared_triangle,
    triangle,
    uniform,
)

__version__ = "0.2.0"

__all__ = [
    "Contrast",
    "Epanechnikov",
    "IMF",
    "IMFResult",
    "Kernel",
    "Quadratic",
    "SmoothAbs",
    "SquaredTriangle",
    "StageInfo",
    "Triangle",
    "Uniform",
    "__version__",
    "epanechnikov",
    "squared_triangle",
    "triangle",
    "uniform",
]
