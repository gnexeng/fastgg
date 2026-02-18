"""
Python implementation of the FASTRGG random graph generator.

This module provides three algorithms inspired by the original CUDA version:

- PER
- PZER
- PPreZER

The public entrypoint for most users is :func:`generate_graph`.
"""

from .algorithms import (
    Algorithm,
    generate_graph,
)

__all__ = ["Algorithm", "generate_graph"]
