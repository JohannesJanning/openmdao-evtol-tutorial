"""Shim package to expose original transportation functions under src.models_jax.

This module intentionally re-exports the original Python implementation
so the JAX package can import the same API without duplicating complex logic.
"""

from src.models.transportation.transportation_modes import transportation_mode_comparison

__all__ = ["transportation_mode_comparison"]
