import jax.numpy as jnp

def SPL_flight(distance, SPL0, decay=20.0):
    """Simple spherical spreading model: SPL = SPL0 - decay*log10(r/r0)"""
    return SPL0 - decay * jnp.log10(distance)
