import jax.numpy as jnp

def mass_fraction(mass_item: float, MTOM: float) -> float:
    """Calculate mass fraction of a component relative to MTOM (JAX parity).

    Raises ValueError if MTOM == 0 to match original behavior.
    """
    if MTOM == 0:
        raise ValueError("MTOM must be greater than zero to calculate mass fraction.")
    return mass_item / MTOM
