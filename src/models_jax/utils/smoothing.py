import jax.numpy as jnp

def soft_floor(x, floor, k=50.0):
    """Smooth approximation of max(x, floor).

    For large k this approaches jnp.maximum(x, floor). Returns a value
    that is C1 continuous to improve gradient quality for optimizers.
    """
    # Use logaddexp for numerical stability when k*(x-floor) is large
    z = k * (x - floor)
    return floor + jnp.logaddexp(0.0, z) / k


def soft_cap(x, cap, k=50.0):
    """Smooth approximation of min(x, cap)."""
    # Numerically stable formulation using logaddexp
    z = k * (cap - x)
    return cap - jnp.logaddexp(0.0, z) / k
