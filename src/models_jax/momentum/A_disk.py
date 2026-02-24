import jax.numpy as jnp

def propeller_disk_area(radius):
    # Guard radius to a small, positive value to avoid zero-area or NaN/Inf
    radius_safe = jnp.maximum(radius, 1e-3)
    return jnp.pi * radius_safe**2
