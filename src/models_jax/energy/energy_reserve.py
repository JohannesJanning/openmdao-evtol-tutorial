import jax.numpy as jnp

def energy_reserve(P_cruise, t_reserve):
    val = P_cruise * t_reserve / 3600.0
    return jnp.maximum(val, 0.0)
