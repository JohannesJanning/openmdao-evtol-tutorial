import jax.numpy as jnp

def energy_cruise(P_cruise, t_cruise):
    val = (P_cruise * t_cruise) / 3600.0
    return jnp.maximum(val, 0.0)
