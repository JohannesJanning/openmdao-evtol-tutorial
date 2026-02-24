import jax.numpy as jnp

def energy_climb(P_climb, t_climb):
    val = (P_climb * t_climb) / 3600.0
    return jnp.maximum(val, 0.0)
