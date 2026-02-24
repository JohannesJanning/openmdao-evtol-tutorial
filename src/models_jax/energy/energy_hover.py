import jax.numpy as jnp

def energy_hover(P_hover, t_hover):
    val = (P_hover * t_hover) / 3600.0
    return jnp.maximum(val, 0.0)
