import jax.numpy as jnp

def total_thrust_required_climb(D_climb, MTOM, g, theta_deg_climb):
    theta_rad = jnp.deg2rad(theta_deg_climb)
    return D_climb + MTOM * g * jnp.sin(theta_rad)
