import jax.numpy as jnp

def total_thrust_required_cruise(D_cruise, alpha_deg):
    alpha_rad = jnp.deg2rad(alpha_deg)
    return D_cruise / jnp.cos(alpha_rad)
