import jax.numpy as jnp

def horizontal_climb_speed(V_climb, theta_deg_climb):
    theta_rad = jnp.deg2rad(theta_deg_climb)
    return V_climb * jnp.cos(theta_rad)
