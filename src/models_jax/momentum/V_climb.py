import jax.numpy as jnp

def climb_speed(MTOM, g, theta_deg, c_l, c, b, rho):
    theta_rad = jnp.deg2rad(theta_deg)
    denominator = c_l * c * b * rho
    return jnp.sqrt((2.0 * MTOM * g * jnp.cos(theta_rad)) / denominator)
