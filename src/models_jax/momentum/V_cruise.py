import jax.numpy as jnp

def cruise_speed(MTOM, g, c_l, c_d, alpha_deg, c, b, rho):
    alpha_rad = jnp.deg2rad(alpha_deg)
    denominator = (c_l + c_d * jnp.tan(alpha_rad)) * c * b * rho
    return jnp.sqrt((2.0 * MTOM * g) / denominator)
