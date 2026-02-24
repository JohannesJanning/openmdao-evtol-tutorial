import jax.numpy as jnp

def cd_calculation(c_l, AR, c_d_min, e):
    """Calculate the total drag coefficient C_D including induced drag.

    Mirrors the original implementation in `src/models/aerodynamics/drag_coefficient.py`.
    """
    c_d_induced = c_l ** 2 / (jnp.pi * AR * e)
    return c_d_min + c_d_induced
