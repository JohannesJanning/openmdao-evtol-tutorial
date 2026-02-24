import jax.numpy as jnp

def compute_ct(thrust_N, rho, V, R_prop):
    """JAX port of compute_ct from the reference model.

    C_T = T / (0.5 * rho * V**2 * A)
    """
    A = jnp.pi * R_prop ** 2
    return thrust_N / (0.5 * rho * V ** 2 * A)
