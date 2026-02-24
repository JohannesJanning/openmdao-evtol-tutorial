import jax.numpy as jnp

def lift_coefficient_to_lift(CL, rho, V, S):
    return 0.5 * rho * V ** 2 * S * CL
