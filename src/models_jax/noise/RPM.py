import jax.numpy as jnp

def rpm_from_omega(omega):
    return omega * 60.0 / (2.0 * jnp.pi)
