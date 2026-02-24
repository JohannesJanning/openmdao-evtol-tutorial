import jax.numpy as jnp

def battery_mass(energy_total_required, rho_bat):
    """Estimate battery mass based on required energy and energy density (JAX).

    Returns mass in kg.
    """
    energy_safe = jnp.maximum(energy_total_required, 1e-6)
    rho_safe = jnp.maximum(rho_bat, 1e-6)
    return energy_safe / (0.64 * rho_safe)
