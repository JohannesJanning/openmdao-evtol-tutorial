import jax.numpy as jnp

def battery_energy_capacity(rho_bat: float, m_battery: float) -> float:
    """Compute battery design energy capacity (JAX)."""
    return rho_bat * m_battery
