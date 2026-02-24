import jax.numpy as jnp

def depth_of_discharge(energy_trip: float, E_battery: float) -> float:
    """Compute depth of discharge (JAX)."""
    return energy_trip / E_battery
