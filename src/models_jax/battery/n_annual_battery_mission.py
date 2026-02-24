import jax.numpy as jnp

def number_of_battery_required_annually(
    N_cycles_available: float,
    N_wd: float,
    T_D: float,
    time_trip: float,
    DH: float
) -> float:
    """Calculate number of batteries required per year (JAX)."""
    return (1.0 / N_cycles_available) * N_wd * T_D / (time_trip * DH)
