import jax.numpy as jnp

def c_rate_average(C_hover: float, C_climb: float, C_cruise: float,
                   t_hover: float, t_climb: float, t_cruise: float, t_trip: float) -> float:
    """Compute average trip C-rate (JAX)."""
    return (C_hover * t_hover + C_climb * t_climb + C_cruise * t_cruise) / t_trip
