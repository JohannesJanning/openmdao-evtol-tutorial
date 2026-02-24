import jax.numpy as jnp

def energy_total_trip(e_hover, e_climb, e_cruise):
    val = e_hover + e_climb + e_cruise
    return jnp.maximum(val, 0.0)

def energy_reserve(P_cruise, t_reserve):
    val = (P_cruise * t_reserve) / 3600.0
    return jnp.maximum(val, 0.0)

def energy_total_required(e_trip, e_reserve):
    val = e_trip + e_reserve
    return jnp.maximum(val, 0.0)
