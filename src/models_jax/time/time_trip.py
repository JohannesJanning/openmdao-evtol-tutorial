import jax.numpy as jnp

def total_trip_time(t_hover, t_climb, t_cruise):
    return t_hover + t_climb + t_cruise
