import jax.numpy as jnp

def compute_time_cruise(distance_total, v_climb_hor, t_climb, v_cruise):
    distance_climb = v_climb_hor * t_climb
    distance_cruise = distance_total - distance_climb
    # protect against tiny/negative cruise distance or near-zero cruise speed
    distance_cruise_safe = jnp.maximum(distance_cruise, 1e-6)
    v_cruise_safe = jnp.maximum(v_cruise, 1e-3)
    return distance_cruise_safe / v_cruise_safe
