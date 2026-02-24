import jax.numpy as jnp

def lift_to_drag_ratio(CL, CD):
    return CL / CD
