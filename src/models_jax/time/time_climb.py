import jax.numpy as jnp

def climb_time(h_cruise, h_hover, roc):
    # protect against zero or negative ROC during tracing/jit
    roc_safe = jnp.maximum(roc, 1e-6)
    return (h_cruise - h_hover) / roc_safe
