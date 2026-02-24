import jax.numpy as jnp

def roc_calculation(alpha_deg_climb: float, v_climb: float) -> float:
    """Calculate the rate of climb (ROC) — JAX parity with original.

    Parameters:
        alpha_deg_climb: climb angle in degrees
        v_climb: climb speed (m/s)
    """
    alpha_rad = jnp.deg2rad(alpha_deg_climb)
    return jnp.sin(alpha_rad) * v_climb
