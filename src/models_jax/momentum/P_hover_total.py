import jax.numpy as jnp

def power_required_hover(sigma_hover, T_req_total_hover, rho, eta_h):
    """Calculate total power required during hover (JAX parity).

    Mirrors `src/models/momentum/P_hover_total.py`.
    """
    # ensure disk loading is non-negative to avoid sqrt of negative
    from src.models_jax.utils.smoothing import soft_cap, soft_floor

    sigma_safe = soft_floor(sigma_hover, 1e-8, k=80.0)
    # induced velocity from momentum theory (safe)
    # compute directly from physics; avoid artificial hard caps
    v_i = jnp.sqrt(sigma_safe / (2.0 * jnp.maximum(rho, 1e-8)))
    eta_safe = jnp.maximum(eta_h, 1e-6)
    return T_req_total_hover * v_i / eta_safe
