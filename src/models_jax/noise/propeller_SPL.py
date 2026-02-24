import jax.numpy as jnp

def propeller_SPL(n_rpm, R_prop, T_req_per_prop, rho, A_prop, V_forward, h_obs_ft):
    """JAX port of the SKM propeller SPL model from src/models/noise/SPL_flight.py."""
    K = 6.1e-27
    P_ref_sq = 1e-16
    CL_ref = 0.4

    n_rad = 2.0 * jnp.pi * n_rpm / 60.0
    V_tip = n_rad * R_prop
    V_07 = 0.7 * V_tip
    # avoid division by zero in V_forward
    V_for_safe = jnp.maximum(V_forward, 1e-3)
    C_L = 2.0 * T_req_per_prop / (jnp.maximum(rho, 1e-8) * jnp.maximum(A_prop, 1e-8) * V_for_safe ** 2)
    C_L_safe = jnp.maximum(C_L, 1e-6)

    term1 = 10.0 * jnp.log10((K * A_prop * V_07 ** 6) / P_ref_sq)
    term2 = 20.0 * jnp.log10(C_L_safe / CL_ref)
    correction = 20.0 * jnp.log10(300.0 / jnp.maximum(h_obs_ft, 1.0))

    SPL = term1 + term2 + correction
    return SPL
