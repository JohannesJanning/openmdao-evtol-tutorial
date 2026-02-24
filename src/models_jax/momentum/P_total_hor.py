import jax.numpy as jnp

def power_total_required(V, T_req_total, T_req_per_prop, rho, A_prop, n_props, eta_c):
    v_i = -V / 2.0 + jnp.sqrt((V / 2.0)**2 + T_req_per_prop / (2.0 * rho * A_prop))
    return (T_req_total * V + T_req_per_prop * v_i * n_props) / eta_c
