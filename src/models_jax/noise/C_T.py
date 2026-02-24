import jax.numpy as jnp

def thrust_coefficient(T, rho, A_disk, n_rotors):
    """Compute nondimensional thrust coefficient C_T = T / (rho * A * n_rotors * (omega*R)^2)
    Caller supplies consistent units; this is a lightweight port."""
    return T / (rho * A_disk * n_rotors)
