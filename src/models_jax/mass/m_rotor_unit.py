import jax.numpy as jnp

def rotor_mass_per_unit(m_rotor, n_prop_vert, n_prop_hor):
    return m_rotor / (n_prop_vert + n_prop_hor)
