import jax.numpy as jnp

def rotor_mass(n_prop_vert, n_prop_hor, R_prop_hover, R_prop_cruise):
    k_evtol = 13.0
    div = 0.0

    term_hover = n_prop_vert * k_evtol * (1 - div) * (0.7484 * R_prop_hover ** 1.2 - 0.0403 * R_prop_hover)
    term_cruise = n_prop_hor * k_evtol * (1 - div) * (0.7484 * R_prop_cruise ** 1.2 - 0.0403 * R_prop_cruise)
    return term_hover + term_cruise
