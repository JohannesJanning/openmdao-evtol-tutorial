import jax.numpy as jnp

def interior_mass(MTOM, rho, V_cruise):
    kg_to_lb = 2.205
    Npm2_to_lbf_ft2 = 0.020885434273039363

    WO = MTOM * kg_to_lb
    q = 0.5 * rho * V_cruise ** 2 * Npm2_to_lbf_ft2

    W_fur_raymer = (0.0582 * WO - 65) / kg_to_lb
    W_fur_nicolai = (34.5 * q ** 0.25) / kg_to_lb

    m_furnish = 0.5 * (W_fur_raymer + W_fur_nicolai)
    return m_furnish
