import jax.numpy as jnp

def fuselage_mass(MTOM, l_fus_m, r_fus_m, rho, V_cruise):
    """Estimate fuselage mass (JAX).

    Implements the same empirical Raymer/Nicolai combination as the NumPy version.
    """
    kg_to_lb = 2.205
    m_to_ft = 3.281
    Npm2_to_lbf_ft2 = 0.020885434273039363

    l_ft = l_fus_m * m_to_ft
    r_ft = r_fus_m * m_to_ft
    dFS = 2 * r_ft
    SFUS = 2 * jnp.pi * r_ft * l_ft + jnp.pi * r_ft ** 2

    nz = 2.5
    WO_lb = MTOM * kg_to_lb
    q = 0.5 * rho * V_cruise ** 2 * Npm2_to_lbf_ft2
    lHT = l_ft * 0.5
    lFS = l_ft

    VP = 0.0
    deltaP = 0.0

    lF = l_ft
    wF = r_ft
    dF = r_ft
    VH = V_cruise * 1.9438

    WFUS_raymer = (0.052 * SFUS ** 1.086 * (nz * WO_lb) ** 0.177 * lHT ** (-0.051)
                   * (lFS / dFS) ** (-0.072) * q ** 0.241 + 11.9 * (VP * deltaP) ** 0.271) / kg_to_lb

    WFUS_nicolai = (200 * ((nz * WO_lb) / 1e5) ** 0.286 * (lF / 10) ** 0.857
                    * ((wF + dF) / 10) ** 0.338 * (VH / 100) ** 1.1) / kg_to_lb

    m_fuselage = 0.5 * (WFUS_raymer + WFUS_nicolai)
    return m_fuselage
