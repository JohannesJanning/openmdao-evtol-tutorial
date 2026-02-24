import jax.numpy as jnp

def wing_mass(MTOM, V_cruise, b, c, rho):
    kg_to_lb = 2.205
    m_to_ft = 3.281
    Npm2_to_lbf_ft2 = 0.020885434273039363

    b_ft = b * m_to_ft
    c_ft = c * m_to_ft
    SW = b_ft * c_ft
    ARW = b / c
    deltaC4 = 0.0
    l_ambda = 1.0
    t_c = 0.12

    q = 0.5 * rho * V_cruise ** 2 * Npm2_to_lbf_ft2
    nz = 2.5
    WO = MTOM * kg_to_lb
    VH = V_cruise * 1.9438

    WW_raymer = (0.036 * SW ** 0.758 * 1 * (ARW / jnp.cos(jnp.deg2rad(deltaC4)) ** 2) ** 0.6
                 * q ** 0.006 * l_ambda ** 0.04 * (100 * t_c / jnp.cos(jnp.deg2rad(deltaC4))) ** (-0.3)
                 * (nz * WO) ** 0.49) / kg_to_lb

    WW_nicolai = (96.948 * ((nz * WO) / 1e5) ** 0.65 * (ARW / jnp.cos(jnp.deg2rad(deltaC4)) ** 2) ** 0.57
                  * (SW / 100) ** 0.61 * (1 + l_ambda / 2 * t_c) ** 0.36 * jnp.sqrt(1 + VH / 500) ** 0.993) / kg_to_lb

    m_wing = 0.5 * (WW_raymer + WW_nicolai)
    return m_wing
