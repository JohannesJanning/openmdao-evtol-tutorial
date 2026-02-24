import math
import numpy as np

def fuselage_mass(MTOM, l_fus_m, r_fus_m, rho, V_cruise):
    """
    Estimate fuselage mass using Raymer and Nicolai empirical models.

    Parameters:
        MTOM (float): Maximum take-off mass [kg]
        l_fus_m (float): Fuselage length [m]
        r_fus_m (float): Fuselage radius [m]
        rho (float): Air density [kg/m³]
        V_cruise (float): Cruise speed [m/s]

    Returns:
        float: Estimated fuselage mass [kg]
    """

    # --- Conversions ---
    kg_to_lb = 2.205
    m_to_ft = 3.281
    Npm2_to_lbf_ft2 = 0.020885434273039363

    # --- Geometric transformations ---
    l_ft = l_fus_m * m_to_ft
    r_ft = r_fus_m * m_to_ft
    dFS = 2 * r_ft                                    # Fuselage diameter [ft]
    SFUS = 2 * np.pi * r_ft * l_ft + np.pi * r_ft**2  # Wetted fuselage area [ft²]

    # --- Load and aero inputs ---
    nz = 2.5                                          # Load factor (EASA SC-VTOL)
    WO_lb = MTOM * kg_to_lb                           # [lb]
    q = 0.5 * rho * V_cruise**2 * Npm2_to_lbf_ft2     # Dynamic pressure [lbf/ft²]
    lHT = l_ft * 0.5                                  # Tail arm [ft]
    lFS = l_ft                                        # Fuselage length [ft]

    # --- Pressurization (none assumed) ---
    VP = 0                                            # Pressurization volume ratio [-]
    deltaP = 0                                        # Pressure differential [Pa]

    # --- NICOLAI geometric parameters ---
    lF = l_ft
    wF = r_ft
    dF = r_ft
    VH = V_cruise * 1.9438                            # Convert to knots (EAS correction = 1)

    # --- Mass estimation formulas ---
    WFUS_raymer = (0.052 * SFUS**1.086 * (nz * WO_lb)**0.177 * lHT**(-0.051) * 
                   (lFS / dFS)**(-0.072) * q**0.241 + 
                   11.9 * (VP * deltaP)**0.271) / kg_to_lb

    WFUS_nicolai = (200 * ((nz * WO_lb) / 1e5)**0.286 *
                    (lF / 10)**0.857 *
                    ((wF + dF) / 10)**0.338 *
                    (VH / 100)**1.1) / kg_to_lb

    # --- Final average mass ---
    m_fuselage = 0.5 * (WFUS_raymer + WFUS_nicolai)

    return m_fuselage
