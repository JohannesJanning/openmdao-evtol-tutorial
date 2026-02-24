import math

def interior_mass(MTOM, rho, V_cruise):
    """
    Estimate aircraft furnish mass using Raymer and Nicolai empirical models.

    Parameters:
        MTOM (float): Maximum take-off mass [kg]
        rho (float): Air density [kg/m³]
        V_cruise (float): Cruise speed [m/s]

    Returns:
        float: Estimated furnish mass [kg]
    """
    # Conversion factors
    kg_to_lb = 2.205                                  # [lb/kg]
    Npm2_to_lbf_ft2 = 0.020885434273039363            # [lbf/ft² per N/m²]

    # 1) Take-off weight in pounds
    WO = MTOM * kg_to_lb                              # [lb]

    # 2) Dynamic pressure in imperial units
    q = 0.5 * rho * V_cruise**2 * Npm2_to_lbf_ft2      # [lbf/ft²]

    # 3) Furnish weight estimates (converted back to SI)
    W_fur_raymer = (0.0582 * WO - 65) / kg_to_lb       # [kg]
    W_fur_nicolai = (34.5 * q**0.25) / kg_to_lb        # [kg]

    # 4) Mean furnish mass
    m_furnish = 0.5 * (W_fur_raymer + W_fur_nicolai)   # [kg]

    return m_furnish
