import math

def wing_mass(MTOM, V_cruise, b, c, rho):
    """
    Estimate the wing mass using empirical models by Raymer and Nicolai.

    Parameters:
        MTOM (float): Maximum takeoff mass [kg]
        V_cruise (float): Cruise speed [m/s]
        b (float): Wing span [m]
        c (float): Wing chord [m]
        rho (float): Air density [kg/m³]

    Returns:
        float: Estimated wing mass [kg]
    """
    # --- Unit conversions ---
    kg_to_lb = 2.205
    m_to_ft = 3.281
    Npm2_to_lbf_ft2 = 0.020885434273039363

    # Geometry
    b_ft = b * m_to_ft
    c_ft = c * m_to_ft
    SW = b_ft * c_ft                           # Wing area [ft²]
    ARW = b / c                                # Aspect ratio
    deltaC4 = 0                                # Quarter-chord sweep angle (°)
    l_ambda = 1                                # Taper ratio
    t_c = 0.12                                 # Thickness-to-chord ratio (NACA 0012)

    # Environment & flight
    q = 0.5 * rho * V_cruise**2 * Npm2_to_lbf_ft2  # Dynamic pressure [lbf/ft²]
    nz = 2.5                                      # Load factor (CS-23/25)
    WO = MTOM * kg_to_lb                          # Takeoff weight [lb]
    VH = V_cruise * 1.9438                        # Speed in knots

    # Mass models
    WW_raymer = (0.036 * SW**0.758 * 1 * (ARW / math.cos(math.radians(deltaC4))**2)**0.6 *
                 q**0.006 * l_ambda**0.04 * (100 * t_c / math.cos(math.radians(deltaC4)))**(-0.3) *
                 (nz * WO)**0.49) / kg_to_lb

    WW_nicolai = (96.948 * ((nz * WO) / 1e5)**0.65 * (ARW / math.cos(math.radians(deltaC4))**2)**0.57 *
                  (SW / 100)**0.61 * (1 + l_ambda / 2 * t_c)**0.36 *
                  math.sqrt(1 + VH / 500)**0.993) / kg_to_lb

    # Mean value
    m_wing = 0.5 * (WW_raymer + WW_nicolai)

    return m_wing
