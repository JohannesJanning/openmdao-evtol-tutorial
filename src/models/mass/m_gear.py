import math

def gear_mass(MTOM, R_prop_cruise, r_fus_m):
    """
    Estimate landing gear mass using Raymer and Nicolai empirical methods.

    Parameters:
        MTOM (float): Maximum take-off mass [kg]
        R_prop_cruise (float): Radius of cruise propeller [m]
        r_fus_m (float): Radius of fuselage [m]

    Returns:
        float: Estimated landing gear mass [kg]
    """
    # --- Constants ---
    kg_to_lb = 2.205                                 # [lb/kg]
    m_to_inch = 39.37                                # [in/m]
    n_safety = 1.5                                   # [-]
    n_certification = 2.5                            # [-]
    n_l = n_safety * n_certification                 # Ultimate landing load factor [-]

    # --- 1) Convert take-off weight to pounds ---
    W_L = MTOM * kg_to_lb                            # [lb]

    # --- 2) Clearances and geometry (in inches) ---
    prop_clearance_min_m = 0.1778                    # Minimum prop clearance [m]
    prop_clearance_min_in = prop_clearance_min_m * m_to_inch
    R_cruise_in = R_prop_cruise * m_to_inch
    r_fuselage_in = r_fus_m * m_to_inch
    l_main_in = R_cruise_in - r_fuselage_in + prop_clearance_min_in
    l_nose_in = l_main_in                            # Assume same as main

    # --- 3) Raymer main gear mass ---
    M_mg_raymer_lb = 0.095 * (n_l * W_L)**0.768 * (l_main_in / 12)**0.409
    M_mg_raymer = M_mg_raymer_lb / kg_to_lb          # [kg]

    # --- 4) Raymer nose gear mass ---
    M_ng_raymer_lb = 0.125 * (n_l * W_L)**0.566 * (l_nose_in / 12)**0.845
    M_ng_raymer = M_ng_raymer_lb / kg_to_lb          # [kg]

    # --- 5) Nicolai total gear mass ---
    M_tg_nicolai_lb = 0.054 * (n_l * W_L)**0.684 * (l_main_in / 12)**0.601
    M_tg_nicolai = M_tg_nicolai_lb / kg_to_lb        # [kg]

    # --- 6) Average of both methods ---
    m_gear = 0.5 * (M_mg_raymer + M_ng_raymer + M_tg_nicolai)  # [kg]

    return m_gear
