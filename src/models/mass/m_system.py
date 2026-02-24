import math

def system_mass(MTOM, l_fus_m, b):
    """
    Estimate system/control mass using Raymer and Nicolai models.

    Parameters:
        MTOM (float): Maximum take-off mass [kg]
        l_fus_m (float): Fuselage length [m]
        b (float): Wing span [m]
        nz (float): Load factor (default = 2.5 for EASA SC)

    Returns:
        float: Estimated system mass [kg]
    """
    # Conversions
    l_ft = l_fus_m * 3.281
    b_ft = b * 3.281
    WO_lb = MTOM * 2.205  # Takeoff weight in pounds

    # Nicolai estimation
    WS_nicolai = (1.08 * WO_lb**0.7) / 2.205

    # Raymer estimation
    nz=2.5
    WS_raymer = (0.054 * l_ft**1.536 * b_ft**0.371 * (1.5 * nz * WO_lb * 1e-4)**0.8) / 2.205

    return 0.5 * (WS_nicolai + WS_raymer)
