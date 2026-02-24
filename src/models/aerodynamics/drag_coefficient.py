import math

def cd_calculation(c_l, AR, c_d_min, e):
    """
    Calculate the total drag coefficient C_D including induced drag.

    Parameters:
        c_l (float): Lift coefficient (dimensionless)
        AR (float): Wing aspect ratio (span/chord) [-]
        c_d_min (float): Minimum (parasite) drag coefficient [-]
        e (float): Oswald efficiency factor [-]

    Returns:
        float: Total drag coefficient C_D (dimensionless)
    """
    c_d_induced = c_l**2 / (math.pi * AR * e)
    return c_d_min + c_d_induced
