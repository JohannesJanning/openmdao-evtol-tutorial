import math

def cl_calculation(alpha_deg, AR, c_l_0, e):
    """
    Calculate the 3D lift coefficient of a finite wing using lift-curve slope correction.

    Parameters:
        alpha_deg (float): Angle of attack in degrees.
        AR        (float): Wing aspect ratio (span/chord).
        c_l_0     (float): Zero-lift coefficient of the airfoil.
        e         (float): Oswald efficiency factor (default=0.8 for NACA 2412).

    Returns:
        float: Total lift coefficient (dimensionless).
    """
    alpha_rad = math.radians(alpha_deg)
    a_airfoil = 5.747
    a_wing = a_airfoil / (1 + (a_airfoil / (math.pi * AR * e)))
    return a_wing * alpha_rad + c_l_0
