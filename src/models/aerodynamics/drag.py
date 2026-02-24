import math

def drag_calculation(rho, V, c, b, C_D):
    """
    Calculate aerodynamic drag force.

    Parameters:
        rho (float): Air density [kg/m³]
        V   (float): Flight speed [m/s]
        c   (float): Wing chord length [m]
        b   (float): Wing span [m]
        C_D (float): Drag coefficient [-]

    Returns:
        float: Drag force [N]
    """
    S = c * b  # Wing reference area [m²]
    return 0.5 * rho * V**2 * S * C_D
