import math

def climb_speed(MTOM, g, theta_deg, c_l, c, b, rho):
    """
    Calculate the climb speed based on vertical force balance.

    Parameters:
        MTOM      (float): Maximum take-off mass [kg]
        g         (float): Gravitational acceleration [m/s²]
        theta_deg (float): Climb angle [deg]
        c_l       (float): Lift coefficient during climb (-)
        c         (float): Wing chord [m]
        b         (float): Wing span [m]
        rho       (float): Air density [kg/m³]

    Returns:
        float: Climb speed [m/s]
    """
    theta_rad = math.radians(theta_deg)
    denominator = c_l * c * b * rho
    return math.sqrt((2 * MTOM * g * math.cos(theta_rad)) / denominator)
