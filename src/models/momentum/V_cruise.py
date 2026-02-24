import math

def cruise_speed(MTOM, g, c_l, c_d, alpha_deg, c, b, rho):
    """
    Calculate the cruise speed based on vertical force balance.

    Parameters:
        MTOM       (float): Maximum take-off mass [kg]
        g          (float): Gravitational acceleration [m/s²]
        c_l        (float): Lift coefficient in cruise (-)
        c_d        (float): Drag coefficient in cruise (-)
        alpha_deg  (float): Angle of attack in cruise [deg]
        c          (float): Wing chord [m]
        b          (float): Wing span [m]
        rho        (float): Air density [kg/m³]

    Returns:
        float: Cruise speed [m/s]
    """
    alpha_rad = math.radians(alpha_deg)
    denominator = (c_l + c_d * math.tan(alpha_rad)) * c * b * rho
    return math.sqrt((2 * MTOM * g) / denominator)
