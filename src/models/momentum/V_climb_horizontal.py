import math

def horizontal_climb_speed(V_climb, theta_deg_climb):
    """
    Calculate horizontal component of climb velocity.

    Parameters:
        V_climb (float): Total climb velocity [m/s]
        theta_deg_climb (float): Climb angle [deg]

    Returns:
        float: Horizontal climb speed [m/s]
    """
    theta_rad = math.radians(theta_deg_climb)
    return V_climb * math.cos(theta_rad)
