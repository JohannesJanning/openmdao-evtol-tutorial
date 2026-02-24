import math

def total_thrust_required_climb(D_climb, MTOM, g, theta_deg_climb):
    """
    Calculate the total thrust required during climb.

    Parameters:
        D_climb (float): Aerodynamic drag during climb [N]
        MTOM (float): Maximum takeoff mass [kg]
        g (float): Gravitational acceleration [m/s²]
        theta_deg_climb (float): Climb angle [deg]

    Returns:
        float: Total required thrust in climb [N]
    """
    theta_rad = math.radians(theta_deg_climb)
    return D_climb + MTOM * g * math.sin(theta_rad)
