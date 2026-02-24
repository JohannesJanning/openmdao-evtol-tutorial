import math

def total_thrust_required_cruise(D_cruise, alpha_deg):
    """
    Calculate total thrust required during cruise flight.

    Parameters:
        D_cruise (float): Aerodynamic drag in cruise [N]
        alpha_deg (float): Angle of attack in cruise [deg]

    Returns:
        float: Total required thrust in cruise [N]
    """
    alpha_rad = math.radians(alpha_deg)
    return D_cruise / math.cos(alpha_rad)
