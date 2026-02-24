import math

def energy_cruise(P_cruise, t_cruise):
    """
    Calculate energy required for cruise phase.

    Parameters:
        P_cruise (float): Power required during cruise [W]
        t_cruise (float): Duration of cruise phase [s]

    Returns:
        float: Energy required for cruise [Wh]
    """
    return P_cruise * t_cruise / 3600