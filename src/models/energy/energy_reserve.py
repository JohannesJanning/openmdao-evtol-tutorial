import math

def energy_reserve(P_cruise, t_reserve):
    """
    Calculate energy required for reserve segment.

    Parameters:
        P_cruise (float): Cruise power used as reserve baseline [W]
        t_reserve (float): Reserve duration [s]

    Returns:
        float: Reserve energy [Wh]
    """
    return P_cruise * t_reserve / 3600
