import math

def climb_time(h_cruise, h_hover, roc):
    """
    Calculate time required for climb phase.

    Parameters:
        h_cruise (float): Cruise altitude above ground [m]
        h_hover (float): Hover altitude above ground [m]
        roc (float): Rate of climb [m/s]

    Returns:
        float: Time required to climb from hover to cruise altitude [s]
    """
    return (h_cruise - h_hover) / roc
