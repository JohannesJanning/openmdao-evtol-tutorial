import math

def energy_climb(P_climb, t_climb):
    """
    Calculate energy required for climb phase.

    Parameters:
        P_climb (float): Power required during climb [W]
        t_climb (float): Duration of climb phase [s]

    Returns:
        float: Energy required for climb [Wh]
    """
    return P_climb * t_climb / 3600

