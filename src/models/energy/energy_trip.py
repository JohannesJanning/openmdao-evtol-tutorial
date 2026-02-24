import math

def energy_total_trip(E_hover, E_climb, E_cruise):
    """
    Calculate total trip energy (excluding reserve).

    Parameters:
        E_hover (float): Energy for hover [Wh]
        E_climb (float): Energy for climb [Wh]
        E_cruise (float): Energy for cruise [Wh]

    Returns:
        float: Total energy for main trip [Wh]
    """
    return E_hover + E_climb + E_cruise

