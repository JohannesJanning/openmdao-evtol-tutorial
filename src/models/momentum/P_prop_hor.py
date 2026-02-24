import math

def power_per_propeller(P_total, n_prop):
    """
    Calculate required power per propeller.

    Parameters:
        P_total (float): Total required power [W]
        n_prop (int): Number of propellers [-]

    Returns:
        float: Power required per propeller [W]
    """
    return P_total / n_prop
