import math

def thrust_per_propeller(total_thrust, n_propellers):
    """
    Distribute total thrust equally among propellers.

    Parameters:
        total_thrust (float): Total required thrust [N]
        n_propellers (int): Number of propellers [-]

    Returns:
        float: Required thrust per propeller [N]
    """
    return total_thrust / n_propellers
