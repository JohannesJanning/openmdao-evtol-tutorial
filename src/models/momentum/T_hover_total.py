import math

def total_thrust_required_hover(MTOM, g):
    """
    Calculate total thrust required to hover.

    Parameters:
        MTOM (float): Maximum takeoff mass [kg]
        g (float): Gravitational acceleration [m/s²]

    Returns:
        float: Total thrust required in hover [N]
    """
    return MTOM * g
