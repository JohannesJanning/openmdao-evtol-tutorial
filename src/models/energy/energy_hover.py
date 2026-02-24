import math

def energy_hover(P_hover, t_hover):
    """
    Calculate energy required for hover phase.

    Parameters:
        P_hover (float): Power required during hover [W]
        t_hover (float): Duration of hover phase [s]

    Returns:
        float: Energy required for hover [Wh]
    """
    return P_hover * t_hover / 3600
