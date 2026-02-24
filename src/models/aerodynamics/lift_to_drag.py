import math

def ld_calculation(c_l, c_d):
    """
    Calculate the lift-to-drag ratio (L/D).

    Parameters:
        c_l (float): Lift coefficient (dimensionless)
        c_d (float): Drag coefficient (dimensionless)

    Returns:
        float: Lift-to-drag ratio (L/D)
    """
    return c_l / c_d
