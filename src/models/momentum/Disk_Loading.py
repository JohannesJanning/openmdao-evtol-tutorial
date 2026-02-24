import math

def disk_loading_hover(T_total_hover, A_disk_total):
    """
    Calculate total disk loading during hover.

    Parameters:
        T_total_hover (float): Total required hover thrust [N]
        A_disk_total (float): Total rotor disk area [m²]

    Returns:
        float: Disk loading during hover [N/m²]
    """
    return T_total_hover / A_disk_total
