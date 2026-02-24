import math

def energy_total_required(E_trip, E_reserve):
    """
    Calculate total required energy including reserve.

    Parameters:
        E_trip (float): Net-trip energy without reserve [Wh]
        E_reserve (float): Reserve energy [Wh]

    Returns:
        float: Total energy requirement [Wh]
    """
    return E_trip + E_reserve
