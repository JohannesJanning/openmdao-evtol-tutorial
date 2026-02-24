import math

def compute_time_cruise(distance_total, v_climb_hor, t_climb, v_cruise):
    """
    Compute the cruise time based on total trip distance, climb dynamics, and cruise speed.

    Parameters:
        distance_total (float): Total trip distance [m]
        v_climb_hor (float): Horizontal climb speed [m/s]
        t_climb (float): Time spent climbing [s]
        v_cruise (float): Cruise speed [m/s]

    Returns:
        float: Time spent cruising [s]
    """
    distance_climb = v_climb_hor * t_climb
    distance_cruise = distance_total - distance_climb
    return distance_cruise / v_cruise
