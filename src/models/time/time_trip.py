import math

def total_trip_time(t_hover, t_climb, t_cruise):
    """
    Calculate the total time for a complete trip.

    Parameters:
        t_hover (float): Total hover time (e.g. takeoff + landing) [s]
        t_climb (float): Time spent climbing [s]
        t_cruise (float): Time spent cruising [s]

    Returns:
        float: Total trip time [s]
    """
    return t_hover + t_climb + t_cruise
