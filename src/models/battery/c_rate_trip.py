import math

def c_rate_average(C_hover: float, C_climb: float, C_cruise: float,
                   t_hover: float, t_climb: float, t_cruise: float, t_trip: float) -> float:
    """
    Compute the average discharge C-rate over the trip.

    Parameters:
        C_hover (float): Hover C-rate [1/h]
        C_climb (float): Climb C-rate [1/h]
        C_cruise (float): Cruise C-rate [1/h]
        t_hover (float): Hover time [s]
        t_climb (float): Climb time [s]
        t_cruise (float): Cruise time [s]
        t_trip (float): Total trip time [s]

    Returns:
        float: Average discharge C-rate [-]
    """
    return (C_hover * t_hover + C_climb * t_climb + C_cruise * t_cruise) / t_trip
