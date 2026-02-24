import math

def depth_of_discharge(energy_trip: float, E_battery: float) -> float:
    """
    Compute the depth of discharge (DoD) of the battery for the trip.

    Parameters:
        energy_trip (float): Energy consumed during the trip [Wh]
        E_battery (float): Battery design energy capacity [Wh]

    Returns:
        float: Depth of discharge [-]
    """
    return energy_trip / E_battery
