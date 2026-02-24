import math

def c_rate(P_req: float, E_battery: float) -> float:
    """
    Compute the discharge C-rate during cruise.

    Parameters:
        P_req_cruise (float): Power required during cruise [W]
        E_battery (float): Battery energy capacity [Wh]

    Returns:
        float: Discharge C-rate during cruise [1/h]
    """
    return P_req / E_battery
