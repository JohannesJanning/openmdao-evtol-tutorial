import math

def roc_calculation(alpha_deg_climb: float, v_climb: float) -> float:
    """
    Calculate the rate of climb (ROC).

    Parameters:
    alpha_deg_climb (float): Climb angle of attack in degrees
    v_climb            (float): Climb speed (m/s)

    Returns:
    float: Rate of climb (m/s)
    """
    # Convert climb angle to radians
    alpha_rad = math.radians(alpha_deg_climb)
    # ROC is vertical component of velocity
    return math.sin(alpha_rad) * v_climb
