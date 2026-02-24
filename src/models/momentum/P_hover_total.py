import math

def power_required_hover(sigma_hover, T_req_total_hover, rho, eta_h):
    """
    Calculate total power required during hover.

    Parameters:
        T_req_total_hover (float): Required thrust total in hover [N]
        n_prop (int): Number of hover propellers [-]
        rho (float): Air density [kg/m³]
        eta_h (float): Hover efficiency [-]

    Returns:
        float: Total power required in hover [W]
    """
    v_i = math.sqrt(sigma_hover / (2 * rho))
    return T_req_total_hover * v_i / eta_h
