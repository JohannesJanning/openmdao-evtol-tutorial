import math

def power_total_required(V, T_req_total, T_req_per_prop, rho, A_prop, n_props, eta_c):
    """
    Calculate the total power required during climb.

    Parameters:
        V_climb (float): Climb velocity [m/s]
        T_req_total (float): Total required thrust in climb [N]
        T_req_per_prop (float): Required thrust per propeller [N]
        rho (float): Air density [kg/m³]
        A_prop (float): Propeller disk area [m²]
        n_props (int): Number of propellers [-]
        eta_c (float): Propulsive efficiency in climb [-]

    Returns:
        float: Total power required in climb [W]
    """
    v_i = -V / 2 + math.sqrt((V / 2)**2 + T_req_per_prop / (2 * rho * A_prop))
    return (T_req_total * V + T_req_per_prop * v_i * n_props) / eta_c
