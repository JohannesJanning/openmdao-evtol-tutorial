import numpy as np

def rotation_speed_rpm(T_req_prop, rho, R_prop, C_T):
    """
    Calculate the rotation speed of the cruise propeller based on thrust coefficient.

    Parameters:
        T_req_prop_cruise (float): Required thrust per propeller [N]
        rho                (float): Air density [kg/m³]
        R_prop_cruise      (float): Propeller radius [m]
        C_T                (float): Thrust coefficient (dimensionless)

    Returns:
        float: Rotation speed [RPM]
    """
    n_rotation_s = np.sqrt(T_req_prop / (C_T * rho * (2 * R_prop)**4))  # rotations per second
    return n_rotation_s * 60  # convert to RPM
