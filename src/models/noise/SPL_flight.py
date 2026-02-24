import numpy as np

def propeller_SPL(n_rpm, R_prop, T_req_per_prop, rho, A_prop, V_forward, h_obs_ft):
    """
    Calculate the Sound Pressure Level (SPL) using the Schegel–King–Mull (SKM) model.

    Parameters:
        n_rpm           (float): Propeller rotation speed [RPM]
        R_prop          (float): Propeller radius [m]
        T_req_per_prop  (float): Required thrust per propeller [N]
        rho             (float): Air density [kg/m³]
        A_prop          (float): Propeller disk area [m²]
        V_forward       (float): Forward flight speed [m/s]
        h_obs_ft        (float): Observation distance in feet (typically 300 ft, or use 50 ft for hover)

    Returns:
        float: Sound Pressure Level (SPL) [dB]
    """
    # Constants for SKM model
    K = 6.1e-27          # Empirical constant
    P_ref_sq = 1e-16     # Reference pressure squared (Pa²)
    CL_ref = 0.4         # Reference lift coefficient

    # Rotational speed in rad/s
    n_rad = 2 * np.pi * n_rpm / 60
    # Tip speed
    V_tip = n_rad * R_prop
    # Speed at 70% blade radius
    V_07 = 0.7 * V_tip
    # Propeller lift coefficient
    C_L = 2 * T_req_per_prop / (rho * A_prop * V_forward**2)

    # SKM noise terms
    term1 = 10 * np.log10((K * A_prop * V_07**6) / P_ref_sq)
    term2 = 20 * np.log10(C_L / CL_ref)
    correction = 20 * np.log10(300 / h_obs_ft)

    SPL = term1 + term2 + correction
    return SPL
