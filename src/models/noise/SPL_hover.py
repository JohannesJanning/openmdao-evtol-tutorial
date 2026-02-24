import numpy as np
from scipy.special import jv
import math

def tonal_noise_hover(T_req_prop_hover, Power_req_hover, R_prop_hover, rho, n_prop_vert, n_blade_vert, params):
    """
    Compute total tonal Sound Pressure Level (SPL) for hovering rotors using the Gutin–Deming model.

    Parameters:
        T_req_prop_hover (float): Required thrust per hover propeller [N]
        Power_req_hover   (float): Power required in hover [W]
        R_prop_hover      (float): Radius of hover propeller [m]
        rho               (float): Air density [kg/m³]
        n_prop_vert       (int): Number of vertical propellers [-]
        n_blade_vert      (int): Number of blades per vertical propeller [-]
        params (module): Parameter module (e.g. `p`) with attributes:
                         - C_T_hover
                         - h_hover_ft

    Returns:
        float: Total SPL in hover [dB]
    """
    # Constants
    q_GTM = 1
    C_T = params.C_T_hover
    r_obs_ft = 250
    theta = np.pi / 2 + np.arcsin(100/r_obs_ft)  # observation angle in radians 
    r_obs = r_obs_ft / 3.28084   # observer distance from prop [m]
    OAT_K = 15 + 273.15
    R_sos = 287.5
    gamma = 1.4
    sos = np.sqrt(gamma * R_sos * OAT_K) # sos = speed of sound (m/s)

    # Rotational speed and torque
    n_rot_s = np.sqrt(T_req_prop_hover / (C_T * rho * (2 * R_prop_hover)**4))
    #n_rot_s = 0.65*sos / R_prop_hover
    n_rot_rpm = n_rot_s * 60
    n_rot_rad = 2 * np.pi * n_rot_rpm / 60    # rotational speed in rad/s
    Q = Power_req_hover / n_prop_vert / n_rot_rad  # Torque per propeller [Nm]

    # Acoustic wavenumber and Bessel function
    R_e = 0.8 * R_prop_hover
    k = q_GTM * n_blade_vert * n_rot_rad / sos
    J_qn = jv(q_GTM * n_blade_vert, k * R_e * np.sin(theta))

    # Acoustic pressure
    prefactor = (q_GTM * n_blade_vert * n_rot_rad) / (2 * np.sqrt(2) * np.pi * sos * r_obs)
    bracket_term = -T_req_prop_hover * np.cos(theta) + Q * (sos / (n_rot_rad * R_e**2))
    p_rms = prefactor * abs(bracket_term) * J_qn

    # SPL for one propeller
    p_ref = 20e-6  # reference pressure [Pa]
    SPL_prop = 20 * np.log10(p_rms / p_ref)

    # Total SPL from multiple propellers
    SPL_total = SPL_prop + 10 * np.log10(n_prop_vert)

    return SPL_total
