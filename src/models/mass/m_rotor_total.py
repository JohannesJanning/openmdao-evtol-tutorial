import math

def rotor_mass(n_prop_vert, n_prop_hor, R_prop_hover, R_prop_cruise):
    """
    Estimate the total rotor mass.

    Parameters:
        n_prop_vert (int): Number of vertical (hover) propellers [-]
        n_prop_hor (int): Number of horizontal (cruise) propellers [-]
        R_prop_hover (float): Radius of hover propellers [m]
        R_prop_cruise (float): Radius of cruise propellers [m]
        k_evtol (float): Empirical scaling constant for eVTOL rotor mass [-]
        div (float): Reduction factor for lightweight rotor tech [0-1]

    Returns:
        float: Total rotor mass [kg]
    """
    k_evtol= 13            #22.649
    div=0

    term_hover = n_prop_vert * k_evtol * (1 - div) * (0.7484 * R_prop_hover**1.2 - 0.0403 * R_prop_hover)
    term_cruise = n_prop_hor * k_evtol * (1 - div) * (0.7484 * R_prop_cruise**1.2 - 0.0403 * R_prop_cruise)
    return term_hover + term_cruise
