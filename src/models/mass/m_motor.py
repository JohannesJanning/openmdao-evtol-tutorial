import math

def motor_mass(P_hover, P_climb, n_vert, n_hor):
    """
    Estimate total motor mass using empirical sizing from power requirements.

    Parameters:
        P_hover (float): Total power required in hover [W]
        P_climb (float): Total power required in climb [W]
        n_vert (int): Number of hover propellers
        n_hor (int): Number of cruise propellers

    Returns:
        float: Estimated total motor mass [kg]
    """
    hp_hover = P_hover / (n_vert * 745.7) * 1.5
    hp_climb = P_climb / (n_hor * 745.7) * 1.5
    
    m_motor_vert = n_vert * 0.6756 * hp_hover**0.783
    m_motor_hor = n_hor * 0.6756 * hp_climb**0.783

    return m_motor_vert + m_motor_hor
