import math

def rotor_mass_per_unit(m_rotor, n_prop_vert, n_prop_hor):
    """
    Compute average rotor mass per propeller.

    Parameters:
        m_rotor (float): Total rotor mass [kg]
        n_prop_vert (int): Number of vertical propellers [-]
        n_prop_hor (int): Number of horizontal propellers [-]

    Returns:
        float: Mass per rotor [kg]
    """
    return m_rotor / (n_prop_vert + n_prop_hor)
