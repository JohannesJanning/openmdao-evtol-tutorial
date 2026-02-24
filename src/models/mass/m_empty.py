import math

def empty_mass(m_wing, m_motor, m_rotor, m_crew, m_furnish, m_fuselage, m_system, m_gear):
    """
    Calculate empty mass of the eVTOL (excluding battery).

    Parameters:
        m_wing (float): Wing mass [kg]
        m_motor (float): Motor mass [kg]
        m_rotor (float): Rotor mass [kg]
        m_crew (float): Crew mass [kg]
        m_furnish (float): Furnish mass [kg]
        m_fuselage (float): Fuselage mass [kg]
        m_system (float): Systems mass [kg]
        m_gear (float): Landing gear mass [kg]

    Returns:
        float: Empty mass (excluding battery) [kg]
    """
    return m_wing + m_motor + m_rotor + m_crew + m_furnish + m_fuselage + m_system + m_gear
