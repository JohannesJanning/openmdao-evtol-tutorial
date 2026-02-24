import math

def compute_mtom_actual(m_empty, m_battery, m_payload):
    """
    Compute the actual Maximum Take-Off Mass (MTOM) based on structural and payload components.

    Parameters:
        m_empty (float): Empty mass of the aircraft [kg]
        m_battery (float): Battery mass [kg]
        m_payload (float): Payload mass [kg]

    Returns:
        float: Actual MTOM [kg]
    """
    return m_empty + m_battery + m_payload
