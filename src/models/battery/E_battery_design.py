import math

def battery_energy_capacity(rho_bat: float, m_battery: float) -> float:
    """
    Compute the battery design energy capacity.

    Parameters:
        rho_bat (float): Battery energy density [Wh/kg]
        m_battery (float): Battery mass [kg]

    Returns:
        float: Battery energy capacity [Wh]
    """
    return rho_bat * m_battery
