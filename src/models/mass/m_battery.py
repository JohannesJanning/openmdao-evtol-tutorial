import math

def battery_mass(energy_total_required, rho_bat):
    """
    Estimate battery mass based on total required energy and battery energy density.

    Parameters:
        energy_total_required (float): Total required energy for the mission [Wh]
        rho_bat (float): Battery energy density [Wh/kg]

    Returns:
        float: Estimated battery mass [kg]
    """
    return energy_total_required / (0.64 * rho_bat)
