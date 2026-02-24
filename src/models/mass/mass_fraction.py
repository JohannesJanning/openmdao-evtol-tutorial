import math

def mass_fraction(mass_item: float, MTOM: float) -> float:
    """
    Calculate the mass fraction of a given component relative to MTOM.

    Parameters:
        mass_item (float): Mass of the component [kg]
        MTOM      (float): Maximum Take-Off Mass [kg]

    Returns:
        float: Mass fraction (dimensionless, e.g. 0.25 means 25%)
    """
    if MTOM == 0:
        raise ValueError("MTOM must be greater than zero to calculate mass fraction.")
    return mass_item / MTOM
