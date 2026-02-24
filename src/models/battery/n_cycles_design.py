import math

def battery_cycle_life(DOD: float, C_rate_avg: float, c_charge: float) -> float:
    """
    Estimate the number of available battery discharge cycles (N_cycles_available)
    based on depth of discharge, average C-rate, and charging rate.

    Parameters:
        DOD (float): Depth of discharge [-]
        C_rate_avg (float): Average discharge C-rate [1/h]
        c_charge (float): Charging rate (C-rate) [1/h]

    Returns:
        float: Estimated number of available discharge cycles
    """
    a_n = -5986.8421   # empirical parameter for DoD effect
    b_n = 11776.3158   # empirical parameter for DoD effect
    c_n = 1.1          # exponent for discharge rate penalty
    d_n = 1.2          # exponent for charging rate penalty

    return (a_n * DOD + b_n) * (1 / C_rate_avg**c_n) * (0.5 / c_charge**d_n)
