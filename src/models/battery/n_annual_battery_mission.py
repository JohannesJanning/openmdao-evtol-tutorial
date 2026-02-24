def number_of_battery_required_annually(
    N_cycles_available: float,
    N_wd: float,
    T_D: float,
    time_trip: float,
    DH: float
) -> float:
    """
    Calculate the number of battery cycles required annually.

    Parameters:
        N_cycles_available (float): Available battery discharge cycles [-]
        N_wd (float): Number of working days per year [-]
        T_D (float): Daily operating time [s]
        time_trip (float): Total time per trip [s]
        DH (float): Deadhead ratio [-]

    Returns:
        float: Number of batteries required per year [-]
    """
    return (1 / N_cycles_available) * N_wd * T_D / (time_trip * DH)
