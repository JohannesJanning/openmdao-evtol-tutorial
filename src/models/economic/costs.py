def battery_unit_cost(P_bat_s, E_battery_design):
    """
    Calculate the cost of a single battery.

    Parameters:
        P_bat_s (float): Battery price per kWh [€/kWh]
        E_battery_design (float): Battery design energy [Wh]

    Returns:
        float: Cost of one battery [€]
    """
    return P_bat_s * E_battery_design / 1000


def energy_cost_model(energy_trip, P_e):
    return energy_trip / 1000 * P_e

def navigation_cost_model(MTOM, unitrate, distance_trip_km):
    C_terminal = (MTOM / 1000 / 50) ** 0.7 * unitrate
    C_enroute = (MTOM / 1000 / 50) ** 0.5 * distance_trip_km / 100 * unitrate
    return C_terminal + C_enroute

def crew_cost_model(S_P, N_wd, T_D, U_pilot, N_AC, time_trip, DH):
    N_pilots = N_wd * T_D / (U_pilot * N_AC)
    time_ratio = time_trip / (T_D / DH * N_wd)
    return S_P * N_pilots * time_ratio

def wrap_maintenance_cost(time_trip):
    """
    Calculate wrap-rated maintenance cost per flight.

    Parameters:
        time_trip (float): Total trip time [s]

    Returns:
        float: Wrap-rated maintenance cost [€]
    """
    return 33 * time_trip / 3600


def battery_maintenance_cost(n_battery_required_annual, P_bat_s, E_battery_design, time_trip, DH, T_D, N_wd):
    """
    Calculate battery-related maintenance cost per flight.

    Parameters:
        n_battery_required_annual (float): Number of battery replacements per year
        P_bat_s (float): Battery price per kWh [€/kWh]
        E_battery_design (float): Battery design energy [Wh]
        time_trip (float): Total trip time [s]
        DH (float): Deadhead ratio [-]
        T_D (float): Daily operation time [s]
        N_wd (int): Number of working days per year [-]

    Returns:
        float: Battery-related maintenance cost per flight [€]
    """
    return n_battery_required_annual * P_bat_s * E_battery_design / 1000 * (time_trip * DH) / (T_D * N_wd)

def maintenance_cost_model(battery_maintenance_cost, wrap_maintenance_cost):
    return battery_maintenance_cost + wrap_maintenance_cost

def cash_operating_cost(C_energy, C_navigation, C_crew, C_maintenance):
    return C_energy + C_navigation + C_crew + C_maintenance

def ownership_cost_model(COC_flight, omega_empty, MTOM, P_s_empty, N_wd, T_D, time_trip, DH):
    return 0.06 * COC_flight + 0.0796 * omega_empty * MTOM * P_s_empty / (N_wd * T_D / (time_trip * DH))

def direct_operating_cost(COC_flight, COO_value_flight):
    return COC_flight + COO_value_flight

def indirect_operating_cost(COC_flight, omega_empty, MTOM, P_s_empty, N_wd, FC_d):
    return 0.233 * COC_flight + 0.0175 * omega_empty * MTOM * P_s_empty / (N_wd * FC_d)

def total_operating_cost(DOC_flight, IOC_value_flight):
    return DOC_flight + IOC_value_flight

def toc_per_seat_min(TOC_flight, N_s, time_trip):
    return TOC_flight / (N_s * time_trip / 60)

def toc_per_seat_km(TOC_flight, N_s, distance_trip_km):
    return TOC_flight / (N_s * distance_trip_km)



