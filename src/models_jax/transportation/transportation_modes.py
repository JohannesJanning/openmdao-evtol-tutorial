import numpy as np

def transportation_mode_comparison(
    t_tot,
    e_trip,
    D_trip,
    toc_flight,
    time_weight,
    co2_weight,
    energy_weight,
    costs_weight,
    gwp_flight,
    LF,
    N_s,
):
    """
    Compare eVTOL performance against conventional transport modes.

    This function computes a Figure of Merit (FoM) for different
    transportation modes (cars, buses, trains, aircraft, bicycles, and
    eVTOL) based on time, CO2 emissions, energy use, and cost. It adjusts
    distances for circuity (detour factors), applies realistic speed
    assumptions, and incorporates additional penalties (e.g. airport
    access time for airplanes). Each mode is then rated on a 1–10 scale
    relative to all others, and combined into a weighted FoM.

    Parameters
    ----------
    t_tot : float
        Total trip time of the eVTOL [s].
    e_trip : float
        Total trip energy consumption of the eVTOL [Wh].
    D_trip : float
        Direct mission distance [km].
    toc_flight : float
        Total operating cost per trip of the eVTOL [€].
    time_weight : float
        Weighting factor for time rating in FoM.
    co2_weight : float
        Weighting factor for CO2 rating in FoM.
    energy_weight : float
        Weighting factor for energy rating in FoM.
    costs_weight : float
        Weighting factor for cost rating in FoM.
    gwp_flight : float
        Global warming potential of the eVTOL trip [kg CO2e].
    LF : float
        Load factor of the eVTOL (fraction of seats filled, 0–1).
    N_s : int
        Number of passenger seats in the eVTOL.

    Returns
    -------
    results : list of dict
        Each entry corresponds to a transportation mode and contains:
        
        - "Mode (LF)" : str
            Mode description (with load factor where applicable).
        - "FoM" : float
            Weighted Figure of Merit score (higher is better).
        - "Time (min)" : float
            Total travel time [min].
        - "Time Rating (1–10)" : float
            Relative rating for time demand.
        - "CO2 (kg)" : float
            Total CO2 emissions [kg].
        - "CO2 Rating (1–10)" : float
            Relative rating for CO2.
        - "Energy (Wh)" : float
            Total energy consumption [Wh].
        - "Energy Rating (1–10)" : float
            Relative rating for energy demand.
        - "Cost (€)" : float
            Total operating cost [€].
        - "Cost Rating (1–10)" : float
            Relative rating for cost.

    Notes
    -----
    - Airplane trips include a fixed 120 min penalty for check-in and airport access.
    - Circuity factors are applied to represent detours in ground and rail transport.
    - Ratings are normalized so that the "best" performer achieves ~10, 
      the "worst" ~1, with linear scaling in between.
    - The FoM is computed as a weighted sum of the four criteria.
    """

    # eVTOL-specific parameters
    time_requirement = t_tot / 60  # minutes
    eVTOL_CO2_kg_skm = gwp_flight / (LF * N_s * D_trip)
    eVTOL_energy_consumption = e_trip / (LF * N_s * D_trip)
    eVTOL_costs_eur_skm = toc_flight / (LF * N_s * D_trip)

    # Modes
    modes = [
        "Airplane (100%)", "Gasonline Vehicle (20%)", "Diesel Vehicle (20%)",
        "Electric Vehicle (20%)", "Gasonline Vehicle (100%)",
        "Diesel Vehicle (100%)", "Public Bus (100%)", "Electric Vehicle (100%)",
        "Train (100%)", "Bicycle", "Airplane (79.6%)", "Diesel Vehicle (26%)",
        "Electric Vehicle (26%)", "Gasonline Vehicle (26%)",
        "Public Bus (60%)", "Train (50%)", "eVTOL",
    ]

    CO2_kg_skm = np.array([
        0.198, 0.157, 0.128, 0.065, 0.031, 0.026, 0.013, 0.013, 0.007, 0.000,
        0.249, 0.099, 0.050, 0.120, 0.022, 0.012, eVTOL_CO2_kg_skm,
    ])
    energy_consumption = np.array([
        216.06, 632.40, 480.00, 172.70, 126.48, 96.00, 49.42, 34.54, 57.02,
        -111.11, 340.99, 369.23, 132.85, 486.46, 82.84, 114.04,
        eVTOL_energy_consumption,
    ])
    costs_eur_skm = np.array([
        0.46, 0.117, 0.083, 0.105, 0.023, 0.017, 0.060, 0.021, 0.200, -0.491,
        0.579, 0.064, 0.081, 0.090, 0.104, 0.402, eVTOL_costs_eur_skm,
    ])

    # Speeds (km/h)
    velocities = {
        "Cars_up_to_60km": 60.0,
        "Cars_above_60km": 85.0,
        "Airplane_below_400km": 74.0,
        "Airplane_above_400km": 151.0,
        "Bicycle": 18.8,
        "Public_Bus_up_to_60km": 39.7,
        "Public_Bus_above_60km": 64.0,
        "Train_up_to_60km": 49.1,
        "Train_above_60km": 99.0,
    }

    # Circuity factors
    circuity = {
        "Cars_up_to_180km": 1.30,
        "Cars_above_180km": 1.20,
        "Train_all": 1.20,
        "Airplane_all": 1.05,
        "Bicycle_all": 1.28,
        "Bus_up_to_100km": 1.60,
        "Bus_above_100km": 1.25,
    }

    # Adjusted distances
    adjusted_dist = np.zeros(len(modes))
    for i, m in enumerate(modes):
        if "Vehicle" in m:
            adjusted_dist[i] = D_trip * (
                circuity["Cars_up_to_180km"]
                if D_trip <= 180 else circuity["Cars_above_180km"]
            )
        elif "Airplane" in m:
            adjusted_dist[i] = D_trip * circuity["Airplane_all"]
        elif "Bicycle" in m:
            adjusted_dist[i] = D_trip * circuity["Bicycle_all"]
        elif "Public Bus" in m:
            adjusted_dist[i] = D_trip * (
                circuity["Bus_up_to_100km"]
                if D_trip <= 100 else circuity["Bus_above_100km"]
            )
        elif "Train" in m:
            adjusted_dist[i] = D_trip * circuity["Train_all"]
        elif "eVTOL" in m:
            adjusted_dist[i] = D_trip

    # Totals
    CO2_total = CO2_kg_skm * adjusted_dist
    energy_total = energy_consumption * adjusted_dist
    costs_total = costs_eur_skm * adjusted_dist

    # Time demand
    time_demand = np.zeros(len(modes))
    for i, m in enumerate(modes):
        dist = adjusted_dist[i]
        if "Vehicle" in m:
            v = (
                velocities["Cars_up_to_60km"]
                if D_trip <= 60 else velocities["Cars_above_60km"]
            )
        elif "Airplane" in m:
            v = (
                velocities["Airplane_below_400km"]
                if D_trip <= 400 else velocities["Airplane_above_400km"]
            )
            time_demand[i] = dist / v * 60 + 120
            continue
        elif "Bicycle" in m:
            v = velocities["Bicycle"]
        elif "Public Bus" in m:
            v = (
                velocities["Public_Bus_up_to_60km"]
                if D_trip <= 60 else velocities["Public_Bus_above_60km"]
            )
        elif "Train" in m:
            v = (
                velocities["Train_up_to_60km"]
                if D_trip <= 60 else velocities["Train_above_60km"]
            )
        elif "eVTOL" in m:
            time_demand[i] = time_requirement
            continue
        time_demand[i] = dist / v * 60

    # Rating function
    def calc_rating(x):
        return (1 * (x - min(x)) - 10 * (x - max(x))) / (max(x) - min(x))

    time_rating = calc_rating(time_demand)
    co2_rating = calc_rating(CO2_total)
    energy_rating = calc_rating(energy_total)
    cost_rating = calc_rating(costs_total)

    # Figure of Merit
    FoM = (
        time_weight * time_rating
        + co2_weight * co2_rating
        + energy_weight * energy_rating
        + costs_weight * cost_rating
    )

    # Results table
    results = []
    for i, mode in enumerate(modes):
        results.append({
            "Mode (LF)": mode,
            "FoM": round(FoM[i], 3),

            "Time (min)": round(time_demand[i], 3),
            "Time Rating (1–10)": round(time_rating[i], 3),

            "CO2 (kg)": round(CO2_total[i], 3),
            "CO2 Rating (1–10)": round(co2_rating[i], 3),

            "Energy (Wh)": round(energy_total[i], 3),
            "Energy Rating (1–10)": round(energy_rating[i], 3),

            "Cost (€)": round(costs_total[i], 3),
            "Cost Rating (1–10)": round(cost_rating[i], 3),
        })

    return results
