import pandas as pd
import os
from datetime import datetime


####
# LIBRARIES
####
import src.parameters as p
from src.models.aerodynamics import AR_calculation, drag_calculation, lift_calculation, ld_calculation, roc_calculation, cl_calculation, cd_calculation
from src.models.battery import c_rate, c_rate_average, depth_of_discharge, battery_energy_capacity, number_of_battery_required_annually, battery_cycle_life
from src.models.mass import interior_mass, gear_mass, fuselage_mass, battery_mass, motor_mass, wing_mass, rotor_mass, rotor_mass_per_unit, system_mass, empty_mass, compute_mtom_actual, mtom_iteration_loop, mass_fraction
from src.models.energy import energy_hover, energy_climb, energy_cruise, energy_total_trip, energy_reserve, energy_total_required
from src.models.time import climb_time, compute_time_cruise, total_trip_time
from src.models.momentum import cruise_speed, climb_speed, horizontal_climb_speed, total_thrust_required_climb, thrust_per_propeller, propeller_disk_area, power_total_required, power_per_propeller, total_thrust_required_cruise, disk_loading_hover, total_thrust_required_hover, power_required_hover
from src.models.noise import rotation_speed_rpm, propeller_SPL, tonal_noise_hover, compute_ct
from src.models.operations import turnaround_time, time_efficiency_ratio, daily_flight_hours, daily_flight_cycles, annual_flight_hours, annual_flight_cycles
from src.models.gwp import battery_annual_ops_gwp, battery_flight_cycle_gwp, battery_lifecycle_gwp, gwp_annual, gwp_battery_fraction, gwp_energy_fraction, gwp_flight, gwp_operational_per_cycle
from src.models.economic import energy_cost_model, navigation_cost_model, crew_cost_model, cash_operating_cost, ownership_cost_model, direct_operating_cost, indirect_operating_cost, total_operating_cost, toc_per_seat_min, toc_per_seat_km, maintenance_cost_model, wrap_maintenance_cost, battery_maintenance_cost, battery_unit_cost, revenue_per_flight, ticket_price_per_passenger, profit_per_flight, annual_profit, revenue_per_flight_pm
from src.models.transportation.transportation_modes import transportation_mode_comparison




def full_model_evaluation(b, c, R_prop_cruise, R_prop_hover, rho_bat, c_charge, p):
    """
    Compute all metrics from the model and return them in a structured format.
    """
    ############################################################################
    # AERODYNAMIC MODEL
    ############################################################################
    # Calculate Aspect Ratio 
    AR = AR_calculation(b, c)
    # Calculate coefficients in cruise
    c_l_cruise = cl_calculation(p.alpha_deg_cruise, AR, p.c_l_0, p.e)
    c_d_cruise = cd_calculation(c_l_cruise, AR, p.c_d_min, p.e)
    LD_cruise = ld_calculation(c_l_cruise, c_d_cruise)
    # Calculate coefficients in cruise
    c_l_climb = cl_calculation(p.alpha_deg_climb, AR, p.c_l_0, p.e)
    c_d_climb = cd_calculation(c_l_climb, AR, p.c_d_min, p.e)
    LD_climb = ld_calculation(c_l_climb, c_d_climb)

    MTOM_converged = mtom_iteration_loop(p.MTOM_initial,b, c, R_prop_cruise, R_prop_hover, rho_bat, p, verbose=False)
    MTOM = MTOM_converged


    ############################################################################
    # !!!!! - COMPUTATION FOR OUPUT GENERATION WITH ITRATED MTOM
    ############################################################################

    #
    # Momentum Theory Model & Force Balance 
    #
    V_cruise = cruise_speed(MTOM, p.g, c_l_cruise, c_d_cruise, p.alpha_deg_cruise, c, b, p.rho)
    V_climb = climb_speed(MTOM, p.g, p.theta_deg_climb, c_l_climb, c, b, p.rho)
    V_climb_hor = horizontal_climb_speed(V_climb, p.theta_deg_climb)
    D_cruise = drag_calculation(p.rho, V_cruise, c, b, c_d_cruise)
    D_climb = drag_calculation(p.rho, V_climb, c, b, c_d_climb)
    A_prop_hor = propeller_disk_area(R_prop_cruise)
    # Climb Power required
    T_req_total_climb = total_thrust_required_climb(D_climb, MTOM, p.g, p.theta_deg_climb)
    T_req_prop_climb = thrust_per_propeller(T_req_total_climb, p.n_prop_hor)
    P_req_total_climb = power_total_required(V_climb, T_req_total_climb, T_req_prop_climb, p.rho, A_prop_hor, p.n_prop_hor, p.eta_c)
    P_req_prop_climb = power_per_propeller(P_req_total_climb, p.n_prop_hor)
    # Cruise Power required
    T_req_total_cruise = total_thrust_required_cruise(D_cruise, p.alpha_deg_cruise)
    T_req_prop_cruise = thrust_per_propeller(T_req_total_cruise, p.n_prop_hor)
    P_req_total_cruise = power_total_required(V_cruise, T_req_total_cruise, T_req_prop_cruise, p.rho, A_prop_hor, p.n_prop_hor, p.eta_c)
    P_req_prop_cruise = power_per_propeller(P_req_total_cruise, p.n_prop_hor)
    # Hover Power required
    A_hover_disk_prop = propeller_disk_area(R_prop_hover)
    T_req_total_hover = total_thrust_required_hover(MTOM, p.g)
    T_req_prop_hover = thrust_per_propeller(T_req_total_hover, p.n_prop_vert)
    sigma_hover = disk_loading_hover(T_req_prop_hover, A_hover_disk_prop)
    P_req_total_hover = power_required_hover(sigma_hover, T_req_total_hover, p.rho, p.eta_h)
    P_req_prop_hover = power_per_propeller(P_req_total_hover, p.n_prop_vert)
    #
    # Trip Time & Distance Model / ROC 
    #
    ROC = roc_calculation(p.alpha_deg_climb, V_climb)
    t_climb = climb_time(p.h_cruise, p.h_hover, ROC)
    t_cruise = compute_time_cruise(p.distance_trip, V_climb_hor, t_climb, V_cruise)
    t_trip = total_trip_time(p.time_hover, t_climb, t_cruise)
    #
    # Energy Model  
    #
    e_hover = energy_hover(P_req_total_hover, p.time_hover)
    e_climb = energy_climb(P_req_total_climb, t_climb)
    e_cruise = energy_cruise(P_req_total_cruise, t_cruise)
    e_trip = energy_total_trip(e_hover, e_climb, e_cruise)
    e_reserve = energy_reserve(P_req_total_cruise, p.time_reserve)
    e_total_required = energy_total_required(e_trip, e_reserve)
    #
    # Mass Model  
    #
    m_interior = interior_mass(MTOM, p.rho, V_cruise)
    m_gear = gear_mass(MTOM, R_prop_cruise, p.r_fus_m)
    m_fuselage = fuselage_mass(MTOM, p.l_fus_m, p.r_fus_m, p.rho, V_cruise)
    m_motor = motor_mass(P_req_total_hover, P_req_total_climb, p.n_prop_vert, p.n_prop_hor)
    m_rotor = rotor_mass(p.n_prop_vert, p.n_prop_hor, R_prop_hover, R_prop_cruise)
    m_wing = wing_mass(MTOM, V_cruise, b, c, p.rho)
    m_system = system_mass(MTOM, p.l_fus_m, b)
    m_rotor_unit = rotor_mass_per_unit(m_rotor, p.n_prop_vert, p.n_prop_hor)
    m_empty = empty_mass(m_wing, m_motor, m_rotor, p.m_crew, m_interior, m_fuselage, m_system, m_gear)
    m_battery = battery_mass(e_total_required, rho_bat)
    mtom_model_check = compute_mtom_actual(m_empty, m_battery, p.m_pay)

    # fractions 
    omega_empty = mass_fraction(m_empty, MTOM)
    omega_battery = mass_fraction(m_battery, MTOM)

    ############################################################################
    # !!!!! END OF MTOM CHECKING 
    ############################################################################

    ############################################################################
    # NOISE MODEL
    ############################################################################
    # SCHLEGEL-KING-MULL (SKM) MODEL - CRUISE & CLIMB - BROADBAND NOISE 
    C_T_hor = compute_ct(T_req_prop_climb, p.rho, V_climb, R_prop_cruise)
    rpm_cruise = rotation_speed_rpm(T_req_prop_cruise, p.rho, R_prop_cruise, p.C_T)
    SPL_cruise = propeller_SPL(rpm_cruise, R_prop_cruise, T_req_prop_cruise, p.rho, A_prop_hor, V_cruise, p.r_obs_ft)
    rpm_climb = rotation_speed_rpm(T_req_prop_climb, p.rho, R_prop_cruise, p.C_T)
    SPL_climb = propeller_SPL(rpm_climb, R_prop_cruise, T_req_prop_climb, p.rho, A_prop_hor, V_climb, p.r_obs_ft)
    # GUTIN-DEMING MODEL (GTM) - HOVER - TONAL NOISE 
    rpm_hover = rotation_speed_rpm(T_req_prop_hover, p.rho, R_prop_hover, p.C_T_hover)
    SPL_hover = tonal_noise_hover(T_req_prop_hover, P_req_prop_hover, R_prop_hover, p.rho, p.n_prop_vert, p.n_blade_vert, p)

    ############################################################################
    # BATTERY LIFE MODEL
    ############################################################################
    E_battery_design = battery_energy_capacity(rho_bat, m_battery)
    c_rate_hover = c_rate(P_req_total_hover, E_battery_design)
    c_rate_climb = c_rate(P_req_total_climb, E_battery_design)
    c_rate_cruise = c_rate(P_req_total_cruise, E_battery_design)
    c_rate_trip = c_rate_average(c_rate_hover, c_rate_climb, c_rate_cruise, p.time_hover, t_climb, t_cruise, t_trip)
    DOD = depth_of_discharge(e_trip, E_battery_design)
    n_battery_lifecycle = battery_cycle_life(DOD, c_rate_trip, c_charge)

    ############################################################################
    # OPERATIONS MODEL
    ############################################################################
    t_turnaround = turnaround_time(c_charge, DOD)
    te_r = time_efficiency_ratio(t_turnaround, t_trip)  # time efficiency ratio
    FC_day = daily_flight_cycles(p.T_D, t_trip, te_r)
    FC_annual = annual_flight_cycles(p.N_wd, FC_day)
    FH_day = daily_flight_hours(FC_day, t_trip)
    FH_annual = annual_flight_hours(FC_annual, t_trip)

    ############################################################################
    # GWP MODEL
    ############################################################################
    n_annual_battery_required = number_of_battery_required_annually(n_battery_lifecycle, p.N_wd, p.T_D, t_trip, te_r)
    gwp_battery_lifecycle = battery_lifecycle_gwp(p.GWP_battery, E_battery_design)
    gwp_battery_annual = battery_annual_ops_gwp(n_annual_battery_required, gwp_battery_lifecycle)
    gwp_battery_cycle = battery_flight_cycle_gwp(gwp_battery_annual, FC_annual)
    gwp_energy_cycle = gwp_operational_per_cycle(e_trip, p.GWP_energy)
    gwp_total_cycle = gwp_flight(gwp_energy_cycle, gwp_battery_cycle)
    gwp_total_annual = gwp_annual(gwp_total_cycle, FC_annual)
    omega_gwp_energy = gwp_energy_fraction(e_trip, p.GWP_energy, gwp_total_cycle)
    omega_gwp_battery = gwp_battery_fraction(omega_gwp_energy)

    ############################################################################
    # ECONOMIC MODEL
    ############################################################################
    # COST MODEL per trip   
    cost_energy = energy_cost_model(e_trip, p.P_e)
    cost_nav = navigation_cost_model(MTOM, p.unitrate, p.distance_trip_km)
    cost_crew = crew_cost_model(p.S_P, p.N_wd, p.T_D, p.U_pilot, p.N_AC, t_trip, te_r)
    cost_maint_wrap = wrap_maintenance_cost(t_trip)
    cost_battery_unit = battery_unit_cost(p.P_bat_s, E_battery_design)
    cost_maint_battery = battery_maintenance_cost(n_annual_battery_required, p.P_bat_s , E_battery_design, t_trip, te_r, p.T_D, p.N_wd)
    cost_maint_total = maintenance_cost_model(cost_maint_battery, cost_maint_wrap)
    coc = cash_operating_cost(cost_energy, cost_nav, cost_crew, cost_maint_total)
    coo = ownership_cost_model(coc, omega_empty, MTOM, p.P_s_empty, p.N_wd, p.T_D, t_trip, te_r)
    doc = direct_operating_cost(coc, coo)
    ioc = indirect_operating_cost(coc, omega_empty, MTOM, p.P_s_empty, p.N_wd, FC_day)
    toc = total_operating_cost(doc, ioc)
    toc_as_km = toc_per_seat_km(toc, p.N_s, p.distance_trip_km)
    toc_as_min = toc_per_seat_min(toc, p.N_s, t_trip)
    # REVENUE MODEL per trip 
    revenue_flight = revenue_per_flight(p.fare_km, p.distance_trip_km, p.N_s, p.LF)
    #revenue_flight_pm = revenue_per_flight_pm(toc, p.pm)
    # PROFIT MODEL
    ticket_cost_for_pax = ticket_price_per_passenger(revenue_flight, p.N_s, p.LF)
    profit_flight = profit_per_flight(revenue_flight , toc)
    profit_annual = annual_profit(revenue_flight, toc, FC_annual)

    ############################################################################
    # TRANSPORTATION MODE COMPARISON FOM
    ############################################################################


    comparison_table = transportation_mode_comparison(
        t_tot=t_trip,
        e_trip=e_trip,
        D_trip=p.distance_trip_km,
        toc_flight = toc,
        time_weight=0.3333, 
        co2_weight=0.3333,
        energy_weight=0,
        costs_weight=0.3333, 
        gwp_flight= gwp_total_cycle,
        LF = p.LF, 
        N_s = p.N_s
    )




    results = {
        "DESIGN VARIABLE": {
        "Wing span": (round(b, 3), "m"),
        "Chrod length": (round(c, 3), "m"),
        "Pusher rotor radius": (round(R_prop_cruise, 3), "m"),
        "Hover rotor radius": (round(R_prop_hover, 3), "m"),
        "battery energy density": (round(rho_bat, 3), "Wh/kg"),
        "Charging C-rate": (round(c_charge, 3), "1/h"),
    },

    "AERODYNAMICS": {
        "Wing Aspect Ratio": (round(AR, 3), "-"),
        "Lift coefficient (cruise)": (round(c_l_cruise, 3), "-"),
        "Drag coefficient (cruise)": (round(c_d_cruise, 3), "-"),
        "Lift-to-drag Ratio (cruise)": (round(LD_cruise, 3), "-"),
        "Drag (cruise)": (round(D_cruise, 3), "N"),
        "Lift coefficient (climb)": (round(c_l_climb, 3), "-"),
        "Drag coefficient (climb)": (round(c_d_climb, 3), "-"),
        "Lift-to-drag Ratio (climb)": (round(LD_climb, 3), "-"),
        "Drag (climb)": (round(D_climb, 3), "N"),
    },

    "SPEED MODEL": {
        "Speed (cruise)": (round(V_cruise, 3), "m/s"),
        "Speed (climb)": (round(V_climb, 3), "m/s"),
        "Horizontal Speed (climb)": (round(V_climb_hor, 3), "m/s"),
        "ROC (Rate of Climb)": (round(ROC, 3), "m/s"),
    },

    "THRUST & POWER MODEL": {
        "Total Thrust req. (climb)": (round(T_req_total_climb, 3), "N"),
        "Propeller Thrust req. (climb)": (round(T_req_prop_climb, 3), "N"),
        "Total Power req. (climb)": (round(P_req_total_climb, 3), "W"),
        "Propeller Power req. (climb)": (round(P_req_prop_climb, 3), "W"),
        "Total Thrust req. (cruise)": (round(T_req_total_cruise, 3), "N"),
        "Propeller Thrust req. (cruise)": (round(T_req_prop_cruise, 3), "N"),
        "Total Power req. (cruise)": (round(P_req_total_cruise, 3), "W"),
        "Propeller Power req. (cruise)": (round(P_req_prop_cruise, 3), "W"),
        "Total Thrust req. (hover)": (round(T_req_total_hover, 3), "N"),
        "Propeller Thrust req. (hover)": (round(T_req_prop_hover, 3), "N"),
        "Total Power req. (hover)": (round(P_req_total_hover, 3), "W"),
        "Propeller Power req. (hover)": (round(P_req_prop_hover, 3), "W"),
        "Disk Loading (hover)": (round(sigma_hover, 3), "N/m²"),
    },

    "TIME MODEL": {
        "Time for climb": (round(t_climb / 60, 3), "min"),
        "Time for cruise": (round(t_cruise / 60, 3), "min"),
        "Time for trip": (round(t_trip / 60, 3), "min"),
    },

    "ENERGY MODEL": {
        "Energy in hover": (round(e_hover, 3), "Wh"),
        "Energy in climb": (round(e_climb, 3), "Wh"),
        "Energy in cruise": (round(e_cruise, 3), "Wh"),
        "Energy per trip": (round(e_trip, 3), "Wh"),
        "Energy in reserve": (round(e_reserve, 3), "Wh"),
        "Energy per mission": (round(e_total_required, 3), "Wh"),
        "Energy of battery designed": (round(E_battery_design, 3), "Wh"),
        "Energy not useable": (round(E_battery_design-e_total_required, 3), "Wh"),
    },

    "MASS MODEL": {
        "Rotor mass per unit": (round(m_rotor_unit, 3), "kg"),
        "Interior mass": (round(m_interior, 3), "kg"),
        "Gear mass": (round(m_gear, 3), "kg"),
        "Fuselage mass": (round(m_fuselage, 3), "kg"),
        "Motor mass": (round(m_motor, 3), "kg"),
        "Rotor mass": (round(m_rotor, 3), "kg"),
        "System mass": (round(m_system, 3), "kg"),
        "Wing mass": (round(m_wing, 3), "kg"),
        "Payload mass": (round(p.m_pay, 3), "kg"),
        "Crew mass": (round(p.m_crew, 3), "kg"),
        "Battery mass": (round(m_battery, 3), "kg"),
        "Empty mass": (round(m_empty, 3), "kg"),
        "MTOM from model": (round(mtom_model_check, 3), "kg"),
        "MTOM iterated": (round(MTOM, 3), "kg"),
        "Battery mass fraction": (round(omega_battery, 3), "-"),
        "Empty mass fraction": (round(omega_empty, 3), "-"),
    }, 

    "NOISE MODEL": {
        "Thrust coefficient horizontal": (round(C_T_hor, 3), "-"),
        "RPM in cruise": (round(rpm_cruise, 3), "RPM"),
        "SPL in cruise": (round(SPL_cruise, 3), "dB"),
        "RPM in climb": (round(rpm_climb, 3), "RPM"),
        "SPL in climb": (round(SPL_climb, 3), "dB"),
        "RPM in hover": (round(rpm_hover, 3), "RPM"),
        "SPL in hover": (round(SPL_hover, 3), "dB"),
    },

    "BATTERY LIFE MODEL": {
        "C-rate in hover": (round(c_rate_hover, 3), "1/h"),
        "C-rate in climb": (round(c_rate_climb, 3), "1/h"),
        "C-rate in cruise": (round(c_rate_cruise, 3), "1/h"),
        "Time avg. c-rate per trip": (round(c_rate_trip, 3), "1/h"),
        "Depth of discharge (DoD)": (round(DOD, 3), "-"),
        "Battery cycle life": (round(n_battery_lifecycle, 3), "-"),
        "Number of annual battery replacements": (round(n_annual_battery_required, 3), "-"),
    },

    "OPERATIONS MODEL": {
        "Turnaround time": (round(t_turnaround / 60, 3), "min"),
        "Full leg time (flight + turnaround)": (round((t_turnaround + t_trip) / 60, 3), "min"),
        "Time Efficiency Ratio": (round(te_r, 3), "-"),
        "Daily flight cycles": (round(FC_day, 3), "flights"),
        "Daily flight hours": (round(FH_day, 3), "hours"),
        "Annual flight cycles": (round(FC_annual, 3), "flights"),
        "Annual flight hours": (round(FH_annual, 3), "hours"),
    }, 

    "GWP MODEL": {
    "GWP of full battery lifecycle": (round(gwp_battery_lifecycle, 3), "kg CO2e"),
    "Ops GWP of battery annually": (round(gwp_battery_annual, 3), "kg CO2e"),
    "Ops GWP of battery per flight": (round(gwp_battery_cycle, 3), "kg CO2e"),
    "Ops GWP of energy per flight": (round(gwp_energy_cycle, 3), "kg CO2e"),
    "Ops GWP total per flight": (round(gwp_total_cycle, 3), "kg CO2e"),
    "Ops GWP total per year": (round(gwp_total_annual, 3), "kg CO2e"), 
    "GWP fraction due energy": (round(omega_gwp_energy * 100, 3), "%"),
    "GWP fraction due battery": (round(omega_gwp_battery * 100, 3), "%"),
    },

    "ECONOMIC MODEL - COST": {
        "Energy cost per trip": (round(cost_energy, 3), "€"),
        "Navigation cost per trip": (round(cost_nav, 3), "€"),
        "Crew cost per trip": (round(cost_crew, 3), "€"),
        "Wrap-rated maintenance cost per trip": (round(cost_maint_wrap, 3), "€"),
        "Battery unit cost": (round(cost_battery_unit, 3), "€"),
        "Battery maintenance cost per trip": (round(cost_maint_battery, 3), "€"),
        "Total maintenance cost per trip": (round(cost_maint_total, 3), "€"),
        "Cash operating cost per trip": (round(coc, 3), "€"),
        "Cost of ownership per trip": (round(coo, 3), "€"),
        "Direct operating cost per trip": (round(doc, 3), "€"),
        "Indirect operating cost per trip": (round(ioc, 3), "€"),
        "Total operating cost per trip": (round(toc, 3), "€"),
        "TOC per available seat per km": (round(toc_as_km, 3), "€/km"),
        "TOC per available seat per min": (round(toc_as_min, 3), "€/min"),
    },

    "ECONOMIC MODEL - REVENUE": {
        "Ticket price per pax per trip": (round(ticket_cost_for_pax, 3), "€"),
        "Revenue per trip": (round(revenue_flight, 3), "€"),
    },

    "ECONOMIC MODEL - PROFIT": {
        "Total profit per trip": (round(profit_flight, 3), "€"),
        "Total profit per year": (round(profit_annual, 3), "€"),
    },


    }

    return results, comparison_table





def write_results_to_excel(results_dict, comparison_list, mode="tests", filename=None):
    """
    Writes model results (dict) and transportation mode comparison (list of dicts)
    to one Excel file in 'src/results/<mode>/' with timestamp-based name.
    """
    import os
    from datetime import datetime
    import pandas as pd

    # 1. Zielverzeichnis prüfen
    assert mode in ["tests", "GA", "GBO", "FoM", "GWP", "Profit", "TOC"], "Invalid mode."
    results_base = os.path.join("src", "results", mode)
    if not os.path.exists(results_base):
        raise FileNotFoundError(f"The folder '{results_base}' does not exist.")

    # 2. Timestamp and path
    if filename is None:
        timestamp = datetime.now().strftime("%d%m%y_%H%M")
        filename = f"{timestamp}_results.xlsx"
    else:
        # allow user to pass a filename; ensure it ends with .xlsx
        if not filename.lower().endswith('.xlsx'):
            filename = f"{filename}.xlsx"

    output_path = os.path.join(results_base, filename)

    # If file exists and a custom filename was provided, overwrite it.
    # Using os.remove ensures ExcelWriter will create a fresh file.
    try:
        if os.path.exists(output_path):
            os.remove(output_path)
    except Exception:
        # If removal fails, we'll still attempt to write and let pandas raise if necessary
        pass

    # 3. Model Results vorbereiten (als flache Tabelle)
    rows = []
    for section, metrics in results_dict.items():
        rows.append((section, "", "", ""))  # Abschnittskopf
        for label, (value, unit) in metrics.items():
            rows.append(("", label, value, unit))
    df_results = pd.DataFrame(rows, columns=["Section", "Metric", "Value", "Unit"])

    # 4. Comparison Table
    df_comparison = pd.DataFrame(comparison_list).sort_values(by="FoM", ascending=False)


    # 5. Excel schreiben mit zwei Sheets
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_results.to_excel(writer, sheet_name="Model Results", index=False)
        df_comparison.to_excel(writer, sheet_name="Comparison Modes", index=False)

    print(f"Results written to: {output_path}")
    return df_results, df_comparison

