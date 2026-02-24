"""
Parameter configuration for eVTOL conceptual design model.

"""

# =============================================================================
# INITIAL VALUES
# =============================================================================
MTOM_initial = 5000  # maximum take-off mass [kg]

# =============================================================================
# OPERATIONS PARAMETERS
# =============================================================================
h_hover = 15.24                  # Hover altitude above ground [m] (100 ft = 30.48 m)
h_hover_ft = h_hover * 3.28084   # Hover altitude [ft]
h_cruise = 1219.2                # Cruise altitude above ground [m] (4000 ft)
h_cruise_ft = h_cruise * 3.28084 # Cruise altitude [ft]
r_obs_ft = 250                   # Noise observer distance [ft]
distance_trip_km = 70            # Total design trip distance [km]
distance_trip = distance_trip_km * 1000  # Trip distance [m]
time_hover = 60                  # Hover duration (takeoff + landing) [s]
time_reserve = 20 * 60           # Reserve flight time (e.g. 20 min) [s]
# c_charge = 1                   # Charging C-rate [-], scenario-dependent
min_per_s = 1 / 60               # Conversion: minutes per second
km_per_m = 1 / 1000              # Conversion: kilometers per meter

# =============================================================================
# ECONOMIC OPERATIONS PARAMETERS
# =============================================================================
N_wd = 260         # Working days per year [-], based on Uber model
T_D = 8 * 60 * 60  # Daily operating time window [s] (8 hours)

# =============================================================================
# PHYSICAL CONSTANTS & EFFICIENCIES
# =============================================================================
g = 9.81                 # Gravitational acceleration [m/s²]
eta_e = 0.9              # Electrical efficiency [-]
eta_p = 0.85             # Propulsive efficiency (cruise) [-]
eta_hp = 0.7             # Hover propulsive efficiency [-]
eta_h = eta_hp * eta_e   # Total hover efficiency [-]
eta_c = eta_p * eta_e    # Total cruise efficiency [-]

# =============================================================================
# AERODYNAMIC PARAMETERS
# =============================================================================
rho = 1.225                # Air density at sea level [kg/m³]
alpha_deg_cruise = 3       # Angle of attack in cruise [deg]
alpha_deg_climb = 8        # Angle of attack in climb [deg]
alpha_deg_max = 15         # Maximum AoA before stall [deg]
e = 0.8                    # Oswald efficiency factor [-]
c_l_0 = 0.2834             # Lift coefficient at zero AoA [-], from regression (NACA2412)
c_d_min = 0.0397           # Minimum drag coefficient [-]
theta_deg_climb = alpha_deg_climb  # Climb angle, assumed equal to climb AoA [deg]
C_T = 0.1                  # Propeller thrust coefficient [-], rough estimate
C_T_hover = 0.1            # Thrust coefficient of hovering rotor [-]

# =============================================================================
# AIRCRAFT GEOMETRY & FIXED PARAMETERS
# =============================================================================
n_prop_hor = 1          # Number of cruise (horizontal) propellers [-]
n_prop_vert = 8         # Number of hover (vertical) propellers [-]
n_bladed_hor = 2        # Number of blades per cruise propeller [-]
n_blade_vert = 2        # Number of blades per hover propeller [-]
l_fus_m = 6             # Fuselage length [m]
r_fus_m = 0.75          # Fuselage radius [m] (→ diameter = 1.5 m)
# rho_bat = 400          # Battery energy density [Wh/kg]
m_pay = 392.8           # Payload mass (4 pax + luggage) [kg]
m_crew = 82.5 + 14      # Crew mass (pilot + equipment) [kg]
d_rotors_space = 0.00125  # Minimum horizontal distance between hover rotors [m]

# =============================================================================
# ECONOMIC PARAMETERS
# =============================================================================
GWP_battery = 124.5    # GWP of battery manufacturing [kg CO2e/kWh]
GWP_energy = 0.37896   # GWP of electricity generation [kg CO2e/kWh]
P_bat_s = 115          # Battery replacement cost [€/kWh]
P_e = 0.096668         # Electricity price [€/kWh]
P_s_empty = 1436.5     # Aircraft acquisition cost [€/kg empty mass]
U_pilot = 2000 * 60 * 60  # Annual pilot utilization time [s]
S_P = 45300            # Pilot salary per year [€]
N_AC = 1               # Number of aircraft per pilot [-]
fare_km = 1.98         # Passenger fare per km [€/km]
N_s = 4                # Number of paying seats [-]
LF = 0.68              # Average load factor [-], from Uber data
unitrate = 80.14       # Unit rate [€], DFS 2024
pm = 0.05              # Profit margin [-]


def params_as_tuple():
    """Return model parameters as a deterministic tuple for JAX static args.

    Order is stable and should be used when passing parameters into jitted
    JAX functions as a static argument (or converted into a PyTree/array).
    """
    return (
        MTOM_initial,
        h_hover,
        h_hover_ft,
        h_cruise,
        h_cruise_ft,
        r_obs_ft,
        distance_trip_km,
        distance_trip,
        time_hover,
        time_reserve,
        min_per_s,
        km_per_m,
        N_wd,
        T_D,
        g,
        eta_e,
        eta_p,
        eta_hp,
        eta_h,
        eta_c,
        rho,
        alpha_deg_cruise,
        alpha_deg_climb,
        alpha_deg_max,
        e,
        c_l_0,
        c_d_min,
        theta_deg_climb,
        C_T,
        C_T_hover,
        n_prop_hor,
        n_prop_vert,
        n_bladed_hor,
        n_blade_vert,
        l_fus_m,
        r_fus_m,
        m_pay,
        m_crew,
        d_rotors_space,
        GWP_battery,
        GWP_energy,
        P_bat_s,
        P_e,
        P_s_empty,
        U_pilot,
        S_P,
        N_AC,
        fare_km,
        N_s,
        LF,
        unitrate,
        pm,
    )
