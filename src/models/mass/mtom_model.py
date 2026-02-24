import math
import numpy as np
from src.models.transportation.transportation_modes import transportation_mode_comparison
from scipy.special import jv

import src.parameters as p
from src.models.aerodynamics.AR import AR_calculation
from src.models.aerodynamics.lift_coefficient import cl_calculation
from src.models.aerodynamics.drag_coefficient import cd_calculation
from src.models.aerodynamics.lift_to_drag import ld_calculation
from src.models.aerodynamics.drag import drag_calculation
from src.models.aerodynamics.ROC import roc_calculation
from src.models.momentum.V_cruise import cruise_speed
from src.models.momentum.V_climb import climb_speed
from src.models.momentum.V_climb_horizontal import horizontal_climb_speed
from src.models.momentum.T_climb_total import total_thrust_required_climb
from src.models.momentum.T_prop import thrust_per_propeller
from src.models.momentum.A_disk import propeller_disk_area
from src.models.momentum.P_total_hor import power_total_required
from src.models.momentum.P_prop_hor import power_per_propeller
from src.models.momentum.T_cruise_total import total_thrust_required_cruise
from src.models.momentum.Disk_Loading import disk_loading_hover
from src.models.momentum.T_hover_total import total_thrust_required_hover
from src.models.momentum.P_hover_total import power_required_hover
from src.models.time.time_climb import climb_time
from src.models.time.time_cruise import compute_time_cruise
from src.models.time.time_trip import total_trip_time
from src.models.energy.energy_hover import energy_hover
from src.models.energy.energy_climb import energy_climb
from src.models.energy.energy_cruise import energy_cruise
from src.models.energy.energy_trip import energy_total_trip
from src.models.energy.energy_reserve import energy_reserve
from src.models.energy.energy_total_req import energy_total_required
from src.models.mass.m_interior import interior_mass
from src.models.mass.m_gear import gear_mass
from src.models.mass.m_fuselage import fuselage_mass
from src.models.mass.m_battery import battery_mass
from src.models.mass.m_motor import motor_mass
from src.models.mass.m_wing import wing_mass
from src.models.mass.m_rotor_total import rotor_mass 
from src.models.mass.m_rotor_unit import rotor_mass_per_unit
from src.models.mass.m_system import system_mass
from src.models.mass.m_empty import empty_mass
from src.models.mass.mtom_actual import compute_mtom_actual

def full_mtom_model(MTOM, p, b, c, R_prop_cruise, R_prop_hover, rho_bat):
    # Geometry
    AR = AR_calculation(b, c)

    # Lift and Drag Coefficients
    c_l_cruise = cl_calculation(p.alpha_deg_cruise, AR, p.c_l_0, p.e)
    c_d_cruise = cd_calculation(c_l_cruise, AR, p.c_d_min, p.e)
    c_l_climb = cl_calculation(p.alpha_deg_climb, AR, p.c_l_0, p.e)
    c_d_climb = cd_calculation(c_l_climb, AR, p.c_d_min, p.e)

    # Speeds
    V_cruise = cruise_speed(MTOM, p.g, c_l_cruise, c_d_cruise, p.alpha_deg_cruise, c, b, p.rho)
    V_climb = climb_speed(MTOM, p.g, p.theta_deg_climb, c_l_climb, c, b, p.rho)
    V_climb_hor = horizontal_climb_speed(V_climb, p.theta_deg_climb)

    # Drag Forces
    D_cruise = drag_calculation(p.rho, V_cruise, c, b, c_d_cruise)
    D_climb = drag_calculation(p.rho, V_climb, c, b, c_d_climb)

    # Propeller Areas
    A_prop_hor = propeller_disk_area(R_prop_cruise)
    A_hover_disk_prop = propeller_disk_area(R_prop_hover)

    # Climb Power
    T_req_total_climb = total_thrust_required_climb(D_climb, MTOM, p.g, p.theta_deg_climb)
    T_req_prop_climb = thrust_per_propeller(T_req_total_climb, p.n_prop_hor)
    P_req_total_climb = power_total_required(V_climb, T_req_total_climb, T_req_prop_climb, p.rho, A_prop_hor, p.n_prop_hor, p.eta_c)

    # Cruise Power
    T_req_total_cruise = total_thrust_required_cruise(D_cruise, p.alpha_deg_cruise)
    T_req_prop_cruise = thrust_per_propeller(T_req_total_cruise, p.n_prop_hor)
    P_req_total_cruise = power_total_required(V_cruise, T_req_total_cruise, T_req_prop_cruise, p.rho, A_prop_hor, p.n_prop_hor, p.eta_c)

    # Hover Power
    T_req_total_hover = total_thrust_required_hover(MTOM, p.g)
    T_req_prop_hover = thrust_per_propeller(T_req_total_hover, p.n_prop_vert)
    sigma_hover = disk_loading_hover(T_req_prop_hover, A_hover_disk_prop)
    P_req_total_hover = power_required_hover(sigma_hover, T_req_total_hover, p.rho, p.eta_h)

    # Time Model
    ROC = roc_calculation(p.alpha_deg_climb, V_climb)
    t_climb = climb_time(p.h_cruise, p.h_hover, ROC)
    t_cruise = compute_time_cruise(p.distance_trip, V_climb_hor, t_climb, V_cruise)
    t_trip = total_trip_time(p.time_hover, t_climb, t_cruise)

    # Energy
    e_hover = energy_hover(P_req_total_hover, p.time_hover)
    e_climb = energy_climb(P_req_total_climb, t_climb)
    e_cruise = energy_cruise(P_req_total_cruise, t_cruise)
    e_trip = energy_total_trip(e_hover, e_climb, e_cruise)
    e_reserve = energy_reserve(P_req_total_cruise, p.time_reserve)
    e_total_required = energy_total_required(e_trip, e_reserve)

    # Masses
    m_interior = interior_mass(MTOM, p.rho, V_cruise)
    m_gear = gear_mass(MTOM, R_prop_cruise, p.r_fus_m)
    m_fuselage = fuselage_mass(MTOM, p.l_fus_m, p.r_fus_m, p.rho, V_cruise)
    m_motor = motor_mass(P_req_total_hover, P_req_total_climb, p.n_prop_vert, p.n_prop_hor)
    m_rotor = rotor_mass(p.n_prop_vert, p.n_prop_hor, R_prop_hover, R_prop_cruise)
    m_wing = wing_mass(MTOM, V_cruise, b, c, p.rho)
    m_system = system_mass(MTOM, p.l_fus_m, b)
    m_empty = empty_mass(m_wing, m_motor, m_rotor, p.m_crew, m_interior, m_fuselage, m_system, m_gear)
    m_battery = battery_mass(e_total_required, rho_bat)

    # MTOM estimate
    return compute_mtom_actual(m_empty, m_battery, p.m_pay)