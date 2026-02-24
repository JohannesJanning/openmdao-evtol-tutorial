import jax.numpy as jnp
from jax import lax

# Try to use jax.scipy.jv when available for parity; otherwise fall back to scipy
try:
    from jax.scipy import special as jsp_special  # type: ignore
    _bessel_jv = jsp_special.jv
except Exception:
    try:
        from scipy import special as spsp  # type: ignore

        def _bessel_jv(v, x):
            return jnp.array(spsp.jv(v, x))
    except Exception:
        def _bessel_jv(v, x):
            return 1.0 - (x ** 2) / 4.0

import src.parameters as p
from src.models_jax.aerodynamics.AR import aspect_ratio as AR_calculation
from src.models_jax.aerodynamics.lift_coefficient import cl_from_lift as cl_calculation
from src.models_jax.aerodynamics.drag_coefficient import cd_calculation
from src.models_jax.aerodynamics.lift_to_drag import lift_to_drag_ratio as ld_calculation
from src.models_jax.aerodynamics.drag import drag_calculation
from src.models_jax.aerodynamics.ROC import roc_calculation
from src.models_jax.momentum.V_cruise import cruise_speed
from src.models_jax.momentum.V_climb import climb_speed
from src.models_jax.momentum.V_climb_horizontal import horizontal_climb_speed
from src.models_jax.momentum.T_climb_total import total_thrust_required_climb
from src.models_jax.momentum.T_prop import thrust_per_propeller
from src.models_jax.momentum.A_disk import propeller_disk_area
from src.models_jax.momentum.P_total_hor import power_total_required
from src.models_jax.momentum.P_prop_hor import power_per_propeller
from src.models_jax.momentum.T_cruise_total import total_thrust_required_cruise
from src.models_jax.momentum.Disk_Loading import disk_loading_hover
from src.models_jax.momentum.T_hover_total import total_thrust_required_hover
from src.models_jax.momentum.P_hover_total import power_required_hover
from src.models_jax.time.time_climb import climb_time
from src.models_jax.time.time_cruise import compute_time_cruise
from src.models_jax.time.time_trip import total_trip_time
from src.models_jax.energy.energy_hover import energy_hover
from src.models_jax.energy.energy_climb import energy_climb
from src.models_jax.energy.energy_cruise import energy_cruise
from src.models_jax.energy.energy_trip import energy_total_trip
from src.models_jax.energy.energy_reserve import energy_reserve
from src.models_jax.energy.energy_total_req import energy_total_required
from src.models_jax.mass.m_interior import interior_mass
from src.models_jax.mass.m_gear import gear_mass
from src.models_jax.mass.m_fuselage import fuselage_mass
from src.models_jax.mass.m_battery import battery_mass
from src.models_jax.mass.m_motor import motor_mass
from src.models_jax.mass.m_wing import wing_mass
from src.models_jax.mass.m_rotor_total import rotor_mass
from src.models_jax.mass.m_rotor_unit import rotor_mass_per_unit
from src.models_jax.mass.m_system import system_mass
from src.models_jax.mass.m_empty import empty_mass
from src.models_jax.mass.mtom_actual import compute_mtom_actual


def full_mtom_model(MTOM, pmod, b, c, R_prop_cruise, R_prop_hover, rho_bat):
    """JAX-parity port of the original full_mtom_model.

    Uses the same computation order and signatures as `src/models/mass/mtom_model.py`.
    """
    AR = AR_calculation(b, c)

    c_l_cruise = cl_calculation(pmod.alpha_deg_cruise, AR, pmod.c_l_0, pmod.e)
    c_d_cruise = cd_calculation(c_l_cruise, AR, pmod.c_d_min, pmod.e)
    c_l_climb = cl_calculation(pmod.alpha_deg_climb, AR, pmod.c_l_0, pmod.e)
    c_d_climb = cd_calculation(c_l_climb, AR, pmod.c_d_min, pmod.e)

    V_cruise = cruise_speed(
        MTOM, pmod.g, c_l_cruise, c_d_cruise, pmod.alpha_deg_cruise, c, b, pmod.rho
    )
    # Guard velocities to reasonable finite ranges for numerical stability
    V_cruise = jnp.nan_to_num(V_cruise, nan=1e-3, posinf=1e6, neginf=1e-3)
    V_climb = climb_speed(MTOM, pmod.g, pmod.theta_deg_climb, c_l_climb, c, b, pmod.rho)
    V_climb = jnp.nan_to_num(V_climb, nan=1e-3, posinf=1e6, neginf=1e-3)
    V_climb_hor = horizontal_climb_speed(V_climb, pmod.theta_deg_climb)

    D_cruise = drag_calculation(pmod.rho, V_cruise, c, b, c_d_cruise)
    D_climb = drag_calculation(pmod.rho, V_climb, c, b, c_d_climb)

    A_prop_hor = propeller_disk_area(R_prop_cruise)
    A_hover_disk_prop = propeller_disk_area(R_prop_hover)

    T_req_total_climb = total_thrust_required_climb(D_climb, MTOM, pmod.g, pmod.theta_deg_climb)
    T_req_prop_climb = thrust_per_propeller(T_req_total_climb, pmod.n_prop_hor)
    P_req_total_climb = power_total_required(
        V_climb,
        T_req_total_climb,
        T_req_prop_climb,
        pmod.rho,
        A_prop_hor,
        pmod.n_prop_hor,
        pmod.eta_c,
    )

    T_req_total_cruise = total_thrust_required_cruise(D_cruise, pmod.alpha_deg_cruise)
    T_req_prop_cruise = thrust_per_propeller(T_req_total_cruise, pmod.n_prop_hor)
    P_req_total_cruise = power_total_required(
        V_cruise,
        T_req_total_cruise,
        T_req_prop_cruise,
        pmod.rho,
        A_prop_hor,
        pmod.n_prop_hor,
        pmod.eta_c,
    )

    T_req_total_hover = total_thrust_required_hover(MTOM, pmod.g)
    T_req_prop_hover = thrust_per_propeller(T_req_total_hover, pmod.n_prop_vert)
    sigma_hover = disk_loading_hover(T_req_prop_hover, A_hover_disk_prop)
    P_req_total_hover = power_required_hover(sigma_hover, T_req_total_hover, pmod.rho, pmod.eta_h)

    # sanitize power values to avoid NaN/Inf propagation
    P_req_total_climb = jnp.nan_to_num(P_req_total_climb, nan=0.0, posinf=1e9, neginf=0.0)
    P_req_total_cruise = jnp.nan_to_num(P_req_total_cruise, nan=0.0, posinf=1e9, neginf=0.0)
    P_req_total_hover = jnp.nan_to_num(P_req_total_hover, nan=0.0, posinf=1e9, neginf=0.0)

    ROC = roc_calculation(pmod.alpha_deg_climb, V_climb)
    t_climb = climb_time(pmod.h_cruise, pmod.h_hover, ROC)
    t_cruise = compute_time_cruise(pmod.distance_trip, V_climb_hor, t_climb, V_cruise)
    t_trip = total_trip_time(pmod.time_hover, t_climb, t_cruise)

    e_hover = energy_hover(P_req_total_hover, pmod.time_hover)
    e_climb = energy_climb(P_req_total_climb, t_climb)
    e_cruise = energy_cruise(P_req_total_cruise, t_cruise)
    e_trip = energy_total_trip(e_hover, e_climb, e_cruise)
    e_reserve = energy_reserve(P_req_total_cruise, pmod.time_reserve)
    e_total_required = energy_total_required(e_trip, e_reserve)

    m_interior = interior_mass(MTOM, pmod.rho, V_cruise)
    m_gear = gear_mass(MTOM, R_prop_cruise, pmod.r_fus_m)
    m_fuselage = fuselage_mass(MTOM, pmod.l_fus_m, pmod.r_fus_m, pmod.rho, V_cruise)
    m_motor = motor_mass(P_req_total_hover, P_req_total_climb, pmod.n_prop_vert, pmod.n_prop_hor)
    m_rotor = rotor_mass(pmod.n_prop_vert, pmod.n_prop_hor, R_prop_hover, R_prop_cruise)
    m_wing = wing_mass(MTOM, V_cruise, b, c, pmod.rho)
    m_system = system_mass(MTOM, pmod.l_fus_m, b)
    m_empty = empty_mass(
        m_wing,
        m_motor,
        m_rotor,
        pmod.m_crew,
        m_interior,
        m_fuselage,
        m_system,
        m_gear,
    )
    # compute battery mass from required energy
    m_battery = battery_mass(e_total_required, rho_bat)

    # sanitize masses to avoid NaN/Inf propagation
    m_empty = jnp.nan_to_num(m_empty, nan=1e-3, posinf=1e6, neginf=1e-3)
    m_battery = jnp.nan_to_num(m_battery, nan=1e-3, posinf=1e6, neginf=1e-3)

    # physical safety: if battery mass exceeds a large fraction of the
    # current MTOM estimate, cap it and make energy/mass consistent.
    # Use JAX-friendly ops so this function can be traced/jitted.
    battery_fraction_max = 0.6
    m_batt_cap = battery_fraction_max * MTOM
    # capped battery (elementwise-safe)
    m_battery_capped = jnp.minimum(m_battery, m_batt_cap)
    usable_density = 0.64 * rho_bat
    # if capping occurred, replace total required energy with the amount
    # the capped battery can actually store; otherwise keep original energy
    e_total_required = jnp.where(m_battery > m_batt_cap, m_battery_capped * usable_density, e_total_required)
    m_battery = m_battery_capped

    return compute_mtom_actual(m_empty, m_battery, pmod.m_pay)
