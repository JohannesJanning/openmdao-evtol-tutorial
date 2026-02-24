import jax.numpy as jnp
from src.models_jax.utils.smoothing import soft_floor


def gear_mass(MTOM, R_prop_cruise, r_fus_m):
    # Apply a smooth minimum prop radius to keep geometry positive
    R_prop_cruise = soft_floor(R_prop_cruise, 0.7, k=80.0)

    kg_to_lb = 2.205
    m_to_inch = 39.37
    n_safety = 1.5
    n_certification = 2.5
    n_l = n_safety * n_certification

    W_L = MTOM * kg_to_lb

    prop_clearance_min_m = 0.1778
    prop_clearance_min_in = prop_clearance_min_m * m_to_inch
    R_cruise_in = R_prop_cruise * m_to_inch
    r_fuselage_in = r_fus_m * m_to_inch
    l_main_in = R_cruise_in - r_fuselage_in + prop_clearance_min_in
    l_nose_in = l_main_in

    M_mg_raymer_lb = 0.095 * (n_l * W_L) ** 0.768 * (l_main_in / 12) ** 0.409
    M_mg_raymer = M_mg_raymer_lb / kg_to_lb

    M_ng_raymer_lb = 0.125 * (n_l * W_L) ** 0.566 * (l_nose_in / 12) ** 0.845
    M_ng_raymer = M_ng_raymer_lb / kg_to_lb

    M_tg_nicolai_lb = 0.054 * (n_l * W_L) ** 0.684 * (l_main_in / 12) ** 0.601
    M_tg_nicolai = M_tg_nicolai_lb / kg_to_lb

    m_gear = 0.5 * (M_mg_raymer + M_ng_raymer + M_tg_nicolai)
    return m_gear
