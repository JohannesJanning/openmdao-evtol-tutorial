import jax.numpy as jnp

def system_mass(MTOM, l_fus_m, b):
    l_ft = l_fus_m * 3.281
    b_ft = b * 3.281
    WO_lb = MTOM * 2.205

    WS_nicolai = (1.08 * WO_lb ** 0.7) / 2.205

    nz = 2.5
    WS_raymer = (0.054 * l_ft ** 1.536 * b_ft ** 0.371 * (1.5 * nz * WO_lb * 1e-4) ** 0.8) / 2.205

    return 0.5 * (WS_nicolai + WS_raymer)
