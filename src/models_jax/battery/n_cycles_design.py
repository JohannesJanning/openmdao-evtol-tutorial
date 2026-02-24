import jax.numpy as jnp

def battery_cycle_life(DOD: float, C_rate_avg: float, c_charge: float) -> float:
    a_n = -5986.8421
    b_n = 11776.3158
    c_n = 1.1
    d_n = 1.2

    return (a_n * DOD + b_n) * (1.0 / C_rate_avg ** c_n) * (0.5 / c_charge ** d_n)
