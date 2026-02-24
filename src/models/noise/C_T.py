import numpy as np

def compute_ct(thrust_N, rho, V, R_prop):
    A = np.pi * R_prop**2
    return thrust_N / (0.5 * rho * V**2 * A)

