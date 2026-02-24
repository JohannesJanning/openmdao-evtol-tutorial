import jax.numpy as jnp

def cl_calculation(alpha_deg, AR, c_l_0, e):
    """
    Calculate the 3D lift coefficient of a finite wing using lift-curve slope correction.

    Parameters:
        alpha_deg (float): Angle of attack in degrees.
        AR        (float): Wing aspect ratio (span/chord).
        c_l_0     (float): Zero-lift coefficient of the airfoil.
        e         (float): Oswald efficiency factor.

    Returns:
        float: Total lift coefficient (dimensionless).
    """
    alpha_rad = alpha_deg * (jnp.pi / 180.0)
    a_airfoil = 5.747
    a_wing = a_airfoil / (1 + (a_airfoil / (jnp.pi * AR * e)))
    return a_wing * alpha_rad + c_l_0

# Backwards-compatible alias: some modules import `cl_from_lift`
def cl_from_lift(alpha_deg, AR, c_l_0, e):
    return cl_calculation(alpha_deg, AR, c_l_0, e)
