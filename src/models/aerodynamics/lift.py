def lift_calculation(rho: float, v: float, cl: float, c: float, b: float) -> float:
    """
    Calculate aerodynamic lift.

    Parameters:
    rho (float): Air density (kg/m³)
    v   (float): Flight speed (m/s)
    cl  (float): Lift coefficient (dimensionless)
    c   (float): Wing chord length (m)
    b   (float): Wing span (m)

    Returns:
    float: Lift force (N)
    """
    S = c * b  # wing area in m²
    return 0.5 * rho * v**2 * cl * S
