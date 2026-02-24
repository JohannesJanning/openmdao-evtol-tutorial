import math

from src.models.mass.mtom_model import full_mtom_model

def mtom_iteration_loop(MTOM_initial, b, c,R_prop_cruise, R_prop_hover, rho_bat, params, tol=1e-7, max_iter=500, verbose=True):
    """
    Fixed-point iteration to converge MTOM based on internal mass and energy models.

    Parameters:
        MTOM_initial (float): Initial MTOM guess [kg]
        params (module): Parameter module (e.g. `p`)
        tol (float): Convergence tolerance [kg]
        max_iter (int): Maximum number of iterations
        verbose (bool): Print intermediate values if True

    Returns:
        float: Converged MTOM value [kg]
    """
    MTOM = MTOM_initial

    for i in range(max_iter):
        mtom_actual = full_mtom_model(MTOM, params, b, c, R_prop_cruise, R_prop_hover, rho_bat)

        if verbose:
            print(f"Iteration {i+1}: MTOM = {MTOM:.2f}, MTOM_actual = {mtom_actual:.2f}, Δ = {abs(MTOM - mtom_actual):.4f}")

        if abs(MTOM - mtom_actual) < tol:
            break

        MTOM = 0.5 * (MTOM + mtom_actual)  # Relaxed update for stability

    return MTOM
