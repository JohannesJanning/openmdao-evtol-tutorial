
import jax.numpy as jnp
from jax import lax

from src.models_jax.mass.mtom_model import full_mtom_model
from src.models_jax.mass.m_battery import battery_mass
from src.models_jax.mass.m_empty import empty_mass
from src.models_jax.mass.m_gear import gear_mass
from src.models_jax.mass.m_fuselage import fuselage_mass
from src.models_jax.mass.m_wing import wing_mass
from src.models_jax.mass.m_motor import motor_mass
from src.models_jax.mass.m_rotor_total import rotor_mass
import src.parameters as pmod


def mtom_iteration_loop(MTOM_initial, b, c, R_prop_cruise, R_prop_hover, rho_bat, params, tol=1e-3, max_iter=500, verbose=True):
    MTOM = float(MTOM_initial)
    alpha = 0.2  # relaxation factor to improve stability
    MTOM_min = 1000.0
    MTOM_max = 1e6

    # Helper: residual function f(M) = mtom_actual(M) - M
    def residual(M):
        val = full_mtom_model(M, params, b, c, R_prop_cruise, R_prop_hover, rho_bat)
        try:
            return float(val) - float(M)
        except Exception as e:
            raise RuntimeError(f"residual evaluation failed at M={M}: {e}")

    # Try a robust bisection root-find first (deterministic, monotonic behavior expected physically)
    try:
        a = MTOM_min
        bb = MTOM_max
        fa = residual(a)
        fb = residual(bb)
        if fa == 0:
            return a
        if fb == 0:
            return bb

        bracket_ok = False
        # if signs differ, we have a bracket
        if fa * fb < 0:
            bracket_ok = True
        else:
            # attempt to build a bracket around the initial guess
            lo = max(MTOM_min, MTOM * 0.5)
            hi = min(MTOM_max, MTOM * 1.5)
            try:
                flo = residual(lo)
                fhi = residual(hi)
                if flo * fhi < 0:
                    a, bb, fa, fb = lo, hi, flo, fhi
                    bracket_ok = True
            except Exception:
                bracket_ok = False

        if bracket_ok:
            if verbose:
                print(f"Bisection: found bracket a={a}, b={bb}")
            left, right, fleft, fright = a, bb, fa, fb
            for i in range(60):
                mid = 0.5 * (left + right)
                fmid = residual(mid)
                if verbose:
                    print(f"Bisection iter {i+1}: left={left:.3f}, mid={mid:.3f}, right={right:.3f}, fmid={fmid:.6g}")
                if abs(fmid) < tol:
                    return mid
                # narrow bracket
                if fleft * fmid <= 0:
                    right, fright = mid, fmid
                else:
                    left, fleft = mid, fmid
                if abs(right - left) < max(1e-6, tol):
                    return 0.5 * (left + right)
    except Exception:
        # fall through to relaxed iteration on any failure
        if verbose:
            print("Bisection failed or unstable; falling back to relaxed fixed-point iteration")
    
        # If bisection couldn't find a bracket, try a damped secant method (often converges faster)
        try:
            x0 = max(MTOM_min, MTOM * 0.8)
            x1 = min(MTOM_max, MTOM * 1.2)
            f0 = residual(x0)
            f1 = residual(x1)
            if verbose:
                print(f"Secant: starting x0={x0:.3f}, x1={x1:.3f}, f0={f0:.6g}, f1={f1:.6g}")
            for si in range(60):
                denom = (f1 - f0)
                if abs(denom) < 1e-12:
                    break
                x_new = x1 - f1 * (x1 - x0) / denom
                # damp the secant step to improve stability
                x_new = x1 + 0.2 * (x_new - x1)
                x_new = max(MTOM_min, min(MTOM_max, x_new))
                f_new = residual(x_new)
                if verbose:
                    print(f"Secant iter {si+1}: x_new={x_new:.6f}, f_new={f_new:.6g}")
                if abs(f_new) < tol:
                    return x_new
                # shift
                x0, f0 = x1, f1
                x1, f1 = x_new, f_new
            if verbose:
                print("Secant did not converge quickly; falling back to relaxed iteration")
        except Exception:
            if verbose:
                print("Secant method failed; falling back to relaxed iteration")

    # Fallback: original relaxed fixed-point loop with battery clamp guard
    for i in range(max_iter):
        try:
            mtom_actual = full_mtom_model(MTOM, params, b, c, R_prop_cruise, R_prop_hover, rho_bat)
        except Exception as e:
            if verbose:
                print(f"Iteration {i+1}: full_mtom_model raised {type(e).__name__}: {e}")
            raise

        # Ensure numeric safety
        try:
            mtom_actual_val = float(mtom_actual)
        except Exception:
            if verbose:
                print(f"Iteration {i+1}: mtom_actual is not a finite real: {mtom_actual}")
            raise RuntimeError("mtom_actual is not a finite real number")

        # Additional safety: recompute battery_mass and clamp if it dominates MTOM
        try:
            # recompute major mass terms to estimate battery contribution
            m_gear = float(gear_mass(MTOM, R_prop_cruise, pmod.r_fus_m))
            # use simplified estimates for components (some functions depend on power; we approximate)
            m_wing = float(wing_mass(MTOM, 0.0, b, c, pmod.rho))
            m_motor = float(motor_mass(0.0, 0.0, pmod.n_prop_vert, pmod.n_prop_hor))
            m_rotor = float(rotor_mass(pmod.n_prop_vert, pmod.n_prop_hor, R_prop_hover, R_prop_cruise))
            m_fuselage = float(fuselage_mass(MTOM, pmod.l_fus_m, pmod.r_fus_m, pmod.rho, 0.0))
            m_system = float(pmod.m_crew)  # conservative placeholder
            m_interior = float(0.0)
            m_empty_est = float(empty_mass(m_wing, m_motor, m_rotor, pmod.m_crew, m_interior, m_fuselage, m_system, m_gear))

            # Extract battery estimate as mtom_actual - m_empty_est - payload
            m_battery_est = mtom_actual_val - m_empty_est - float(pmod.m_pay)
            if not (m_battery_est is None) and (m_battery_est > 0):
                # enforce a cap: battery cannot exceed a fraction of MTOM
                battery_fraction_max = 0.6
                battery_cap = battery_fraction_max * MTOM
                if m_battery_est > battery_cap:
                    if verbose:
                        print(f"Iteration {i+1}: battery_mass {m_battery_est:.1f} > cap {battery_cap:.1f}, clamping")
                    m_battery_est = battery_cap
                    # recompute mtom_actual_val from clamped battery + empty + payload
                    mtom_actual_val = m_empty_est + m_battery_est + float(pmod.m_pay)
        except Exception:
            # if guard computation fails, continue with original mtom_actual_val
            pass

        if verbose:
            print(f"Iteration {i+1}: MTOM = {MTOM:.2f}, MTOM_actual = {mtom_actual_val:.2f}, Δ = {abs(MTOM - mtom_actual_val):.4f}")

        if abs(MTOM - mtom_actual_val) < tol:
            MTOM = mtom_actual_val
            break

        # Relaxed update to avoid divergence
        MTOM = MTOM + alpha * (mtom_actual_val - MTOM)

        # Clamp to sensible bounds
        MTOM = max(MTOM_min, min(MTOM, MTOM_max))

    return MTOM


def mtom_iteration_jax(MTOM_initial, b, c, R_prop_cruise, R_prop_hover, rho_bat, params, tol=1e-3, max_iter=500):
    """JAX-native relaxed fixed-point iteration using lax.while_loop.

    This implementation keeps the same relaxed-update behavior but is
    written with JAX primitives so it can be jitted and used in AD flows.
    It relies on `full_mtom_model` to include the parity-preserving
    battery-cap guard so the loop itself remains simple.
    """
    alpha = 0.2

    # Use a fixed-count JAX `lax.scan` for jittable/reservable iteration.
    # Use float64 for the iteration carry to match the dtypes used by
    # surrounding model computations and avoid lax.scan dtype mismatches.
    MTOM0 = jnp.asarray(MTOM_initial, dtype=jnp.float64)

    def scan_body(carry, _):
        mtom = carry
        mtom_actual = full_mtom_model(mtom, params, b, c, R_prop_cruise, R_prop_hover, rho_bat)
        mtom_candidate = mtom + alpha * (mtom_actual - mtom)
        diff = jnp.abs(mtom_candidate - mtom)
        # decide whether to accept the update (if not converged yet)
        accept = diff > tol
        mtom_next = jnp.where(accept, mtom_candidate, mtom)
        return mtom_next, None

    final_mtom, _ = lax.scan(scan_body, MTOM0, None, length=max_iter)
    return final_mtom
