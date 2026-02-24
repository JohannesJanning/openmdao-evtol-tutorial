import jax.numpy as jnp

# Bessel function may be used in the original model (scipy.special.jv).
# Prefer `jax.scipy.special.jv` for JIT-compatibility; if not available,
# fall back to a lightweight Taylor-like approximation that is JAX-native.
try:
    from jax.scipy import special as jsp_special  # type: ignore
    _bessel_jv = jsp_special.jv
except Exception:
    def _bessel_jv(v, x):
        # small-argument approximation: J_v(x) ≈ 1 - x^2/4 for v=0-like behaviour
        return 1.0 - (x ** 2) / 4.0


def tonal_noise_hover(T_req_prop_hover, Power_req_hover, R_prop_hover, rho, n_prop_vert, n_blade_vert, params):
    """
    Compute tonal Sound Pressure Level (SPL) for hovering rotors using
    the Gutin–Deming model. Logic mirrors the original implementation
    in `src/models/noise/SPL_hover.py` but uses `jax.numpy` and the
    `_bessel_jv` helper so it is JAX-friendly when `jax.scipy` is
    available.
    """
    q_GTM = 1
    C_T = params.C_T_hover
    r_obs_ft = params.r_obs_ft if hasattr(params, 'r_obs_ft') else 250
    theta = jnp.pi / 2 + jnp.arcsin(100.0 / r_obs_ft)
    r_obs = r_obs_ft / 3.28084
    OAT_K = 15.0 + 273.15
    R_sos = 287.5
    gamma = 1.4
    sos = jnp.sqrt(gamma * R_sos * OAT_K)

    n_rot_s = jnp.sqrt(T_req_prop_hover / (C_T * rho * (2.0 * R_prop_hover) ** 4))
    n_rot_rpm = n_rot_s * 60.0
    n_rot_rad = 2.0 * jnp.pi * n_rot_rpm / 60.0
    Q = Power_req_hover / n_prop_vert / n_rot_rad

    R_e = 0.8 * R_prop_hover
    k = q_GTM * n_blade_vert * n_rot_rad / sos
    J_qn = _bessel_jv(q_GTM * n_blade_vert, k * R_e * jnp.sin(theta))

    prefactor = (q_GTM * n_blade_vert * n_rot_rad) / (2.0 * jnp.sqrt(2.0) * jnp.pi * sos * r_obs)
    bracket_term = -T_req_prop_hover * jnp.cos(theta) + Q * (sos / (n_rot_rad * R_e ** 2))
    p_rms = prefactor * jnp.abs(bracket_term) * J_qn

    # sanitize p_rms to avoid log-of-zero/NaN/Inf
    p_rms_safe = jnp.nan_to_num(p_rms, nan=1e-20, posinf=1e20, neginf=1e-20)
    p_ref = 20e-6
    safe_p_rms = jnp.maximum(p_rms_safe, 1e-20)
    SPL_prop = 20.0 * jnp.log10(safe_p_rms / p_ref)
    SPL_total = SPL_prop + 10.0 * jnp.log10(n_prop_vert)
    return SPL_total
