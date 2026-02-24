import jax.numpy as jnp

def disk_loading_hover(T_total_hover, A_disk_total):
    """Disk loading with a small-area floor for numerical robustness.

    If the provided disk area is unrealistically small the resulting disk
    loading can blow up and drive induced-velocity (and therefore power)
    to extreme values that destabilize the MTOM fixed-point. We apply a
    conservative minimum area floor to avoid that while preserving
    physical behavior for reasonable rotor sizes.
    """
    # conservative minimum disk area (m^2) — protects against tiny radii
    A_MIN = 0.5
    from src.models_jax.utils.smoothing import soft_floor

    A_safe = soft_floor(A_disk_total, A_MIN, k=80.0)
    return T_total_hover / A_safe
