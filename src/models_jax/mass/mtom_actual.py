import jax.numpy as jnp

def compute_mtom_actual(m_empty, m_battery, m_payload):
    return m_empty + m_battery + m_payload
