import jax.numpy as jnp

def empty_mass(m_wing, m_motor, m_rotor, m_crew, m_furnish, m_fuselage, m_system, m_gear):
    """Calculate empty mass (JAX version)."""
    return m_wing + m_motor + m_rotor + m_crew + m_furnish + m_fuselage + m_system + m_gear
