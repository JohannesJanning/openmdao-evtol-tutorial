import openmdao.api as om
import numpy as np

from src.models_jax.mass.m_battery import battery_mass
from src.models_jax.mass.m_empty import empty_mass
from src.models_jax.mass.m_system import system_mass
from src.models_jax.mass.m_wing import wing_mass
from src.models_jax.mass.m_motor import motor_mass
from src.models_jax.mass.m_rotor_total import rotor_mass
from src.models_jax.mass.m_fuselage import fuselage_mass
from src.models_jax.mass.m_gear import gear_mass
from src.models_jax.mass.m_interior import interior_mass
import jax
import jax.numpy as jnp


class MassComp(om.ExplicitComponent):
    """Compute component masses: battery, empty, system and return MTOM estimate.

    Inputs: E_total_req, MTOM_guess
    Outputs: m_battery, m_empty, m_system, MTOM_est
    """

    def initialize(self):
        self.options.declare('parameters')

    def setup(self):
        self.add_input('E_total_req', val=0.0)
        self.add_input('MTOM', val=1500.0)
        self.add_input('rho_bat', val=300.0)
        self.add_input('b', val=10.0)
        self.add_input('c', val=1.5)
        self.add_input('V_cruise', val=30.0)
        self.add_input('r_cruise', val=1.5)
        self.add_input('r_hover', val=0.5)
        self.add_input('P_req_total_hover', val=0.0)
        self.add_input('P_req_total_climb', val=0.0)
        self.add_input('P_req_total_cruise', val=0.0)

        self.add_output('m_battery', val=0.0)
        self.add_output('m_empty', val=0.0)
        self.add_output('m_system', val=0.0)
        self.add_output('MTOM_est', val=1500.0)

        # provide analytic JAX partials
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        p = self.options['parameters']
        E_total = inputs['E_total_req'][0]
        MTOM_guess = inputs['MTOM'][0]
        rho_bat = inputs['rho_bat'][0]
        b = inputs['b'][0]
        c = inputs['c'][0]
        V_cruise = inputs['V_cruise'][0]
        r_cr = inputs['r_cruise'][0]
        r_hv = inputs['r_hover'][0]

        P_hover = inputs['P_req_total_hover'][0]
        P_climb = inputs['P_req_total_climb'][0]
        P_cruise = inputs['P_req_total_cruise'][0]

        # battery mass uses energy requirement and battery energy density
        m_batt = float(np.asarray(battery_mass(E_total, rho_bat)))

        # detailed empty mass assembly from subcomponents
        m_wing = float(np.asarray(wing_mass(MTOM_guess, V_cruise, b, c, p.rho)))
        m_motor = float(np.asarray(motor_mass(P_hover, P_climb, p.n_prop_vert, p.n_prop_hor)))
        m_rotor = float(np.asarray(rotor_mass(p.n_prop_vert, p.n_prop_hor, r_hv, r_cr)))
        m_fuselage = float(np.asarray(fuselage_mass(MTOM_guess, p.l_fus_m, p.r_fus_m, p.rho, V_cruise)))
        m_gear = float(np.asarray(gear_mass(MTOM_guess, r_cr, p.r_fus_m)))
        m_interior = float(np.asarray(interior_mass(MTOM_guess, p.rho, V_cruise)))

        m_system_val = float(np.asarray(system_mass(MTOM_guess, p.l_fus_m, b)))

        m_empty_val = float(np.asarray(empty_mass(m_wing, m_motor, m_rotor, p.m_crew, m_interior, m_fuselage, m_system_val, m_gear)))

        # `m_empty_val` already includes crew mass; follow evaluation_model
        # where MTOM = m_empty + m_battery + m_pay (payload only)
        MTOM_est = m_empty_val + m_batt + p.m_pay

        outputs['m_battery'] = m_batt
        outputs['m_empty'] = m_empty_val
        outputs['m_system'] = m_system_val
        outputs['MTOM_est'] = MTOM_est

    def compute_partials(self, inputs, partials):
        p = self.options['parameters']
        in_names = ['E_total_req', 'MTOM', 'rho_bat', 'b', 'c', 'V_cruise', 'r_cruise', 'r_hover', 'P_req_total_hover', 'P_req_total_climb', 'P_req_total_cruise']
        out_names = ['m_battery', 'm_empty', 'm_system', 'MTOM_est']
        x = jnp.array([inputs[n][0] for n in in_names])

        def fun(x):
            E_total, MTOM_guess, rho_bat, b, c, V_cruise, r_cr, r_hv, P_hover, P_climb, P_cruise = x
            p = self.options['parameters']
            m_batt = battery_mass(E_total, rho_bat)
            m_wing = wing_mass(MTOM_guess, V_cruise, b, c, p.rho)
            m_motor = motor_mass(P_hover, P_climb, p.n_prop_vert, p.n_prop_hor)
            m_rotor = rotor_mass(p.n_prop_vert, p.n_prop_hor, r_hv, r_cr)
            m_fuselage = fuselage_mass(MTOM_guess, p.l_fus_m, p.r_fus_m, p.rho, V_cruise)
            m_gear = gear_mass(MTOM_guess, r_cr, p.r_fus_m)
            m_interior = interior_mass(MTOM_guess, p.rho, V_cruise)
            m_system_val = system_mass(MTOM_guess, p.l_fus_m, b)
            m_empty_val = empty_mass(m_wing, m_motor, m_rotor, p.m_crew, m_interior, m_fuselage, m_system_val, m_gear)
            # match compute(): payload added to empty+battery; do not add crew twice
            MTOM_est = m_empty_val + m_batt + p.m_pay
            return jnp.array([m_batt, m_empty_val, m_system_val, MTOM_est])

        J = jax.jacfwd(fun)(x)
        J = np.asarray(J)
        J = np.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)

        for i, out in enumerate(out_names):
            for j, inp in enumerate(in_names):
                partials[(out, inp)] = J[i, j]
