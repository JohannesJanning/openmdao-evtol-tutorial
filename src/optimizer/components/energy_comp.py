import openmdao.api as om
import numpy as np
import jax
import jax.numpy as jnp

from src.models_jax.energy.energy_trip import energy_total_trip
from src.models_jax.energy.energy_hover import energy_hover
from src.models_jax.energy.energy_climb import energy_climb
from src.models_jax.energy.energy_trip import energy_reserve
from src.models_jax.energy.energy_total_req import energy_total_required
from src.models_jax.aerodynamics.ROC import roc_calculation
from src.models_jax.time.time_climb import climb_time
from src.models_jax.time.time_cruise import compute_time_cruise
from src.models_jax.time.time_trip import total_trip_time


class EnergyComp(om.ExplicitComponent):
    """Compute energy requirements from mission profile and performance outputs.

    Inputs: V_cruise, V_climb, V_climb_hor, P_req_total_hover, P_req_total_climb, P_req_total_cruise, time params
    Outputs: E_total_req, E_trip, E_hover, E_climb, E_reserve
    """

    def initialize(self):
        self.options.declare('parameters')

    def setup(self):
        self.add_input('V_cruise', val=0.0)
        self.add_input('V_climb', val=0.0)
        self.add_input('V_climb_hor', val=0.0)
        self.add_input('P_req_total_hover', val=0.0)
        self.add_input('P_req_total_climb', val=0.0)
        self.add_input('P_req_total_cruise', val=0.0)

        self.add_output('E_total_req', val=0.0)
        self.add_output('E_trip', val=0.0)
        self.add_output('E_hover', val=0.0)
        self.add_output('E_climb', val=0.0)
        self.add_output('E_reserve', val=0.0)
        self.add_output('t_cruise', val=0.0)
        self.add_output('t_trip', val=0.0)

        # We'll supply analytic JAX partials in compute_partials
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        p = self.options['parameters']
        V_cruise = inputs['V_cruise'][0]
        V_climb = inputs['V_climb'][0]
        V_climb_hor = inputs['V_climb_hor'][0]
        P_hover = inputs['P_req_total_hover'][0]
        P_climb = inputs['P_req_total_climb'][0]
        P_cruise = inputs['P_req_total_cruise'][0]

        # compute climb and cruise times using ROC and cruise/climb speeds
        eps = 1e-3
        # rate of climb from climb angle and V_climb
        ROC = roc_calculation(p.alpha_deg_climb, V_climb)
        t_climb = climb_time(p.h_cruise, p.h_hover, ROC)
        t_hover = p.time_hover
        t_cruise = compute_time_cruise(p.distance_trip, V_climb_hor, t_climb, V_cruise)

        E_hover = energy_hover(P_hover, t_hover)
        E_climb = energy_climb(P_climb, t_climb)
        E_cruise = (P_cruise * t_cruise) / 3600.0
        E_trip = energy_total_trip(E_hover, E_climb, E_cruise)
        E_reserve = energy_reserve(P_cruise, p.time_reserve)

        E_total_req = energy_total_required(E_trip, E_reserve)

        outputs['E_trip'] = E_trip
        outputs['E_hover'] = E_hover
        outputs['E_climb'] = E_climb
        outputs['E_reserve'] = E_reserve
        outputs['E_total_req'] = E_total_req
        outputs['t_cruise'] = t_cruise
        outputs['t_trip'] = t_hover + t_climb + t_cruise

    def compute_partials(self, inputs, partials):
        in_names = ['V_cruise', 'V_climb', 'V_climb_hor', 'P_req_total_hover', 'P_req_total_climb', 'P_req_total_cruise']
        out_names = ['E_total_req', 'E_trip', 'E_hover', 'E_climb', 'E_reserve', 't_cruise', 't_trip']

        x = jnp.array([inputs[n][0] for n in in_names])

        def fun(x):
            V_cruise, V_climb, V_climb_hor, P_hover, P_climb, P_cruise = x
            p = self.options['parameters']
            eps = 1e-3
            ROC = roc_calculation(self.options['parameters'].alpha_deg_climb, V_climb)
            t_climb = climb_time(self.options['parameters'].h_cruise, self.options['parameters'].h_hover, ROC)
            t_hover = self.options['parameters'].time_hover
            t_cruise = compute_time_cruise(self.options['parameters'].distance_trip, V_climb_hor, t_climb, V_cruise)

            E_hover = energy_hover(P_hover, t_hover)
            E_climb = energy_climb(P_climb, t_climb)
            E_cruise = (P_cruise * t_cruise) / 3600.0
            E_trip = energy_total_trip(E_hover, E_climb, E_cruise)
            E_reserve = energy_reserve(P_cruise, p.time_reserve)
            E_total_req = energy_total_required(E_trip, E_reserve)

            t_trip = total_trip_time(t_hover, t_climb, t_cruise)
            return jnp.array([E_total_req, E_trip, E_hover, E_climb, E_reserve, t_cruise, t_trip])

        J = jax.jacfwd(fun)(x)
        J = np.asarray(J)
        J = np.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)

        for i, out in enumerate(out_names):
            for j, inp in enumerate(in_names):
                partials[(out, inp)] = J[i, j]
