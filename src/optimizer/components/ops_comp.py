import openmdao.api as om
import numpy as np
import jax
import jax.numpy as jnp

from src.models_jax.operations.ops_model import (
    turnaround_time,
    time_efficiency_ratio,
    daily_flight_cycles,
    annual_flight_cycles,
)


class OpsComp(om.ExplicitComponent):
    """Compute operational flight cycles per year (FC_a) from trip time.

    Inputs: t_trip
    Outputs: FC_a
    """

    def initialize(self):
        self.options.declare('parameters')

    def setup(self):
        self.add_input('t_trip', val=3600.0)
        # accept runtime charger efficiency and depth-of-discharge
        self.add_input('c_charge', val=1.0)
        self.add_input('DOD', val=0.3)
        self.add_output('FC_a', val=260.0)

        # We'll provide analytic JAX partials
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        p = self.options['parameters']
        t_trip = inputs['t_trip'][0]

        # use runtime battery charging behaviour provided by the model
        c_charge = inputs['c_charge'][0]
        DOD = inputs['DOD'][0]

        time_turn = float(np.asarray(turnaround_time(c_charge, DOD)))
        DH = float(np.asarray(time_efficiency_ratio(time_turn, t_trip)))
        FC_d = float(np.asarray(daily_flight_cycles(p.T_D, t_trip, DH)))
        FC_a = float(np.asarray(annual_flight_cycles(p.N_wd, FC_d)))

        outputs['FC_a'] = FC_a

    def compute_partials(self, inputs, partials):
        p = self.options['parameters']

        in_names = ['t_trip', 'DOD', 'c_charge']

        x = jnp.array([inputs[n][0] for n in in_names])

        def fun(x):
            t_trip, DOD, c_charge = x
            time_turn = turnaround_time(c_charge, DOD)
            DH = time_efficiency_ratio(time_turn, t_trip)
            FC_d = daily_flight_cycles(p.T_D, t_trip, DH)
            FC_a = annual_flight_cycles(p.N_wd, FC_d)
            # ensure FC_a is a JAX scalar/1D array for jacobian tracing
            FC_a = jnp.asarray(FC_a)
            FC_a = jnp.squeeze(FC_a)
            return jnp.atleast_1d(FC_a)

        J = jax.jacfwd(fun)(x)
        J = np.asarray(J)
        J = np.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)

        # assign partials for each input
        for j, inp in enumerate(in_names):
            partials[('FC_a', inp)] = np.atleast_2d(J[0, j])
