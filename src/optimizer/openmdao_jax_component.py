"""
OpenMDAO component that wraps the JAX full-model evaluation to expose
inputs and an objective `ops_gwp_annual` for gradient-based optimization.

This is a minimal, non-jitted wrapper that calls `full_model_evaluation_jax`
and returns the annual operational GWP.
"""
from __future__ import annotations

try:
    import openmdao.api as om
except Exception:
    om = None

from src.analysis.evaluation_model_jax import full_model_evaluation_jax
from src.analysis.evaluation_model_jax_jittable import gwp_and_jacobian
import src.parameters as pmod


class JAXFullModelComp(om.ExplicitComponent if om is not None else object):
    def setup(self):
        # design variables
        self.add_input('b', val=10.0)
        self.add_input('c', val=1.2)
        self.add_input('R_prop_cruise', val=0.8)
        self.add_input('R_prop_hover', val=0.9)
        self.add_input('rho_bat', val=300.0)
        self.add_input('c_charge', val=2.0)

        # outputs
        self.add_output('ops_gwp_annual', val=1.0)

        # declare partials (we'll provide exact JAX derivatives)
        self.declare_partials('ops_gwp_annual', 'b')
        self.declare_partials('ops_gwp_annual', 'c')
        self.declare_partials('ops_gwp_annual', 'R_prop_cruise')
        self.declare_partials('ops_gwp_annual', 'R_prop_hover')
        self.declare_partials('ops_gwp_annual', 'rho_bat')
        self.declare_partials('ops_gwp_annual', 'c_charge')

    def compute(self, inputs, outputs):
        res = full_model_evaluation_jax(inputs['b'], inputs['c'], inputs['R_prop_cruise'], inputs['R_prop_hover'], inputs['rho_bat'], inputs['c_charge'], pmod)
        outputs['ops_gwp_annual'] = res['gwp_ops_annual']

    def compute_partials(self, inputs, partials):
        import jax.numpy as jnp
        design = jnp.array([inputs['b'], inputs['c'], inputs['R_prop_cruise'], inputs['R_prop_hover'], inputs['rho_bat'], inputs['c_charge']])
        gwp, jac = gwp_and_jacobian(design, pmod)
        # fill partials
        partials['ops_gwp_annual', 'b'] = float(jac[0])
        partials['ops_gwp_annual', 'c'] = float(jac[1])
        partials['ops_gwp_annual', 'R_prop_cruise'] = float(jac[2])
        partials['ops_gwp_annual', 'R_prop_hover'] = float(jac[3])
        partials['ops_gwp_annual', 'rho_bat'] = float(jac[4])
        partials['ops_gwp_annual', 'c_charge'] = float(jac[5])


def build_and_run_optimization():
    if om is None:
        raise RuntimeError('OpenMDAO not available in this environment')

    prob = om.Problem()
    model = prob.model
    model.add_subsystem('jax_comp', JAXFullModelComp(), promotes=['*'])

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 1e-6

    # design variables
    prob.model.add_design_var('b', lower=5.0, upper=25.0)
    prob.model.add_design_var('c', lower=0.5, upper=3.0)
    prob.model.add_design_var('R_prop_hover', lower=0.5, upper=3.0)
    prob.model.add_design_var('R_prop_cruise', lower=0.5, upper=3.0)
    prob.model.add_design_var('rho_bat', lower=150.0, upper=600.0)

    # objective
    prob.model.add_objective('ops_gwp_annual')

    prob.setup()
    prob.set_solver_print(level=0)
    prob['b'] = 15.0
    prob['c'] = 1.5
    prob['R_prop_cruise'] = 1.5
    prob['R_prop_hover'] = 1.5
    prob['rho_bat'] = 300.0
    prob['c_charge'] = 2.0

    prob.run_driver()
    return prob
