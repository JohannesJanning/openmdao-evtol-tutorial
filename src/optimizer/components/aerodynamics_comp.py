import openmdao.api as om
import numpy as np
import jax
import jax.numpy as jnp

from src.models_jax.aerodynamics.AR import aspect_ratio as AR_calculation
from src.models_jax.aerodynamics.lift_coefficient import cl_from_lift as cl_calculation
from src.models_jax.aerodynamics.drag_coefficient import cd_calculation


class AerodynamicsComp(om.ExplicitComponent):
    """Compute aerodynamic coefficients from wing geometry.

    Inputs: `b`, `c` (span, chord)
    Outputs: `CL_cruise`, `CD_cruise`, `CL_climb`, `CD_climb`
    """

    def setup(self):
        self.add_input('b', val=10.0)
        self.add_input('c', val=1.5)

        self.add_output('CL_cruise', val=0.0)
        self.add_output('CD_cruise', val=0.0)
        self.add_output('CL_climb', val=0.0)
        self.add_output('CD_climb', val=0.0)

        # provide analytic JAX partials
        self.declare_partials('*', '*')

    def initialize(self):
        self.options.declare('parameters')

    def compute_partials(self, inputs, partials):        
        # compute jacobian via JAX
        x = jnp.array([inputs['b'][0], inputs['c'][0]])

        def fun(x):
            b, c = x
            p = self.options['parameters']
            AR = AR_calculation(b, c)
            CL_cruise = cl_calculation(p.alpha_deg_cruise, AR, p.c_l_0, p.e)
            CD_cruise = cd_calculation(CL_cruise, AR, p.c_d_min, p.e)
            CL_climb = cl_calculation(p.alpha_deg_climb, AR, p.c_l_0, p.e)
            CD_climb = cd_calculation(CL_climb, AR, p.c_d_min, p.e)
            return jnp.array([CL_cruise, CD_cruise, CL_climb, CD_climb])

        J = jax.jacfwd(fun)(x)
        J = np.asarray(J)
        J = np.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)

        out_names = ['CL_cruise', 'CD_cruise', 'CL_climb', 'CD_climb']
        in_names = ['b', 'c']
        for i, out in enumerate(out_names):
            for j, inp in enumerate(in_names):
                partials[(out, inp)] = J[i, j]

    def compute(self, inputs, outputs):
        # inputs are array-like from OpenMDAO; extract safe Python scalars
        b = inputs['b'][0]
        c = inputs['c'][0]

        AR = AR_calculation(b, c)
        # use simple proxy formulas here as placeholders; we'll replace with
        # direct parameterized calls when wiring the full group.
        p = self.options['parameters']
        CL_cruise = cl_calculation(p.alpha_deg_cruise, AR, p.c_l_0, p.e)
        CD_cruise = cd_calculation(CL_cruise, AR, p.c_d_min, p.e)
        CL_climb = cl_calculation(p.alpha_deg_climb, AR, p.c_l_0, p.e)
        CD_climb = cd_calculation(CL_climb, AR, p.c_d_min, p.e)

        outputs['CL_cruise'] = float(np.asarray(CL_cruise).item())
        outputs['CD_cruise'] = float(np.asarray(CD_cruise).item())
        outputs['CL_climb'] = float(np.asarray(CL_climb).item())
        outputs['CD_climb'] = float(np.asarray(CD_climb).item())
