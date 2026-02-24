import openmdao.api as om
import numpy as np

import src.parameters as pmod
from src.models_jax.mass.mtom_model import full_mtom_model
from src.models_jax.mass.mtom_iteration import mtom_iteration_loop


class PyMTOMImplicit(om.ImplicitComponent):
    """Pure-Python implicit component wrapping the robust MTOM loop.

    This avoids tracing issues by using the non-jitted `mtom_iteration_loop`
    for `solve_nonlinear` and a finite-difference `linearize` that calls the
    JAX `full_mtom_model` at concrete Python values.
    """

    def initialize(self):
        self.options.declare('parameters')

    def setup(self):
        self.add_input('b', val=10.0)
        self.add_input('c', val=1.5)
        self.add_input('r_cruise', val=1.0)
        self.add_input('r_hover', val=1.0)
        self.add_input('rho_bat', val=300.0)

        # implicit state
        self.add_output('MTOM', val=float(self.options['parameters'].MTOM_initial))

        # declare partials we'll set in linearize
        self.declare_partials(of='MTOM', wrt=['b', 'c', 'r_cruise', 'r_hover', 'rho_bat', 'MTOM'])

        # use Newton solver for the scalar implicit solve
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        self.linear_solver = om.DirectSolver()

    def solve_nonlinear(self, inputs, outputs, tol=1e-6):
        b = inputs['b'][0]
        c = inputs['c'][0]
        r_cruise = inputs['r_cruise'][0]
        r_hover = inputs['r_hover'][0]
        rho_bat = inputs['rho_bat'][0]

        params = self.options['parameters']

        MTOM_conv = mtom_iteration_loop(params.MTOM_initial, b, c, r_cruise, r_hover, rho_bat, params, tol=tol, max_iter=600, verbose=False)
        outputs['MTOM'] = float(MTOM_conv)

    def linearize(self, inputs, outputs, partials):
        # JAX-based linearization of the MTOM residual R = full_mtom_model(m, params, ...) - m
        params = self.options['parameters']
        import jax
        import jax.numpy as jnp

        # extract current point
        b0 = inputs['b'][0]
        c0 = inputs['c'][0]
        rcr0 = inputs['r_cruise'][0]
        rhu0 = inputs['r_hover'][0]
        rb0 = inputs['rho_bat'][0]
        m0 = outputs['MTOM'][0]

        # JAX-compatible residual function taking a single vector
        def res_vec(x):
            m, b, c, rcr, rhu, rb = x
            # compute residual using jax-enabled full_mtom_model
            r = full_mtom_model(m, params, b, c, rcr, rhu, rb) - m
            # ensure scalar jax type
            return jnp.asarray(r)

        x0 = jnp.array([m0, b0, c0, rcr0, rhu0, rb0], dtype=jnp.float32)

        # compute jacobian (1 x 6)
        try:
            J = jax.jacfwd(res_vec)(x0)
        except Exception:
            # fall back to finite-diff numeric linearization to remain robust
            return super().linearize(inputs, outputs, partials)

        # J may be a scalar array; coerce to numpy
        J = np.asarray(J).reshape(-1)

        # map derivatives: order [m, b, c, rcr, rhu, rb]
        dm = float(np.nan_to_num(J[0], nan=0.0, posinf=0.0, neginf=0.0))
        db = float(np.nan_to_num(J[1], nan=0.0, posinf=0.0, neginf=0.0))
        dc = float(np.nan_to_num(J[2], nan=0.0, posinf=0.0, neginf=0.0))
        drcr = float(np.nan_to_num(J[3], nan=0.0, posinf=0.0, neginf=0.0))
        drhu = float(np.nan_to_num(J[4], nan=0.0, posinf=0.0, neginf=0.0))
        drb = float(np.nan_to_num(J[5], nan=0.0, posinf=0.0, neginf=0.0))

        partials[('MTOM', 'b')] = np.array([[db]], dtype=float)
        partials[('MTOM', 'c')] = np.array([[dc]], dtype=float)
        partials[('MTOM', 'r_cruise')] = np.array([[drcr]], dtype=float)
        partials[('MTOM', 'r_hover')] = np.array([[drhu]], dtype=float)
        partials[('MTOM', 'rho_bat')] = np.array([[drb]], dtype=float)
        # If the dR/dm is numerically zero (singular), fall back to FD linearization
        if abs(dm) < 1e-12 or not np.isfinite(dm):
            # finite-difference linearization around current point (robust fallback)
            params = self.options['parameters']
            import numpy as _np

            def res_at(b, c, rcr, rhu, rb, m):
                val = full_mtom_model(m, params, b, c, rcr, rhu, rb)
                return float(np.asarray(val)) - float(m)

            base = res_at(b0, c0, rcr0, rhu0, rb0, m0)
            eps = 1e-6
            db = (res_at(b0 + eps, c0, rcr0, rhu0, rb0, m0) - base) / eps
            dc = (res_at(b0, c0 + eps, rcr0, rhu0, rb0, m0) - base) / eps
            drcr = (res_at(b0, c0, rcr0 + eps, rhu0, rb0, m0) - base) / eps
            drhu = (res_at(b0, c0, rcr0, rhu0 + eps, rb0, m0) - base) / eps
            drb = (res_at(b0, c0, rcr0, rhu0, rb0 + eps, m0) - base) / eps
            dm = (res_at(b0, c0, rcr0, rhu0, rb0, m0 + eps) - base) / eps

            def s(x):
                if np.isnan(x) or np.isinf(x):
                    return 0.0
                return float(x)

            partials[('MTOM', 'b')] = np.array([[s(db)]], dtype=float)
            partials[('MTOM', 'c')] = np.array([[s(dc)]], dtype=float)
            partials[('MTOM', 'r_cruise')] = np.array([[s(drcr)]], dtype=float)
            partials[('MTOM', 'r_hover')] = np.array([[s(drhu)]], dtype=float)
            partials[('MTOM', 'rho_bat')] = np.array([[s(drb)]], dtype=float)
            partials[('MTOM', 'MTOM')] = np.array([[s(dm)]], dtype=float)
            return

        partials[('MTOM', 'MTOM')] = np.array([[dm]], dtype=float)
