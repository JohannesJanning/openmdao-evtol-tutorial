import openmdao.api as om
import jax
import jax.numpy as jnp
import numpy as np


class JaxMTOMImplicit(om.JaxImplicitComponent):
    """Implicit MTOM component using a simple JAX residual:

    Residual R = MTOM - MTOM_est

    - `MTOM` is the implicit state (output)
    - `MTOM_est` is an input (computed by `MassComp`)

    The residual is intentionally simple so that AD through the coupled
    system is stable: the detailed mass calculation stays in `MassComp` and
    its JAX partials are used by OpenMDAO when computing total derivatives.
    """

    def initialize(self):
        self.options.declare('parameters', default=None)

    def setup(self):
        # inputs that influence mass are provided by upstream components
        self.add_input('MTOM_est', val=1500.0)

        # implicit state
        self.add_output('MTOM', val=1500.0)

        # declare partials for the residual: dR/dMTOM and dR/dMTOM_est
        self.declare_partials(of='MTOM', wrt=['MTOM', 'MTOM_est'])

    def setup_partials(self):
        # configure solvers appropriate for small implicit state
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        self.linear_solver = om.DirectSolver()

    @staticmethod
    @jax.jit
    def _residual(MTOM, MTOM_est):
        # R = MTOM - MTOM_est
        r = MTOM - MTOM_est
        return jnp.nan_to_num(r, nan=0.0, posinf=1e9, neginf=-1e9)

    # JAX/OpenMDAO interface: provide compute_primal for JAX AD tracing
    def compute_primal(self, MTOM_est, MTOM):
        # compute_primal signature: (inputs..., outputs...)
        return self._residual(MTOM, MTOM_est)

    def compute_residual(self, inputs, outputs, residuals):
        MTOM = jnp.array(outputs['MTOM'])
        MTOM_est = jnp.array(inputs['MTOM_est'])
        residuals['MTOM'] = float(self._residual(MTOM, MTOM_est))

    def linearize(self, inputs, outputs, partials):
        # Analytical jacobian of R = MTOM - MTOM_est is trivial
        # dR/dMTOM = 1, dR/dMTOM_est = -1
            # Residual: R = MTOM - MTOM_est
            # Provide analytic residual Jacobians but add a tiny regularization
            # to help stabilize ill-conditioning when OpenMDAO inverts the
            # assembled Jacobian for totals. This is a Tikhonov-like guard.
            eps = 1e-8
            partials[('MTOM', 'MTOM')] = np.array([[1.0 + eps]], dtype=float)
            partials[('MTOM', 'MTOM_est')] = np.array([[-1.0]], dtype=float)
