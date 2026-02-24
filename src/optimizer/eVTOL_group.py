import openmdao.api as om

from src.optimizer.components.aerodynamics_comp import AerodynamicsComp
from src.optimizer.components.mtom_implicit import JaxMTOMImplicit
from src.optimizer.components.performance_comp import PerformanceComp
from src.optimizer.components.energy_comp import EnergyComp
from src.optimizer.components.mass_comp import MassComp
from src.optimizer.components.geometry_comp import GeometryComp


class eVTOLGroup(om.Group):
    """Group composing core components according to the XDSM diagram.

    For the first iteration this wires `aero` -> `mtom`. We'll extend the
    wiring iteratively to include Performance, Time, Energy, Mass, Environment.
    """

    def initialize(self):
        self.options.declare('parameters')

    def setup(self):
        params = self.options['parameters']

        # Wire components following the physical information flow:
        # Design -> Aerodynamics -> MTOM (implicit) -> Performance/Energy/Mass -> GWP/Ops
        self.add_subsystem('aero', AerodynamicsComp(parameters=params), promotes=['*'])
        # MTOM implicit state sits immediately after aero so its state is
        # available to downstream components during the coupled solve.
        self.add_subsystem('mtom', JaxMTOMImplicit(parameters=params))
        # geometry helper (spacing / vertiport span)
        self.add_subsystem('geom', GeometryComp(parameters=params), promotes=['*'])
        self.add_subsystem('perf', PerformanceComp(parameters=params), promotes=['*'])
        self.add_subsystem('energy', EnergyComp(parameters=params), promotes=['*'])
        self.add_subsystem('mass', MassComp(parameters=params), promotes=['*'])
        # operational and environmental components
        from src.optimizer.components.ops_comp import OpsComp

        self.add_subsystem('ops', OpsComp(parameters=params), promotes=['*'])
        # GWP component
        from src.optimizer.components.gwp_comp import GWPComp
        self.add_subsystem('gwp', GWPComp(parameters=params), promotes=['*'])

        # Economic component: computes operating cost metrics (TOC)
        from src.optimizer.components.economic_comp import EconomicComp
        self.add_subsystem('economic', EconomicComp(parameters=params), promotes=['*'])

        # Connect feedback: mass produces an MTOM estimate which defines the
        # residual for the implicit MTOM state. The implicit `mtom.MTOM`
        # supplies the MTOM guess to downstream components.
        # mass.MTOM_est is promoted to group-level; mtom accepts an input
        # named 'MTOM_est' on its subsystem, so connect the promoted name
        # into the mtom subsystem.
        self.connect('MTOM_est', 'mtom.MTOM_est')

        # mtom's output 'MTOM' is not promoted; connect it into the group-level
        # MTOM input used by promoted downstream components (mass, perf).
        self.connect('mtom.MTOM', 'MTOM')

        # Expose solvers for the coupled feedback loop. Use Newton on the
        # group (solve_subsystems=True) and Direct linear solver with assembly
        # so OpenMDAO can use the implicit function theorem with provided
        # analytic partials.
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        self.nonlinear_solver.options['maxiter'] = 50
        self.nonlinear_solver.options['rtol'] = 1e-6
        self.linear_solver = om.DirectSolver(assemble_jac=True)
