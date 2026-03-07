import openmdao.api as om

# Consolidated Imports
from src.optimizer.components.aerodynamics_comp import AerodynamicsComp
from src.optimizer.components.mtom_implicit import JaxMTOMImplicit
from src.optimizer.components.performance_comp import PerformanceComp
from src.optimizer.components.energy_comp import EnergyComp
from src.optimizer.components.mass_comp import MassComp
from src.optimizer.components.geometry_comp import GeometryComp
from src.optimizer.components.ops_comp import OpsComp
from src.optimizer.components.gwp_comp import GWPComp
from src.optimizer.components.economic_comp import EconomicComp

class eVTOLGroup(om.Group):
    """Group composing core components according to the XDSM diagram."""

    def initialize(self):
        self.options.declare('parameters')

    def setup(self):
        params = self.options['parameters']

        # --- 1. PHYSICAL DESIGN & GEOMETRY ---
        self.add_subsystem('aero', AerodynamicsComp(parameters=params), promotes=['*'])
        self.add_subsystem('geom', GeometryComp(parameters=params), promotes=['*'])

        # --- 2. THE IMPLICIT MASS LOOP ---
        # Solving for MTOM where: MTOM = f(Structural Mass + Battery Mass)
        self.add_subsystem('mtom', JaxMTOMImplicit(parameters=params))
        self.add_subsystem('mass', MassComp(parameters=params), promotes=['*'])

        # --- 3. MISSION PERFORMANCE & ENERGY ---
        self.add_subsystem('perf', PerformanceComp(parameters=params), promotes=['*'])
        self.add_subsystem('energy', EnergyComp(parameters=params), promotes=['*'])

        # --- 4. OPERATIONS, ENVIRONMENT & ECONOMICS ---
        self.add_subsystem('ops', OpsComp(parameters=params), promotes=['*'])
        self.add_subsystem('gwp', GWPComp(parameters=params), promotes=['*'])
        self.add_subsystem('economic', EconomicComp(parameters=params), promotes=['*'])

        # --- DATA WIRING: THE FEEDBACK LOOP ---
        # Bridges the Mass Estimation to the Implicit State Solver
        self.connect('MTOM_est', 'mtom.MTOM_est')
        self.connect('mtom.MTOM', 'MTOM')

        # --- SOLVER CONFIGURATION ---
        # Newton handles the cyclic dependency between Weight and Energy
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        self.nonlinear_solver.options['maxiter'] = 50
        self.nonlinear_solver.options['rtol'] = 1e-6
        self.nonlinear_solver.options['iprint'] = 2 # Shows convergence in the console
        
        self.linear_solver = om.DirectSolver(assemble_jac=True)