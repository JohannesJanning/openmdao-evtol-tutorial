#!/usr/bin/env python3
"""Run a single OpenMDAO optimization using the verified JAX-native components.

This script:
- Enables JAX x64 and sets a persistent JAX compilation cache at `./jax_cache`.
- Builds an `om.Problem(model=eVTOLGroup(parameters=...))` and configures a
  `ScipyOptimizeDriver` using SLSQP with strict scaling and bounds.
- Uses the NewtonSolver + DirectSolver already configured inside `eVTOLGroup`.

Usage:
  python3 run_openmdao_opt.py

Notes:
- SPL/noise constraints are NOT added. MTOM and other constraints are scaled.
"""
import os
import sys
import time

# Ensure repo root is cwd
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

# Set JAX env early
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

# persistent compilation cache
jax.config.update("jax_compilation_cache_dir", os.path.join(ROOT, "jax_cache"))
try:
    jax.config.update("jax_enable_x64", True)
except Exception:
    pass

sys.path.insert(0, ROOT)

import numpy as np
import openmdao.api as om

import src.parameters as parameters
from src.optimizer.eVTOL_group import eVTOLGroup
from src.analysis.evaluation_model import full_model_evaluation, write_results_to_excel


def build_and_run():
    prob = om.Problem(model=eVTOLGroup(parameters=parameters))

    # Add design variables as independent variables at top-level
    iv = om.IndepVarComp()
    # Use the verified design as starting point
    x0 = np.array([15.0, 1.7025, 1.5511, 1.6868, 400.0, 1.0], dtype=float)
    iv.add_output('b', val=float(x0[0]))
    iv.add_output('c', val=float(x0[1]))
    iv.add_output('r_cruise', val=float(x0[2]))
    iv.add_output('r_hover', val=float(x0[3]))
    iv.add_output('rho_bat', val=float(x0[4]))
    iv.add_output('c_charge', val=float(x0[5]))

    prob.model.add_subsystem('iv', iv, promotes=['*'])

    # Add an ExecComp to form a small set of constraints (no SPL/noise)
    # c1: b - rotor_spacing >= 0
    # c2: 15.0 - vertiport_span >= 0
    # c3: 5700.0 - MTOM >= 0
    # c4: 129.0 - V_cruise >= 0
    # c5: 129.0 - V_climb >= 0
    cons_expr = (
        'c1 = b - rotor_spacing',
        'c2 = 15.0 - vertiport_span',
        'c3 = 5700.0 - MTOM',
        'c4 = 129.0 - V_cruise',
        'c5 = 129.0 - V_climb',
    )
    cons_comp = om.ExecComp('\n'.join(cons_expr),
                           b=15.0,
                           rotor_spacing=1.0,
                           vertiport_span=1.0,
                           MTOM=1500.0,
                           V_cruise=50.0,
                           V_climb=10.0)

    # Add constraint component without promoting inputs to avoid accidental
    # shadowing; connect each input explicitly from the producing subsystem.
    prob.model.add_subsystem('cons_comp', cons_comp)

    # Connect geometry outputs explicitly (geometry promoted to group-level)
    prob.model.connect('rotor_spacing', 'cons_comp.rotor_spacing')
    prob.model.connect('vertiport_span', 'cons_comp.vertiport_span')

    # Connect the design `b` from the independent variable subsystem
    prob.model.connect('b', 'cons_comp.b')

    # Connect MTOM explicitly from the implicit MTOM subsystem (avoid shadowing)
    prob.model.connect('mtom.MTOM', 'cons_comp.MTOM')

    # Connect speeds from the performance component (promoted to group-level)
    prob.model.connect('V_cruise', 'cons_comp.V_cruise')
    prob.model.connect('V_climb', 'cons_comp.V_climb')

    # Objective: the GWP component promotes `GWP_annual_ops` to group level,
    # reference the top-level name and apply scaling ref to keep values ~1.0.
    prob.model.add_objective('GWP_annual_ops', ref=52000.0)

    # Constraints with scaling
    # Keep the geometry spacing constraint via the ExecComp
    prob.model.add_constraint('cons_comp.c1', lower=0.0, ref=1.0)
    # For the remaining constraints, directly constrain the producing outputs
    # to avoid any ExecComp wiring/shadowing issues.
    prob.model.add_constraint('vertiport_span', upper=15.0, ref=1.0)
    prob.model.add_constraint('MTOM', upper=5700.0, ref=2000.0)
    prob.model.add_constraint('V_cruise', upper=129.0, ref=100.0)
    prob.model.add_constraint('V_climb', upper=129.0, ref=100.0)

    # Design variable bounds and scaling (strict physical bounds)
    prob.model.add_design_var('b', lower=6.0, upper=15.0, ref0=6.0, ref=15.0)
    prob.model.add_design_var('c', lower=1.0, upper=2.5, ref0=1.0, ref=2.5)
    prob.model.add_design_var('r_cruise', lower=0.6, upper=2.5, ref0=0.6, ref=2.5)
    prob.model.add_design_var('r_hover', lower=0.6, upper=2.0, ref0=0.6, ref=2.0)
    prob.model.add_design_var('rho_bat', lower=200.0, upper=400.0, ref0=200.0, ref=400.0)
    prob.model.add_design_var('c_charge', lower=1.0, upper=4.0, ref0=1.0, ref=4.0)

    # Driver: Scipy SLSQP
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 1e-6
    prob.driver.options['disp'] = True

    # Use analytic derivatives where components provide them
    prob.driver.declare_coloring() if hasattr(prob.driver, 'declare_coloring') else None

    # Setup and run
    prob.setup()

    # Initialize design variables to verified design
    prob.set_val('b', x0[0])
    prob.set_val('c', x0[1])
    prob.set_val('r_cruise', x0[2])
    prob.set_val('r_hover', x0[3])
    prob.set_val('rho_bat', x0[4])
    prob.set_val('c_charge', x0[5])

    

    # Run driver
    t0 = time.time()
    try:
        prob.run_driver()
    except Exception as exc:
        print('Optimization failed:', exc)
        raise
    t1 = time.time()

    # Extract final results
    def sval(name):
        try:
            return float(np.asarray(prob.get_val(name)).item())
        except Exception:
            return None

    b_opt = sval('b')
    c_opt = sval('c')
    r_cruise_opt = sval('r_cruise')
    r_hover_opt = sval('r_hover')
    rho_bat_opt = sval('rho_bat')
    c_charge_opt = sval('c_charge')

    # GWP and MTOM live at group/top level (GWP promoted by the gwp subsystem)
    gwp_val = sval('GWP_annual_ops')
    mtom_val = sval('MTOM')
    if mtom_val is None:
        # fall back to implicit subsystem output
        mtom_val = float(np.asarray(prob.get_val('mtom.MTOM')).item())

    print('\n=== OpenMDAO Optimization Result ===')
    print('Design vector: [b, c, r_cruise, r_hover, rho_bat, c_charge]')
    print([round(v, 6) for v in [b_opt, c_opt, r_cruise_opt, r_hover_opt, rho_bat_opt, c_charge_opt]])
    print(f'GWP_annual_ops (kg CO2e/yr): {gwp_val:,.6f}')
    print(f'MTOM (kg): {mtom_val:,.6f}')
    print(f'Elapsed (s): {t1-t0:.1f}')

    # After optimization, run the full model evaluation with the optimal design
    try:
        model_results, comparison_table = full_model_evaluation(
            b_opt, c_opt, r_cruise_opt, r_hover_opt, rho_bat_opt, c_charge_opt, parameters
        )

        write_results_to_excel(
            results_dict=model_results,
            comparison_list=comparison_table,
            mode="GWP",
            filename="optimized_results_GWP.xlsx",
        )
        print('Post-optimization evaluation complete. Results written to src/results/GWP/.')
    except Exception as exc:
        print('Post-optimization evaluation failed:', exc)
        raise


if __name__ == '__main__':
    build_and_run()
