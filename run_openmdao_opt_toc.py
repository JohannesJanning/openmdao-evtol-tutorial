#!/usr/bin/env python3
"""Run optimization minimizing Total Operating Cost (TOC).

This mirrors `run_openmdao_opt.py` but sets the objective to `TOC_flight`.
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

    prob.model.add_subsystem('cons_comp', cons_comp)

    prob.model.connect('rotor_spacing', 'cons_comp.rotor_spacing')
    prob.model.connect('vertiport_span', 'cons_comp.vertiport_span')
    prob.model.connect('b', 'cons_comp.b')
    prob.model.connect('mtom.MTOM', 'cons_comp.MTOM')
    prob.model.connect('V_cruise', 'cons_comp.V_cruise')
    prob.model.connect('V_climb', 'cons_comp.V_climb')

    # Objective: minimize total operating cost per flight (TOC_flight)
    prob.model.add_objective('TOC_flight', ref=1000.0)

    # Constraints with scaling
    prob.model.add_constraint('cons_comp.c1', lower=0.0, ref=1.0)
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

    prob.driver.declare_coloring() if hasattr(prob.driver, 'declare_coloring') else None

    prob.setup()

    # Initialize design variables to verified design
    prob.set_val('b', x0[0])
    prob.set_val('c', x0[1])
    prob.set_val('r_cruise', x0[2])
    prob.set_val('r_hover', x0[3])
    prob.set_val('rho_bat', x0[4])
    prob.set_val('c_charge', x0[5])

    t0 = time.time()
    try:
        prob.run_driver()
    except Exception as exc:
        print('Optimization failed:', exc)
        raise
    t1 = time.time()

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

    toc_val = sval('TOC_flight')
    mtom_val = sval('MTOM')
    if mtom_val is None:
        mtom_val = float(np.asarray(prob.get_val('mtom.MTOM')).item())

    print('\n=== OpenMDAO Optimization Result (TOC) ===')
    print('Design vector: [b, c, r_cruise, r_hover, rho_bat, c_charge]')
    print([round(v, 6) for v in [b_opt, c_opt, r_cruise_opt, r_hover_opt, rho_bat_opt, c_charge_opt]])
    print(f'TOC_flight (EUR/flight): {toc_val:,.6f}')
    print(f'MTOM (kg): {mtom_val:,.6f}')
    print(f'Elapsed (s): {t1-t0:.1f}')

    # Post-optimization: evaluate full model and write results to Excel
    try:
        model_results, comparison_table = full_model_evaluation(
            b_opt, c_opt, r_cruise_opt, r_hover_opt, rho_bat_opt, c_charge_opt, parameters
        )

        write_results_to_excel(
            results_dict=model_results,
            comparison_list=comparison_table,
            mode="TOC",
            filename="optimized_results_TOC.xlsx",
        )
        print('Post-optimization evaluation complete. Results written to src/results/TOC/.')
    except Exception as exc:
        print('Post-optimization evaluation failed:', exc)
        raise


if __name__ == '__main__':
    build_and_run()
