# Cell 1: Imports and JAX/Path setup (Binder compatible)
import os
import sys
import time
import numpy as np

# Binder-friendly JAX settings
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_ENABLE_X64'] = '1'

# Ensure local imports work in cloud notebooks
sys.path.append(os.getcwd())

import jax
import openmdao.api as om
import pandas as pd

# optional: persistent jax compilation cache (local folder)
ROOT = os.getcwd()
jax.config.update('jax_compilation_cache_dir', os.path.join(ROOT, 'jax_cache'))
try:
    jax.config.update('jax_enable_x64', True)
except Exception:
    pass

import src.parameters as parameters
from src.optimizer.eVTOL_group import eVTOLGroup
from src.analysis.evaluation_model import full_model_evaluation, write_results_to_excel

print('Environment and imports ready')

# Cell 2: Design variable initialization (starting point)
x0 = np.array([15.0, 1.7025, 1.5511, 1.6868, 400.0, 1.0], dtype=float)
x0

# Cell 3: OpenMDAO Problem setup (build the problem)
prob = om.Problem(model=eVTOLGroup(parameters=parameters))

# Independent variables
iv = om.IndepVarComp()
iv.add_output('b', val=float(x0[0]))
iv.add_output('c', val=float(x0[1]))
iv.add_output('r_cruise', val=float(x0[2]))
iv.add_output('r_hover', val=float(x0[3]))
iv.add_output('rho_bat', val=float(x0[4]))
iv.add_output('c_charge', val=float(x0[5]))
prob.model.add_subsystem('iv', iv, promotes=['*'])

# Small ExecComp constraints (mirrors script)
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

# Connections
prob.model.connect('rotor_spacing', 'cons_comp.rotor_spacing')
prob.model.connect('vertiport_span', 'cons_comp.vertiport_span')
prob.model.connect('b', 'cons_comp.b')
prob.model.connect('mtom.MTOM', 'cons_comp.MTOM')
prob.model.connect('V_cruise', 'cons_comp.V_cruise')
prob.model.connect('V_climb', 'cons_comp.V_climb')

# Objective: minimize TOC per flight (TOC_flight)
prob.model.add_objective('TOC_flight', ref=1000.0)

# Constraints and design vars (same bounds/scaling)
prob.model.add_constraint('cons_comp.c1', lower=0.0, ref=1.0)
prob.model.add_constraint('vertiport_span', upper=15.0, ref=1.0)
prob.model.add_constraint('MTOM', upper=5700.0, ref=2000.0)
prob.model.add_constraint('V_cruise', upper=129.0, ref=100.0)
prob.model.add_constraint('V_climb', upper=129.0, ref=100.0)

prob.model.add_design_var('b', lower=6.0, upper=15.0, ref0=6.0, ref=15.0)
prob.model.add_design_var('c', lower=1.0, upper=2.5, ref0=1.0, ref=2.5)
prob.model.add_design_var('r_cruise', lower=0.6, upper=2.5, ref0=0.6, ref=2.5)
prob.model.add_design_var('r_hover', lower=0.6, upper=2.0, ref0=0.6, ref=2.0)
prob.model.add_design_var('rho_bat', lower=200.0, upper=400.0, ref0=200.0, ref=400.0)
prob.model.add_design_var('c_charge', lower=1.0, upper=4.0, ref0=1.0, ref=4.0)

# Driver setup (SLSQP)
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['tol'] = 1e-6
prob.driver.options['disp'] = True
prob.driver.declare_coloring() if hasattr(prob.driver, 'declare_coloring') else None

# Finalize setup and initialize values
prob.setup()
prob.set_val('b', x0[0])
prob.set_val('c', x0[1])
prob.set_val('r_cruise', x0[2])
prob.set_val('r_hover', x0[3])
prob.set_val('rho_bat', x0[4])
prob.set_val('c_charge', x0[5])

print('Problem built and ready')

# Cell 4: Run the optimization
t0 = time.time()
try:
    prob.run_driver()
except Exception as exc:
    print('Optimization failed:', exc)
    raise
t1 = time.time()
print(f'Elapsed (s): {t1-t0:.1f}')

# Cell 5: Extract results and run full model evaluation, then write Excel
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

print('Optimal design vector:')
print([b_opt, c_opt, r_cruise_opt, r_hover_opt, rho_bat_opt, c_charge_opt])

# Post-opt full model evaluation and save to Excel (static filename)
model_results, comparison_table = full_model_evaluation(
    b_opt, c_opt, r_cruise_opt, r_hover_opt, rho_bat_opt, c_charge_opt, parameters
)
# Baseline design vector for comparison (left column in the results table)
baseline_vector = [9.903657406161297, 1.0, 0.8411349802542505, 1.3997762343602163, 400.0, 2.0287900575874778]

# Write results to Excel and get the output path + DataFrame
out_path, df_results = write_results_to_excel(
    results_dict=model_results,
    comparison_list=comparison_table,
    mode='TOC',
    filename='optimized_results_TOC.xlsx',
    baseline=baseline_vector,
)

# Build comparison DataFrame locally for further use
df_comp = pd.DataFrame(comparison_table) if comparison_table else pd.DataFrame()

print('Post-optimization evaluation completed')

# --- 1. EVALUATION ---
# Compute all metrics from the optimized design variables
model_results, _ = full_model_evaluation(
    b_opt, c_opt, r_cruise_opt, r_hover_opt, rho_bat_opt, c_charge_opt, parameters
)

# --- 2. STORAGE ---
# Save the full results to Excel for the student to download
from src.analysis.evaluation_model import write_results_to_excel, display_model_dashboard
out_path, _ = write_results_to_excel(
    results_dict=model_results,
    comparison_list=[], # Passing empty list as it is not relevant
    mode='TOC',
    filename='optimized_results_TOC.xlsx',
    baseline=baseline_vector
)

# --- 3. VISUAL DASHBOARD ---
# Display the formatted results directly in the notebook
print("-" * 30)
print("  OPTIMIZED DESIGN DASHBOARD")
print("-" * 30)
display_model_dashboard(model_results, baseline=baseline_vector)

# --- 4. DOWNLOAD LINK ---
# Provide the clickable link for the Excel file
from IPython.display import FileLink, display as ipy_display
if os.path.exists(out_path):
    print("\nDownload full Excel report:")
    ipy_display(FileLink(out_path))


# Optimal design vector baseline: [9.903657406161297, 1.0, 0.8411349802542505, 1.3997762343602163, 400.0, 2.0287900575874778]