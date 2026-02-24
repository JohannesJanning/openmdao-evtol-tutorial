# eVTOL OpenMDAO v3

Repository of the eVTOL conceptual design and optimization models using OpenMDAO and JAX-enabled components.

Quick start

1. Create a virtual environment and activate it (macOS):

   python -m venv .venv
   source .venv/bin/activate

2. Install dependencies:

   pip install -r requirements.txt

3. Run examples:

   - Run optimization scripts: `python run_openmdao_opt_gwp.py` or `python run_openmdao_opt_toc.py`
   - Or open the notebooks `01_optimize_GWP.ipynb` and `02_optimize_TOC.ipynb`.

Outputs

- Excel results are saved under `src/results/GWP/` and `src/results/TOC/`.
- N2 visualizations are saved as HTML in the same folders.

Notes

- The repository uses JAX in parts; if you don't need JAX variants, the primary NumPy-based evaluation is in `src/analysis/evaluation_model.py`.