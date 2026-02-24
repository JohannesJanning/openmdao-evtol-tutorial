# eVTOL Multidisciplinary Design Optimization (MDO) Tutorial

<a href="https://mybinder.org/v2/gh/JohannesJanning/openmdao-evtol-tutorial/main?urlpath=lab/tree/README.md" target="_blank">
  <img src="https://mybinder.org/badge_logo.svg" alt="Binder">
</a>

This repository provides an interactive tutorial for the conceptual design and optimization of Electric Vertical Take-off and Landing (eVTOL) aircraft. It allows students and researchers to explore the trade-offs between environmental sustainability and economic viability in future urban air mobility.

## 1. Research Context
The models and optimization logic provided here are based on the multidisciplinary design optimization framework introduced in the following paper:

> **Janning, J., Armanini, S. F., & Fasel, U. (2025).** [Future pathways for eVTOLs: A design optimization perspective](https://doi.org/10.48550/arXiv.2412.18078). *arXiv:2412.18078 [eess.SY]*.

The framework integrates conventional aircraft design elements with comprehensive operational cost models to capture stakeholder-centric objectives, such as profit modeling, cost-efficiency, and sustainability strategies.

## 2. Technical Framework
The project utilizes a high-performance computational stack to solve complex engineering trade-offs:

* **[OpenMDAO](https://github.com/OpenMDAO/OpenMDAO):** An open-source framework for multidisciplinary analysis and optimization. It manages the coupling between aerodynamics, mass estimation, energy requirements, and cost models.
* **[JAX](https://github.com/jax-ml/jax):** A library for composable transformations of Python and NumPy programs. It enables efficient gradient-based optimization through automatic differentiation, which is critical for the high-dimensional design spaces in eVTOL modeling.

## 3. Getting Started
### Accessing the Workspace
Click the **Binder** badge at the top of this page to launch a cloud-resident JupyterLab session. All necessary dependencies are pre-installed in the container environment.

### Optimization Workflows
The tutorial consists of two primary study cases:
1.  **`01_optimize_GWP.ipynb`**: Focuses on minimizing the Global Warming Potential (GWP) of the aircraft mission.
2.  **`02_optimize_TOC.ipynb`**: Focuses on minimizing the Total Operating Cost (TOC) to evaluate economic incentives.

To execute a study, open a notebook and select **Cell > Run All** from the top menu.

## 4. Repository Structure
* **`01_optimize_GWP.ipynb` & `02_optimize_TOC.ipynb`**: Primary interactive tutorial notebooks.
* **`run_openmdao_opt_gwp.py` & `run_openmdao_opt_toc.py`**: Raw optimization scripts for reference.
* **`src/`**: The core source code directory.
    * **`models_jax/`**: Physics and cost components implemented in JAX.
    * **`analysis/`**: Post-optimization processing and evaluation scripts.
    * **`parameters.py`**: Central configuration for aircraft constants and mission assumptions.
* **`src/results/`**: Directory where generated Excel reports and N2 model visualizations are stored.

## 5. Academic Citations
If you utilize this framework or these models in your research, please cite the following:

```bibtex
@article{janning2025future,
  title={Future pathways for eVTOLs: A design optimization perspective},
  author={Janning, Johannes and Armanini, Sophie F. and Fasel, Urban},
  journal={arXiv preprint arXiv:2412.18078},
  year={2025}
}

@article{openmdao_2019,
  author={Justin S. Gray and John T. Hwang and Joaquim R. R. A. Martins and Kenneth T. Moore and Bret A. Naylor},
  title={OpenMDAO: An Open-Source Framework for Multidisciplinary Design, Analysis, and Optimization},
  journal={Structural and Multidisciplinary Optimization},
  year={2019},
  volume={59},
  pages={1075-1104},
  doi={10.1007/s00158-019-02211-z}
}

@software{jax2018github,
  author = {James Bradbury and Roy Frostig and Peter Hawkins and Matthew James Johnson and Chris Leary and Dougal Maclaurin and George Necula and Adam Paszke and Jake Vander{P}las and Skye Wanderman-{M}ilne and Qiao Zhang},
  title = {{JAX}: composable transformations of {P}ython+{N}um{P}y programs},
  url = {[http://github.com/jax-ml/jax](http://github.com/jax-ml/jax)},
  year = {2018}
}
