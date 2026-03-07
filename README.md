# eVTOL Multidisciplinary Design Optimization (MDO) Tutorial

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JohannesJanning/openmdao-evtol-tutorial/main?urlpath=lab/tree/README.md)

This repository provides an interactive tutorial for the conceptual design and optimization of a lift+cruise Electric Vertical Take-off and Landing (eVTOL) aircraft. It enables you to easily explore the conceptual trade-offs between environmental sustainability and economic viability in future urban air mobility.

---

# 1. Research Context

The models and optimization logic provided here are based on the multidisciplinary design optimization framework introduced in the following paper:

> **Janning, J., Armanini, S. F., & Fasel, U. (2025).** [Future pathways for eVTOLs: A design optimization perspective](https://doi.org/10.48550/arXiv.2412.18078). *arXiv:2412.18078 [eess.SY]*.

The framework integrates conventional aircraft design elements with operational cost models to capture stakeholder-centric objectives, such as profit modeling, cost-efficiency, and sustainability strategies.

<p align="center">
<img src="repo_images/XDSM_evtol_model.png" width="800">
</p>

**Figure 1:** [eXtended Design Structure Matrix](https://openmdao.github.io/PracticalMDO/Notebooks/ModelConstruction/understanding_xdsm_diagrams.html) of the provided design optimization problem. Figure includes single objectives of min. annual GWP (kg CO2e) and min. total operating cost (TOC) (€) per flight as example, that can be adjusted (see tutorial in Section 5).

---

# 2. Technical Framework

The project utilizes the following computational stack:

- **[OpenMDAO](https://github.com/OpenMDAO/OpenMDAO)**: An open-source framework for multidisciplinary analysis and optimization. It manages the coupling between multiple analysis blocks, like aerodynamics, mass estimation, energy requirements, and cost models.

- **[JAX](https://github.com/jax-ml/jax)**: A library for composable transformations of Python and NumPy programs. It enables efficient gradient-based optimization through automatic differentiation.

---

# 3. Getting Started

## Accessing the Workspace

Click the Binder badge at the top of this page to launch a cloud-resident JupyterLab session. All necessary dependencies are pre-installed in the container environment.

## Optimization Workflows

The tutorial consists of one primary notebook: **01_optimize_GWP.ipynb**
The notebook **02_optimize_TOC.ipynb** is identical and can be used as a secondary comparison notebook.
To execute a study, open a notebook and select: "Run > Run All Cells" from the top menu.

---

# 4. Repository Structure

- **01_optimize_GWP.ipynb & 02_optimize_TOC.ipynb**  
  Primary interactive tutorial notebooks (as the objective function is adjustable, either notebook can be used for the tasks in Section 5).

- **src/**  
  The core source code directory.

  - **models_jax/**  
    Physics and cost components implemented in JAX.

  - **analysis/**  
    Post-optimization processing and evaluation scripts.

  - **Components/**  
    components defining the OpenMDAO model.

  - **parameters.py**  
    Central configuration for aircraft constants and mission assumptions.

- **src/results/**  
  Directory where generated Excel reports are stored.

---

# 5. Interactive Optimization Tutorials

Follow these tasks to explore the multidisciplinary coupling of the eVTOL framework.

## Notes

You may find helpful:

**GWP_annual_ops Optimal design vector**

```
[15.0, 1.7179846040279754, 1.5684475976845111, 1.5667718019064831, 400.0, 1.0]
```

**TOC_flight Optimal design vector**

```
[9.874446571073564, 1.0, 0.8561370588805344, 1.394907761845594, 399.99999999999994, 2.0253538731700154]
```

---

# Task 1: Design Trade-offs

Goal: Understand how the selection of an objective function dictates the physical architecture.

## 1.1 Comparing Objectives

1. Ensure the notebook is in its initial state.
2. Cell 2.3: Set objective to minimize **GWP_flight** (set ref: 20).
3. Run the optimization.
4. Cell 5: Copy the resulting **Optimal Design Vector** from Cell 4 into the **baseline_vector** variable.
5. Cell 2.3: Change objective to minimize **TOC_flight** (set ref: 100).
6. Run the optimization and compare results in the Dashboard.

*Observe how these two design variables sets introduce tradeoffs in environmental impact and operational costs.*

---

## 1.2 Environmental Optimization

1. Ensure the notebook is in its initial state.
2. Cell 2.3: Set objective to minimize **GWP_flight** (set ref: 20).
3. Run the optimization.
4. Cell 5: Copy the resulting Optimal Design Vector from Cell 4 into the **baseline_vector**.
5. Cell 2.3: Change objective to minimize **GWP_annual_ops** (set ref: 50000).
6. Run the optimization and compare results in the Dashboard.

*Notice how optimizing for a single flight vs. a whole year shifts the design.*

---

## 1.3 Economic Optimization

1. Reset to initial settings.
2. Cell 2.3: Set objective to minimize **TOC_flight** (set ref: 100).
3. Run the optimization.
4. Cell 5: Copy the Optimal Design Vector into the **baseline_vector**.
5. Cell 2.3: Set objective to maximize **Annual_Profit**.

Note: Use **ref = -1000000** (The negative sign enables maximization in OpenMDAO's minimization-based solver).

6. Run and compare.

*Identify the utilization trade-off. Does a more profitable plane fly differently than a cheaper-to-operate one?*

---

# Task 2: Shifting the bounds

Goal: Observe how design variable limitations impact the design space.

1. Run a baseline **TOC_flight** optimization (as described in task 1.1).
2. Cell 5: Copy the vector into **baseline_vector**.
3. Cell 2.1 (Design Variables): Change the upper bound of the battery energy density (**rho_bat**) from **400** to **300**.
4. Run the optimization.

*Observation: How does our battery mass change, how are our operating costs impacted?*

---

# Task 3: Hitting constraints

Goal: Experience the mathematical struggle of highly-constrained design.

1. Run a baseline **GWP_annual** optimization and save the result to **baseline_vector** (as described in task 1.1).
2. Cell 2.2 (Constraints): Change the maximum **MTOM constraint** from **5700 (EASA SC-VTOL limit)** to **1500**.
3. Run the optimization.

Note: This may take up to **40 seconds**. The optimizer is navigating a much "narrower" feasible region.

4. Analyze: Can the model still find a design? If yes, is the resulting design valid?

---

# Task 4: Manned vs. Autonomous Operations

Goal: Quantify the secondary benefits of pilotless flight systems.

1. Cell 2.3: Set **TOC_flight** as the objective and run a baseline optimization.
2. Cell 5: Copy the design vector into **baseline_vector**.
3. parameters.py: Open the file in the sidebar and modify:

m_crew: Change **96.5** to **20.0** (simulates sensor suite replacing a cockpit).  
N_ac: Change **1** to **3** (simulates 1 ground-pilot supervising 3 aircraft).

4. Run the optimization.

Observation: Compare the **Mass** and **Cost** sections.

5. Further Stress Test: In **parameters.py**, increase **m_pay (payload)** from **392.8** to **600**. Run again and check the impact on mass and cost structure.

---

⚠️ Solver Note:  
If an optimization takes more than **60 seconds** or fails to converge, you may have created a "physically impossible" aircraft (e.g., too much weight for too little battery). Try loosening your constraints or increasing your design variable bounds.

---

# 6. Academic Citations

If you use this framework, these models, or methods in your research, please cite the following as applicable:

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


---

# 8. Further Reading & References

If you want to dive deeper into the theory of **Multidisciplinary Design Optimization (MDO)** and **Urban Air Mobility (UAM)**, the following resources may be helpful:

- **Martins, J. R. R. A., & Lambe, A. B. (2013)**  
  *Multidisciplinary Design Optimization: A Survey of Architectures*  
  A foundational paper explaining how complex engineering systems and MDO architectures (such as XDSM) are structured.  
  https://arc.aiaa.org/doi/10.2514/1.J051895

- **Martins, J. R. R. A., & Ning, A. (2021)**  
  *Engineering Design Optimization*  
  A comprehensive graduate-level textbook covering optimization algorithms, derivative computation, and multidisciplinary design optimization methods used in modern engineering.  
  https://www.cambridge.org/core/product/identifier/9781108980647/type/book

- **Sengupta, R., Bulusu, V., et al. (2025)**  
  *Urban Air Mobility Research: Challenges and Opportunities*  
  A modern review of operational, technological, and autonomy challenges shaping the future of the eVTOL and UAM ecosystem.  
  https://www.annualreviews.org/content/journals/10.1146/annurev-control-022823-031353

---

## 7. License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.