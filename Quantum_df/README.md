Quantum_df — Calculating Pi on Quantum Computing
===============================================

Name: poig
Date: 2025-October-20

Introduction
------------

This project contains notebooks and helper code used for researching quantum approaches to numerical integration and related experiments. The primary notebook is `calculate_pi_fibonnaci.ipynb` and helper utilities are in `helper.py`.

Article
-------

Companion Medium article: [Calculating Pi on Quantum Computing — A Novel Integration-Based Approach](https://medium.com/@poig/calculating-pi-on-quantum-computing-a-novel-integration-based-approach-125daea064b2)

Quick setup from scratch (Conda)
---------------------------------------

1. Create a conda environment

```powershell
conda create -n quantum_df python=3.11 -y
conda activate quantum_df
```

2. Install dependencies

Option A — pip (easy):

```powershell
pip install --upgrade pip
pip install -r requirement.txt
```

Option B — hybrid conda + pip (recommended for binary compatibility):

```powershell
conda install -y numpy scipy scikit-learn matplotlib pandas plotly jupyter
pip install qiskit
pip install -r requirement.txt
```

3. Register kernel (optional, recommended)

```powershell
pip install ipykernel
python -m ipykernel install --user --name quantum_df --display-name "Python (quantum_df)"
```

4. Launch Jupyter (optional)

```powershell
jupyter notebook
```

Files
-----

- `calculate_pi_fibonnaci.ipynb` — main notebook
- `helper.py` — helper functions used by the notebook
- `requirement.txt` — Python packages used by the notebooks

Notes
-----

- If you experience pip wheel/build failures (common with some packages), try installing from `conda-forge` (e.g., `conda install -c conda-forge qiskit`).

License
-------

See the repo-level `LICENSE` file.

Permissions & license
---------------------

This project is the author's self-research. There are no external contributors. Any reuse, redistribution, or republication of the project's notebooks, code, or narrative requires explicit permission from the author (poig).

If you wish to cite or reuse parts of this work, please cite the companion Medium article and request permission via the contact in github profile.