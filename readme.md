# phasor_dlr

**Phasor-DLR** is a Python package for electrical phasor modeling and analysis, providing tools for:

- **Error propagation** in electrical systems  
- **Sobol sensitivity analysis**  
- **Bayesian inference** for parameter estimation  
- **Unscented Kalman Filter (UKF)** for dynamic state estimation  

This package is structured for reproducible simulations, plotting, and analysis of electrical phasor models.

---

## Project Structure

phasor_dlr/
├─ pyproject.toml
├─ results/ # Simulation results and figures
│ ├─ figures/
│ │ ├─ bayesian/
│ │ ├─ error_propagation/
│ │ ├─ sobol/
│ │ └─ ukf/
│ └─ logs/
├─ scripts/ # Scripts to run simulations
│ ├─ bayesian/
│ ├─ error_propagation/
│ ├─ sobol/
│ └─ ukf/
└─ src/
└─ phasor_dlr/
├─ config/
├─ estimation/
│ ├─ bayesian/
│ ├─ error_propagation/
│ ├─ sobol/
│ └─ ukf/
├─ models/
├─ plotting/
├─ synthetic_data/
└─ utils/


---

## Installation

Clone the repository and install with `pip`:

bash
git clone <repo_url>
cd phasor_dlr
pip install -e 

This will install the package in editable mode.

## Usage

#### Running Scripts

The package contains standalone scripts for each type of analysis:

Bayesian inference
python scripts/bayesian/run_inference.py
python scripts/bayesian/compare_priors.py

Error propagation / Monte Carlo
python scripts/error_propagation/run_mc.py
python scripts/error_propagation/sweep_mc.py

Sobol sensitivity analysis
python scripts/sobol/run_sobol.py
python scripts/sobol/sweep_sobol.py

Unscented Kalman Filter (UKF)
python scripts/ukf/run_filter.py

### Using as a Python package

You can also import modules directly:

from phasor_dlr.estimation.ukf import filtering, ukf_core, sigma_points
from phasor_dlr.estimation.bayesian import models, priors, likelihoods
from phasor_dlr.models import phasors, pi_model_dynamic, temperature
from phasor_dlr.plotting import ukf as ukf_plot

### Features

1. Error Propagation

Monte Carlo simulations

Systematic error modeling

Generates voltage, current, and phase difference distributions

2. Sobol Sensitivity Analysis

Computes first-order, second-order, and total Sobol indices

Evaluates sensitivity of temperature and electrical phasor parameters

3. Bayesian Inference

Supports prior specification and likelihood modeling

Posterior estimation with visualization of HDI and RMSE trends

4. Unscented Kalman Filter (UKF)

Full UKF implementation for phasor state estimation

Plots for estimated states, Kalman gain, and smoothed temperature


## Results and Plots

All simulation results are saved in results/figures/ and results/logs/.
Folders are organized by method:

bayesian/ → Posterior distributions and trend plots

error_propagation/ → Distributions, amplitudes, variances, and phase differences

sobol/ → Sensitivity analysis plots (S1, S2, ST)

ukf/ → UKF state estimates, temperature, and Kalman gain plots
