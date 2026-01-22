# phasor_dlr

**Phasor-DLR** is a Python package for electrical phasor modelling and analysis, providing tools for:

- **Error propagation** in of measurement errors to conductor temperature estimation 
- **Sobol sensitivity analysis** to show main sources of error
- **Bayesian inference** for static state estimation  
- **Unscented Kalman Filter (UKF)** for dynamic state estimation  

This package is structured for reproducible simulations, plotting, and analysis of the electrical phasor models.


---

## Usage

#### Running Scripts

The package contains standalone scripts for each type of analysis:

1. Error propagation / Monte Carlo

PYTHONPATH=src python scripts/error_propagation/run_mc.py

PYTHONPATH=src python scripts/error_propagation/sweep_mc.py


2. Sobol sensitivity analysis

PYTHONPATH=src python scripts/sobol/run_sobol.py

PYTHONPATH=src python scripts/sobol/sweep_sobol.py


3. Bayesian inference

PYTHONPATH=src python scripts/bayesian/run_inference.py

PYTHONPATH=src python scripts/bayesian/compare_priors.py


4. Unscented Kalman Filter (UKF)

PYTHONPATH=src python scripts/ukf/run_filter.py


### Features

1. Error Propagation

Monte Carlo simulations

Systematic error modelling

Generates voltage, current, phase difference, average real power, power loss, and temperature distributions

A sweep over current to show temperature variance for different conductor models over their expected operating currents.

2. Sobol Sensitivity Analysis

Computes first-order, second-order, and total Sobol indices

Evaluates sensitivity of temperature and electrical phasor parameters

A sweep over the power factor to show the major contribution to error at different values.

3. Bayesian Inference

Supports prior specification and likelihood modelling

Posterior estimation with visualization of HDI and RMSE trends over number of measurements

4. Unscented Kalman Filter (UKF)

Full UKF implementation for phasor state estimation

Plots for estimated states, Kalman gain, and smoothed temperature

---

## Results and Plots

All simulation results are saved in results/figures/ and results/logs/.
Folders are organized by method:

bayesian/ --> Posterior distributions and trend plots

error_propagation/ --> Distributions, amplitudes, variances, and phase differences

sobol/ --> Sensitivity analysis plots (S1, S2, ST)

ukf/ --> UKF state estimates, temperature, and Kalman gain plots

