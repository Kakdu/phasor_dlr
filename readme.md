# phasor_dlr

**Phasor-DLR** is a Python package for electrical phasor modeling and analysis, providing tools for:

- **Error propagation** in electrical systems  
- **Sobol sensitivity analysis**  
- **Bayesian inference** for parameter estimation  
- **Unscented Kalman Filter (UKF)** for dynamic state estimation  

This package is structured for reproducible simulations, plotting, and analysis of electrical phasor models.


---

## Usage

#### Running Scripts

The package contains standalone scripts for each type of analysis:

Bayesian inference
PYTHONPATH=src python scripts/bayesian/run_inference.py
PYTHONPATH=src python scripts/bayesian/compare_priors.py

Error propagation / Monte Carlo
PYTHONPATH=src python scripts/error_propagation/run_mc.py
PYTHONPATH=src python scripts/error_propagation/sweep_mc.py

Sobol sensitivity analysis
PYTHONPATH=src python scripts/sobol/run_sobol.py
PYTHONPATH=src python scripts/sobol/sweep_sobol.py

Unscented Kalman Filter (UKF)
PYTHONPATH=src python scripts/ukf/run_filter.py

### Features

1. Error Propagation

Monte Carlo simulations

Systematic error modeling

Generates voltage, current, phase difference, average real power, power loss, and temperature distributions

A sweep over current to show temperature variance for different conductor models over their expected operating currents.

2. Sobol Sensitivity Analysis

Computes first-order, second-order, and total Sobol indices

Evaluates sensitivity of temperature and electrical phasor parameters

A sweep over the power factor to show the major contribution to error at different values.

3. Bayesian Inference

Supports prior specification and likelihood modeling

Posterior estimation with visualization of HDI and RMSE trends over number of measurements

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
