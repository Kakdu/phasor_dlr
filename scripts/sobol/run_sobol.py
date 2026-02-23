import numpy as np
import os
import argparse

# Standards for building error intervals
from phasor_dlr.config.standards import make_error_intervals
from phasor_dlr.config.defaults import condParameters

# Sobol modules
from phasor_dlr.estimation.sobol.sampling import generate_sobol_samples_from_standards
from phasor_dlr.estimation.sobol.analysis import T_A_model, build_model_params
from phasor_dlr.estimation.sobol.salib_wrappers import run_sobol_analysis

# pi-model
from phasor_dlr.models.pi_model_dynamic import pi_model_sending_phasor

# Plotting and logging
from phasor_dlr.plotting.sobol import (
    plot_sobol_first_order,
    plot_sobol_total_effect,
    plot_sobol_second_order
)
from phasor_dlr.plotting.logging import log_sobol_indices

# -------------------------------
# Parser setup
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Sobol sensitivity analysis on T_A model."
    )

    # Conductor choice
    parser.add_argument(
        "--conductor", type=str, default="bohus",
        choices=list(condParameters.keys()),
        help="Conductor type from defaults."
    )

    # CT/VT accuracy classes
    parser.add_argument("--CT_class", type=str, default="5P", help="CT accuracy class")
    parser.add_argument("--VT_class", type=str, default="0.2", help="VT accuracy class")

    # Number of Sobol samples
    parser.add_argument("--N_samples", type=int, default=15, help="log2 of number of Sobol samples")

    # Frequency and nominal temperature
    parser.add_argument("--f", type=float, default=50, help="System frequency [Hz]")
    parser.add_argument("--T_nom", type=float, default=50, help="Nominal temperature [°C]")

    # Received signals
    VrAmpDefault = 140_000/np.sqrt(3)
    parser.add_argument("--V_r_amp", type=float, default=VrAmpDefault)
    parser.add_argument("--V_r_angle", type=float, default=0.8, help="Received voltage power factor (cos phi)")
    parser.add_argument("--I_r_amp", type=float, default=None, help="Received current amplitude [A] (default = rateC of conductor)")
 

    return parser.parse_args()


# -------------------------------
# Main execution
# -------------------------------
if __name__ == "__main__":
    args = parse_args()

    # Conductor parameters
    condParam = condParameters[args.conductor]

    # If I_r_amp not specified, default to conductor rateC
    if args.I_r_amp is None:
        I_r_amp = 1 * condParam["rateC"]
    else:
        I_r_amp = args.I_r_amp

    # Generate Sobol samples
    param_values, problem, error_intervals = generate_sobol_samples_from_standards(
        CT_class=args.CT_class,
        VT_class=args.VT_class,
        N_samples=2**args.N_samples,
        calc_second_order=True
    )

    # Impedance and capacitance
    f = args.f
    T_nom = args.T_nom

    R = condParam["L"] * condParam["R_20"] * (1 + condParam["alpha"] * (T_nom - condParam["T_ref"]))
    X = condParam["L"] * condParam["X"]
    C = condParam["C"]

    w = 2 * np.pi * f
    Z = R + 1j * X
    Y = 1j * w * C

    # Received signals
    V_r_amp = args.V_r_amp
    V_r_angle = np.arccos(args.V_r_angle)  # convert PF -> angle
    I_r_angle = 0


# -------------------------------
# Sent signals
# -------------------------------
V_s_amp, V_s_angle, I_s_amp, I_s_angle = pi_model_sending_phasor(
    V_r_amp, V_r_angle, I_r_amp, I_r_angle, Z, Y
)

# -------------------------------
# Model parameters
# -------------------------------
params = build_model_params(
    condParam,
    V_s_amp, V_s_angle, I_s_amp, I_s_angle,
    V_r_amp, V_r_angle, I_r_amp, I_r_angle
)

# -------------------------------
# Model evaluation
# -------------------------------
Y = T_A_model(param_values, params, problem)

# ------------------------------- 
# Sobol analysis
# -------------------------------
Si = run_sobol_analysis(problem, Y)
S1 = Si["S1"]
S1_conf = Si["S1_conf"]
ST = Si["ST"]
ST_conf = Si["ST_conf"]
S2 = Si["S2"]

# -------------------------------
# Plots
# -------------------------------
results_folder = "results/figures/sobol"
os.makedirs(results_folder, exist_ok=True)

plot_sobol_first_order(S1, S1_conf, problem, os.path.join(results_folder,"sobol_S1.png"))
plot_sobol_total_effect(ST, ST_conf, problem, os.path.join(results_folder,"sobol_ST.png"))
plot_sobol_second_order(S2, problem, os.path.join(results_folder,"sobol_S2.png"))


# -------------------------------
# Logging
# -------------------------------
sobol_results = {
    "S1": {name: [value] for name, value in zip(problem["names"], Si["S1"])},
    "ST": {name: [value] for name, value in zip(problem["names"], Si["ST"])},
}
log_sobol_indices(sobol_results, folder="results/logs", filename_prefix="run_sobol")