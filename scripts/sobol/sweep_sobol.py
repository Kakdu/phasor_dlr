# scripts/sobol/run_sobol_sweep.py
import numpy as np
import argparse
from SALib.analyze import sobol
import os

from phasor_dlr.estimation.sobol.sampling import generate_sobol_samples_from_standards
from phasor_dlr.estimation.sobol.analysis import T_A_model, build_model_params
from phasor_dlr.plotting.sobol import plot_sobol_sweep

from phasor_dlr.models.pi_model_dynamic import pi_model_sending_phasor
from phasor_dlr.config.defaults import condParameters

parser = argparse.ArgumentParser(description="Sobol sensitivity sweep vs voltage angle")
parser.add_argument("--cond_name", default="bohus", type=str)
parser.add_argument("--CT_class", default="5P", type=str)
parser.add_argument("--VT_class", default="0.2", type=str)
parser.add_argument("--N_samples", default=2**16, type=int)
parser.add_argument("--f", default=50, type=float, help="Frequency in Hz")
parser.add_argument("--T_nom", default=50, type=float, help="Nominal conductor temperature in C")
parser.add_argument("--V_r_amp", default=140_000, type=float, help="Received voltage amplitude in V")
parser.add_argument("--V_r_pf_start", default=0.9, type=float, help="Start power factor for sweep")
parser.add_argument("--V_r_pf_end", default=0.6, type=float, help="End power factor for sweep")
parser.add_argument("--n_angles", default=20, type=int, help="Number of points in sweep")

args = parser.parse_args()


# -------------------------------
# Conductor parameters
# -------------------------------
condParam = condParameters[args.cond_name]


# -------------------------------
# Generate Sobol samples
# -------------------------------
param_values, problem, _ = generate_sobol_samples_from_standards(
    CT_class=args.CT_class,
    VT_class=args.VT_class,
    N_samples=args.N_samples,
    calc_second_order=False
)

# -------------------------------
# Impedance and capacitance
# -------------------------------
R = condParam["L"] * condParam["R_20"] * (1 + condParam["alpha"] * (args.T_nom - condParam["T_ref"]))
X = condParam["L"] * condParam["X"]
C = condParam["C"]
w = 2 * np.pi * args.f
Z = R + 1j * X
Y_shunt = 1j * w * C


# -------------------------------
# Received signals
# -------------------------------
I_r_amp = 1 * condParam["rateC"]
I_r_angle = 0.0
V_r_amp = args.V_r_amp

# -------------------------------
# Angle sweep
# -------------------------------
V_r_angles = np.arccos(np.linspace(args.V_r_pf_start, args.V_r_pf_end, args.n_angles))
cos_V_r_angles = np.cos(V_r_angles)

# -------------------------------
# Allocate arrays
# -------------------------------
S1_r_sum = np.zeros(args.n_angles)
S1_theta_sum = np.zeros(args.n_angles)
S1_r_conf = np.zeros(args.n_angles)
S1_theta_conf = np.zeros(args.n_angles)
S1_inter_sum = np.zeros(args.n_angles)
S1_inter_conf = np.zeros(args.n_angles)
# -------------------------------



for k, V_r_angle in enumerate(V_r_angles):
    # Calculate sent phasor signals
    V_s_amp, V_s_angle, I_s_amp, I_s_angle = pi_model_sending_phasor(
        V_r_amp, V_r_angle, I_r_amp, I_r_angle, Z, Y_shunt
    )

    # Build model parameters
    params = build_model_params(
        condParam,
        V_s_amp, V_s_angle, I_s_amp, I_s_angle,
        V_r_amp, V_r_angle, I_r_amp, I_r_angle
    )

    # Evaluate T_A
    Y_model = T_A_model(param_values, params, problem)

    # Sobol first-order analysis
    Si = sobol.analyze(problem, Y_model, print_to_console=False, calc_second_order=False)
    S1 = np.clip(Si["S1"], 0, None)
    S1_conf = Si["S1_conf"]

    # Sum r and theta contributions
    S1_r_sum[k] = np.sum(S1[0:14])
    S1_theta_sum[k] = np.sum(S1[14:28])
    S1_r_conf[k] = np.sqrt(np.sum(S1_conf[0:14]**2))
    S1_theta_conf[k] = np.sqrt(np.sum(S1_conf[14:28]**2))

    # Interaction term
    S1_inter_sum[k] = 1.0 - S1_r_sum[k] - S1_theta_sum[k]
    S1_inter_conf[k] = np.sqrt(S1_r_conf[k]**2 + S1_theta_conf[k]**2)

    print(f"Angle {k+1}/{args.n_angles} complete.")


# -------------------------------
# Plot sweep results
# -------------------------------
results_folder = "results/figures/sobol"
os.makedirs(results_folder, exist_ok=True)

plot_sobol_sweep(
    cos_angles=cos_V_r_angles,
    S1_r_sum=S1_r_sum,
    S1_r_conf=S1_r_conf,
    S1_theta_sum=S1_theta_sum,
    S1_theta_conf=S1_theta_conf,
    S1_inter_sum=S1_inter_sum,
    S1_inter_conf=S1_inter_conf,
    save_path= os.path.join(results_folder,"sobol_S1_sum_vs_cos_angle.png")
)