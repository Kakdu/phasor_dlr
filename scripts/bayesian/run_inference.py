import pymc as pm
import argparse
import numpy as np
import multiprocessing as mp
import arviz as az
import os
import matplotlib.pyplot as plt

# Import bayesian model
from phasor_dlr.estimation.bayesian.models import build_temperature_model

# Import measurement generation
from phasor_dlr.synthetic_data.generators import generate_static_measurements
from phasor_dlr.config.standards import make_error_intervals
from phasor_dlr.config.defaults import condParameters

# Import measurement variance
from phasor_dlr.utils.math import combined_uniforms_sigma, combined_uniforms_logsigma

# Import logging
from phasor_dlr.plotting.logging import log_bayesian_run

# -------------------------------
# Parser arguments
# -------------------------------
parser = argparse.ArgumentParser(description="Run power system script.")

parser.add_argument("--f", type=float, default=50)
parser.add_argument("--T_nom", type=float, default=60)
parser.add_argument("--N", type=int, default=1)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--hdi", type=int, default=95)
parser.add_argument("--tP", type=int, default=0)
parser.add_argument("--CT_class", type=str, default="5P")
parser.add_argument("--VT_class", type=str, default="0.2")


parser.add_argument("--cond", type=str, default="bohus")
parser.add_argument("--VrAmp", type=float, default=140000.0)
parser.add_argument("--VrAngle", type=float, default=0.8)
parser.add_argument("--IrAmp", type=float, default=None)

args = parser.parse_args()

# -------------------------------
# Conductor Parameters
# -------------------------------
condParam = condParameters[args.cond]

# -------------------------------
# Measurement accuracy and variance
# -------------------------------
accuracy_r = make_error_intervals(args.CT_class, args.VT_class)
accuracy_s = make_error_intervals(args.CT_class, args.VT_class)

sigma_logV = combined_uniforms_logsigma(accuracy_r.r_V)
sigma_logI = combined_uniforms_logsigma(accuracy_r.r_I)
sigma_phiV = combined_uniforms_sigma(accuracy_r.theta_V)
sigma_phiI = combined_uniforms_sigma(accuracy_r.theta_I)


# -------------------------------
# Physical properties
# -------------------------------
f = args.f
T_nom = args.T_nom

R = condParam["L"] * condParam["R_20"] * (1 + condParam["alpha"] * (T_nom - condParam["T_ref"]))
X = condParam["L"] * condParam["X"]
C = condParam["C"]

w = 2 * np.pi * f
Z = R + 1j * X
Y = 1j * w * C


# -------------------------------
# Received signals
# -------------------------------
if args.IrAmp is None:
    I_r_amp = 1 * condParam["rateC"]
else:
    I_r_amp = args.IrAmp
V_r_amp = args.VrAmp
V_r_angle = np.arccos(args.VrAngle)  # convert PF -> angle
I_r_angle = 0


# -------------------------------
# Sent signals
# -------------------------------
V_r_amp_meas, V_r_angle_meas, I_r_amp_meas, I_r_angle_meas, V_s_amp_meas, V_s_angle_meas, I_s_amp_meas, I_s_angle_meas, V_s_amp, V_s_angle, I_s_amp, I_s_angle = generate_static_measurements(
    V_r_amp, V_r_angle,
    I_r_amp, I_r_angle,
    Z, Y,
    args.N,
    accuracy_r,  # ErrorIntervals object for receiving end
    accuracy_s,  # ErrorIntervals object for sending end
    seed=1
)


# -------------------------------
# Build model
# -------------------------------
model = build_temperature_model(
    condParameter=condParam,
    V_r_nom=V_r_amp,
    I_r_nom=I_r_amp,
    V_r_angle_nom=V_r_angle,
    I_r_angle_nom=I_r_angle,
    V_s_nom=V_s_amp,
    I_s_nom=I_s_amp,
    V_s_angle_nom=V_s_angle,
    I_s_angle_nom=I_s_angle,
    V_r_meas_array=V_r_amp_meas,
    I_r_meas_array=I_r_amp_meas,
    V_r_angle_meas_array=V_r_angle_meas,
    I_r_angle_meas_array=I_r_angle_meas,
    V_s_meas_array=V_s_amp_meas,
    I_s_meas_array=I_s_amp_meas,
    V_s_angle_meas_array=V_s_angle_meas,
    I_s_angle_meas_array=I_s_angle_meas,
    sigma_logV=sigma_logV,
    sigma_logI=sigma_logI,
    sigma_phiV=sigma_phiV,
    sigma_phiI=sigma_phiI,
    w=w,
    C=C,
    tP=0,
    phys_informed=1,
    seed=1234
)


# -------------------------------
# Enclose in function for parallel
# -------------------------------
def main():

    mp.set_start_method("fork")  # <<< force fork instead of forkserver

    with model:
        trace = pm.sample(draws=4000, tune=1000, chains=4, target_accept=0.95, random_seed=1234)


    az.summary(trace, var_names=["T_AV"])
    labeller = az.labels.MapLabeller({
        "T_AV": r"$T_{\mathrm{AV}}$"
    })

    az.plot_posterior(trace, var_names=["T_AV"], hdi_prob=args.hdi/100, labeller=labeller)

    folder = "results/figures/bayesian"
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, "temperature_posterior.png"), dpi=300)

    log_bayesian_run(trace, T_nom=args.T_nom, folder="results/logs", filename_prefix="bayes_run")

if __name__ == "__main__":
    main()

