import pymc as pm
import argparse
import numpy as np
import multiprocessing as mp
import arviz as az
import os
import matplotlib.pyplot as plt
import multiprocessing as mp

# Import bayesian model
from phasor_dlr.estimation.bayesian.models import run_model_for_N

# Import measurement generation
from phasor_dlr.config.standards import make_error_intervals
from phasor_dlr.config.defaults import condParameters

# Import measurement variance
from phasor_dlr.utils.math import combined_uniforms_sigma, combined_uniforms_logsigma

# Import plot functions
from phasor_dlr.plotting.bayesian import plot_rmse_vs_N, plot_hdi_vs_N

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

mp.set_start_method("fork")  # <<< force fork instead of forkserver

# -------------------------------
# Conductor Parameters
# -------------------------------
condParam = condParameters[args.cond]


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
# Iteration variables
# -------------------------------
Nlist = [1, 5, 20, 50, 90, 150, 200, 500, 900, 1200, 2400, 3000, 5000, 10000, 20000, 50000, 60000]
levels = [0.90, 0.95, 0.99]
prior_choice = [True, False]
keys = ["0.2"]
results = {}
# -------------------------------
# Loops
# -------------------------------
for idx, prior in enumerate(prior_choice):
    results[prior] = {}
    for key in keys:
        results[prior][key] = {}
        accuracy_r = make_error_intervals(key, "0.2")
        accuracy_s = make_error_intervals(key, "0.2")

        sigma_logV = combined_uniforms_logsigma(accuracy_r.r_V)
        sigma_logI = combined_uniforms_logsigma(accuracy_r.r_I)
        sigma_phiV = combined_uniforms_sigma(accuracy_r.theta_V)
        sigma_phiI = combined_uniforms_sigma(accuracy_r.theta_I)
        for N in Nlist:
            print(f"Running N = {N}")
            out = run_model_for_N(
                N=N,
                seed=args.seed,
                condParam=condParam,
                V_r_amp=V_r_amp,
                V_r_angle=V_r_angle,
                I_r_amp=I_r_amp,
                I_r_angle=I_r_angle,
                Z=Z,
                Y=Y,
                accuracy_r=accuracy_r,
                accuracy_s=accuracy_s,
                sigma_logV=sigma_logV,
                sigma_logI=sigma_logI,
                sigma_phiV=sigma_phiV,
                sigma_phiI=sigma_phiI,
                w=w,
                C=C,
                T_nom=T_nom,
                phys_informed=prior,
                levels=levels
            )

            results[prior][key][N] = out

            # Early stopping
            if out["hdi"][levels[-1]] < 5:
                break


folder = "results/figures/bayesian"
os.makedirs(folder, exist_ok=True)

plot_rmse_vs_N(
    results=results,
    levels=levels,
    keys=keys,
    Nlist=Nlist,
    folder=folder,
    suffix="new_model",
)

plot_hdi_vs_N(
    results=results,
    levels=levels,
    keys=keys,
    Nlist=Nlist,
    folder=folder,
    suffix="new_model",
)

