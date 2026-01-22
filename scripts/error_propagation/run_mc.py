import argparse
import numpy as np
import os

# Import mc model
from phasor_dlr.estimation.error_propagation.mc import monte_carlo_multi

# Import physical relations
from phasor_dlr.models.pi_model_dynamic import pi_model_sending_phasor
from phasor_dlr.models.temperature import temperature_distribution

# Import conductor parameters and error intervals
from phasor_dlr.config.defaults import condParameters
from phasor_dlr.config.standards import ErrorIntervals, make_error_intervals

# Import plotting and logging modules
from phasor_dlr.plotting.uncertainty import plot_kde, plot_sent_received
from phasor_dlr.plotting.logging import log_monte_carlo_run


# -------------------------------
# Parser arguments
# -------------------------------
parser = argparse.ArgumentParser(description="Run power system script.")

parser.add_argument("--f", type=float, default=50)
parser.add_argument("--M", type=int, default=1)

parser.add_argument("--CT_class", type=str, default="5P")
parser.add_argument("--VT_class", type=str, default="0.2")
parser.add_argument("--cond", type=str, default="bohus")

parser.add_argument("--VrAmp", type=float, default=140000.0)
parser.add_argument("--VrAngle", type=float, default=0.8)
parser.add_argument("--IrAmp", type=float, default=None)
parser.add_argument("--T_nom", type=float, default=60)

parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--NSAMP", type=int, default=500_000)


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

V_r_amp_array = np.full(args.M, V_r_amp)
I_r_amp_array = np.full(args.M, I_r_amp)
V_r_angle_array = np.full(args.M, V_r_angle)
I_r_angle_array = np.full(args.M, I_r_angle)


# -------------------------------
# Sent signals
# -------------------------------
V_s_amp_array, V_s_angle_array, I_s_amp_array, I_s_angle_array = pi_model_sending_phasor(V_r_amp_array, V_r_angle_array, I_r_amp_array, I_r_angle_array, Z, Y)


# -------------------------------
# MC run
# -------------------------------
samples_r = monte_carlo_multi(
    V_r_amp_array,
    V_r_angle_array,
    I_r_amp_array,
    I_r_angle_array,
    intervals_r_I=accuracy_r.r_I,
    intervals_r_V=accuracy_r.r_V,
    intervals_theta_I=accuracy_r.theta_I,
    intervals_theta_V=accuracy_r.theta_V,
    NSAMP=args.NSAMP,
    seed=1234,
)

samples_s = monte_carlo_multi(
    V_s_amp_array,
    V_s_angle_array,
    I_s_amp_array,
    I_s_angle_array,
    intervals_r_I=accuracy_s.r_I,
    intervals_r_V=accuracy_s.r_V,
    intervals_theta_I=accuracy_s.theta_I,
    intervals_theta_V=accuracy_s.theta_V,
    NSAMP=args.NSAMP,
    seed=9876,
)


# -------------------------------
# Power loss, effective current & temperature
# -------------------------------
P_loss = samples_s["S_samples"] - samples_r["S_samples"]
I_AC = 0.5 * (samples_s["I_amp"] + samples_r["I_amp"])

T, T_max, T_avg = temperature_distribution(P_loss, I_AC, cond=condParam)


# -------------------------------
# Plotting
# -------------------------------
folder = "results/figures/error_propagation"
os.makedirs(folder, exist_ok=True)

if args.M > 1:
    filename = "distribution_T_AV_mean"
    plot_kde(
        T_avg,
        title="Distribution of mean temperature",
        xlabel="Temperature",
        style=None,
        confidence_interval=None,
        show=False,
        filename=os.path.join(folder, filename),
    )
else:
    filename = "distribution_T_AV"
    plot_kde(
        T,
        title="Distribution of conductor temperature",
        xlabel="Temperature [C]",
        style=None,
        confidence_interval=None,
        show=False,
        filename=os.path.join(folder, filename),
    )

plot_sent_received(samples_r, samples_s, folder=folder)

plot_kde(
        P_loss,
        title="Distribution of conductor temperature",
        xlabel="Temperature [C]",
        style=None,
        confidence_interval=None,
        show=False,
        filename=os.path.join(folder, "power_loss"),
    )


# -------------------------------
# Logs
# -------------------------------
log_monte_carlo_run(samples_r, samples_s, folder="results/logs", filename_prefix="error_propagation")