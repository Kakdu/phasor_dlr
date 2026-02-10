import pymc as pm
import argparse
import numpy as np
import multiprocessing as mp
import arviz as az
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import skewnorm

# Import bayesian model
from phasor_dlr.estimation.bayesian.models import build_temperature_model

# Import measurement generation
from phasor_dlr.synthetic_data.generators import generate_static_measurements
from phasor_dlr.config.standards import make_error_intervals
from phasor_dlr.config.defaults import condParameters

# Import measurement variance
from phasor_dlr.utils.math import combined_uniforms_sigma, combined_uniforms_logsigma

from phasor_dlr.plotting.logging import log_bayesian_runs

# -------------------------------
# Parser arguments (same as before)
# -------------------------------
parser = argparse.ArgumentParser(description="Run 3x Bayesian temperature inference.")

parser.add_argument("--f", type=float, default=50)
parser.add_argument("--T_nom", type=float, default=50)
parser.add_argument("--N", type=int, default=1)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--hdi", type=int, default=95)

parser.add_argument("--CT_class", type=str, default="5P")
parser.add_argument("--VT_class", type=str, default="0.2")

parser.add_argument("--cond", type=str, default="bohus")
parser.add_argument("--VrAmp", type=float, default=140000.0)
parser.add_argument("--VrAngle", type=float, default=0.8)
parser.add_argument("--IrAmp", type=float, default=None)

parser.add_argument("--eta", type=float, default=1)

args = parser.parse_args()


# -------------------------------
# Conductor Parameters
# -------------------------------
condParam = condParameters[args.cond]

# -------------------------------
# Measurement accuracy and variance
# -------------------------------
accuracy_r = make_error_intervals(args.CT_class, args.VT_class, args.eta)
accuracy_s = make_error_intervals(args.CT_class, args.VT_class, args.eta)

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
V_r_angle = np.arccos(args.VrAngle)
I_r_angle = 0


# -------------------------------
# Generate measurements
# -------------------------------
(
    V_r_amp_meas,
    V_r_angle_meas,
    I_r_amp_meas,
    I_r_angle_meas,
    V_s_amp_meas,
    V_s_angle_meas,
    I_s_amp_meas,
    I_s_angle_meas,
    V_s_amp,
    V_s_angle,
    I_s_amp,
    I_s_angle,
) = generate_static_measurements(
    V_r_amp,
    V_r_angle,
    I_r_amp,
    I_r_angle,
    Z,
    Y,
    args.N,
    accuracy_r,
    accuracy_s,
    seed_r=1,
    seed_s=2,
)


# -------------------------------
# Inference runner
# -------------------------------
def run_inference(tP, phys_informed):

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
        tP=tP,
        phys_informed=phys_informed,
        seed=1234,
    )

    with model:
        trace = pm.sample(
            draws=4000,
            tune=1000,
            chains=4,
            target_accept=0.95,
            random_seed=1234,
        )

    return trace


# -------------------------------
# Main
# -------------------------------
def main():

    mp.set_start_method("fork")

    configs = [
        (0, 0),
        (0, 1),
        (1, 0),
    ]

    traces = []
    labels = []

    for tP, phys in configs:
        print(f"Running inference: tP={tP}, phys_informed={phys}")
        trace = run_inference(tP, phys)
        traces.append(trace)
        labels.append(f"tP={tP}, phys={phys}")

    # -------------------------------
    # Global x-axis limits
    # -------------------------------
    all_samples = np.concatenate(
        [t.posterior["T_AV"].values.flatten() for t in traces]
    )

    x_min = np.min(all_samples)
    x_max = np.max(all_samples)


    # -------------------------------
    # Plotting
    # -------------------------------
    folder = "results/figures/bayesian"
    os.makedirs(folder, exist_ok=True)

    for trace, label, (tP, phys) in zip(traces, labels, configs):

        fig, ax = plt.subplots(figsize=(6, 4))

        az.plot_posterior(
            trace,
            var_names=["T_AV"],
            hdi_prob=args.hdi / 100,
            point_estimate=None,
            ax=ax,
        )

        # Remove ArviZ auto text
        for txt in ax.texts:
            txt.set_visible(False)

        posterior_line = ax.lines[0]
        posterior_line.set_label(r"$T_{\mathrm{AV}}$ posterior")

        # Optional prior overlay
        if args.N < 200 and phys == 1:

            x = np.linspace(x_min, x_max, 500)

            mu = 70
            sigma = 10.0
            alpha = -5

            pdf = skewnorm.pdf(x, a=alpha, loc=mu, scale=sigma)

            prior_line = ax.plot(
                x,
                pdf,
                color="black",
                linestyle="--",
                linewidth=0.8,
                label=r"$T_{\mathrm{EST}}$ prior",
            )[0]

        # Force identical x-axis
        ax.set_xlim(x_min, x_max)

        ax.set_xlabel("Degrees [$^\\circ$C]")
        ax.set_ylabel(None)
        ax.set_title("")

        hdi_handle = Line2D(
            [0],
            [0],
            color="black",
            linewidth=4,
            label=f"{args.hdi:.0f}% HDI",
        )

        handles = [posterior_line, hdi_handle]
        if args.N < 200 and phys == 1:
            handles.append(prior_line)

        ax.legend(handles=handles)

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                folder,
                f"temperature_posterior_tP{tP}_phys{phys}.png",
            ),
            dpi=300,
        )
        plt.close()
    return traces



if __name__ == "__main__":
    traces = main()

labels = [
    "tP=0, phys_informed=0",
    "tP=0, phys_informed=1",
    "tP=1, phys_informed=0",
]

log_bayesian_runs(
    traces,
    labels=labels,
    T_nom=args.T_nom,
)