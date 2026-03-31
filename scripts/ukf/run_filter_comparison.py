import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "legend.fontsize": 14
})
from matplotlib.lines import Line2D

import os
from scipy.stats import norm
from dataclasses import dataclass

import argparse

# Import math functions
from phasor_dlr.utils.math import combined_uniforms_sigma, combined_uniforms_logsigma

# Import conductor parameters and error intervals
from phasor_dlr.config.defaults import condParameters
from phasor_dlr.config.standards import ErrorIntervals, make_error_intervals

# Import noise sampling
from phasor_dlr.synthetic_data.noise import apply_error_to_phasors


DEFAULTS = {
    "min": 30,
    "seed": 0,
    "cond": "bohus",
    "conf": 95,
    "stand": "0.2"
}

parser = argparse.ArgumentParser(description="Run power system script.")

parser.add_argument("--min", type=int, default=DEFAULTS["min"])
parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
parser.add_argument("--conf", type=int, default=DEFAULTS["conf"])
parser.add_argument("--CT_class", type=str, default="0.2")
parser.add_argument("--VT_class", type=str, default="0.2")
parser.add_argument("--cond", type=str, default=DEFAULTS["cond"])

args = parser.parse_args()


# --------------------------
# π-model sending phasor function (returns amps/angles)
# --------------------------
def pi_model_sending_phasor(V_r_amp, V_r_angle, I_r_amp, I_r_angle, T, condParameter):
    R = (
        condParameter["L"]
        * condParameter["R_20"]
        * (1 + condParameter["alpha"] * (T - condParameter["T_ref"]))
    )
    X = condParameter["L"] * condParameter["X"]
    C = condParameter["C"]
    w = 2 * np.pi * 50
    Z = R + 1j * X
    Y = 1j * w * C
    V_r = V_r_amp * np.exp(1j * V_r_angle)
    I_r = I_r_amp * np.exp(1j * I_r_angle)
    I_shunt_r = V_r * (Y/2)
    I_series = I_r + I_shunt_r
    V_s = V_r + I_series * Z
    I_shunt_s = V_s * (Y/2)
    I_s_total = I_series + I_shunt_s
    V_s_amp, V_s_angle = np.abs(V_s), np.angle(V_s)
    I_s_amp, I_s_angle = np.abs(I_s_total), np.angle(I_s_total)
    return V_s_amp, V_s_angle, I_s_amp, I_s_angle


# --------------------------
# Temperature from measurements
# --------------------------
def T_from_measurements(V_r_amp, I_r_amp, V_r_angle, I_r_angle,
                        V_s_amp, I_s_amp, V_s_angle, I_s_angle, condParameter):
    I_AC = np.sqrt(0.5 * (I_s_amp**2 + I_r_amp**2))
    return condParameter["T_ref"] + 1 / condParameter["alpha"] * (
        (V_s_amp * I_s_amp * np.cos(V_s_angle - I_r_angle) - V_r_amp * I_r_amp * np.cos(V_r_angle - I_r_angle))
        / (0.5 * (I_s_amp**2 + I_r_amp**2) * condParameter["R_20"] * condParameter["L"]) - 1
    )


# --------------------------
# Sigmoid for temperature
# --------------------------
def turn_on_exp(x, x0=0.0, tau=1.0, A=1.0):
    return A * (1 - np.exp(-(x - x0) / tau)) * (x > x0)


# --------------------------
# Conductor parameters
# --------------------------
condParameter = condParameters[args.cond]


# -------------------------------
# Measurement accuracy and variance
# -------------------------------
accuracy_r = make_error_intervals(args.CT_class, args.VT_class)
accuracy_s = make_error_intervals(args.CT_class, args.VT_class)

sigma_V = combined_uniforms_logsigma(accuracy_r.r_V)
sigma_I = combined_uniforms_logsigma(accuracy_r.r_I)
sigma_phiV = combined_uniforms_sigma(accuracy_r.theta_V)
sigma_phiI = combined_uniforms_sigma(accuracy_r.theta_I)



# --------------------------
# UKF parameters with log amplitudes
# --------------------------
minutes = args.min
steps_psec = 100
steps = minutes * steps_psec * 60

pre_event = 4 * 60 * steps_psec
post_event = 36 * 60 * steps_psec

startConv = minutes * steps_psec * 60 - post_event
startCurr = minutes * steps_psec * 60 - post_event

n = 5  # [T, log(V_r_amp), log(I_r_amp), received_angle_diff, sent_angle_diff]
m = 6  # [log(V_r_amp), log(I_r_amp), log(V_s_amp), log(I_s_amp), sent_angle_diff, received_angle_diff]

# Chi-square threshold for 90% confidence (n_z = 6)
eta_th = 10.6  # for 90% confidence, use 10.6

# Scaling factor
gamma = 10
alpha, beta, kappa = 1e-3, 2, 0



def run_scenario(mode, direction=None):
    label = f"{mode}" if direction is None else f"{mode}-{direction}"
    # --------------------------
    # Synthetic true states
    # --------------------------
    x = np.linspace(1, steps, steps)

    T_true = 55 + 2*np.sin(np.linspace(0,5*minutes/100,steps))
    V_r_true_amp = 140_000/np.sqrt(3)
    V_r_true_angle = 0.0
    I_r_true_amp = 800 + 25*np.sin(np.linspace(0,4*minutes/100,steps))
    I_r_true_angle = - np.arccos(0.8) + 0.05*np.sin(np.linspace(0,4*minutes/100,steps))


    if mode == "conv":
        tauConv = steps_psec * 60 * 8
        if direction == "up":
            T_true += turn_on_exp(x, startConv, tau=tauConv, A=10)
        if direction == "dn":
            T_true -= turn_on_exp(x, startConv, tau=tauConv, A=10)

    if mode == "curr":
        tauCurr = steps_psec * 60 * 13
        if direction == "up":
            T_true += turn_on_exp(x, startCurr, tau=tauCurr, A=15)
            I_r_true_amp += 400 * np.concatenate([np.zeros(startCurr), np.ones(steps - startCurr)])
        if direction == "dn":
            T_true -= turn_on_exp(x, startCurr, tau=tauCurr, A=12)
            I_r_true_amp -= 300 * np.concatenate([np.zeros(startCurr), np.ones(steps - startCurr)])


    # --------------------------
    # Sending-end phasors
    # --------------------------
    V_s_true_amp, V_s_true_angle, I_s_true_amp, I_s_true_angle = pi_model_sending_phasor(
        V_r_true_amp, V_r_true_angle, I_r_true_amp, I_r_true_angle, T_true, condParameter
    )

    # --------------------------
    # Measurements
    # --------------------------
    np.random.seed(args.seed)
    V_r_meas, V_r_angle_meas, I_r_meas, I_r_angle_meas = apply_error_to_phasors(
        V_r_true_amp, V_r_true_angle, I_r_true_amp, I_r_true_angle,
        accuracy_r, steps, seed=args.seed + 1
    )

    V_s_meas, V_s_angle_meas, I_s_meas, I_s_angle_meas = apply_error_to_phasors(
        V_s_true_amp, V_s_true_angle, I_s_true_amp, I_s_true_angle,
        accuracy_s, steps, seed=args.seed + 2
    )

    sent_angle_meas = V_s_angle_meas - I_s_angle_meas
    received_angle_meas = V_r_angle_meas - I_r_angle_meas

    # --------------------------
    # Process noise Q (linear units for T, additive in log-space for amplitudes)
    # --------------------------
    Q = np.diag([0.1, sigma_V**2, sigma_I**2, sigma_phiV**2 + sigma_phiI**2, sigma_phiV**2 + sigma_phiI**2])
    Q = Q / 500

    # Measurement noise R simplified
    R = np.diag([
        sigma_V**2, sigma_I**2, 
        sigma_V**2, sigma_I**2,
        sigma_phiV**2 + sigma_phiI**2, 
        sigma_phiV**2 + sigma_phiI**2])

    lambda_ = alpha**2 * (n + kappa) - n
    Wm = np.full(2 * n + 1, 0.5 / (n + lambda_))
    Wc = Wm.copy()
    Wm[0] = lambda_ / (n + lambda_)
    Wc[0] = Wm[0] + (1 - alpha**2 + beta)


    # --------------------------
    # UKF initial state
    # --------------------------
    x_est = np.array([
        55.0,                     # T
        np.log(V_r_true_amp),  # log V_r_amp
        np.log(I_r_true_amp[0]),               # log I_r_amp
        np.arccos(0.8),            # received angle diff
        np.arccos(0.8)             # sent angle diff
    ])

    P = np.diag([160.0, sigma_V**2, sigma_I**2, sigma_phiV**2 + sigma_phiI**2, sigma_phiV**2 + sigma_phiI**2])

    X_est = np.zeros((steps, n))
    V_s_est_list = np.zeros(steps)
    I_s_est_list = np.zeros(steps)
    T_est_list = np.zeros(steps)
    sent_angle_est_list = np.zeros(steps)
    received_angle_est_list = np.zeros(steps)
    sigma_list = np.zeros((steps, n))
    K_state_norms = np.zeros((steps, n))

    # --------------------------
    # UKF loop (log amplitudes)
    # --------------------------
    for k in range(steps):
        # --- Sigma points ---
        P_sqrt = np.linalg.cholesky((n + lambda_) * P)
        chi = np.zeros((2 * n + 1, n))
        chi[0] = x_est
        for i in range(n):
            chi[i + 1] = x_est + P_sqrt[:, i]
            chi[i + 1 + n] = x_est - P_sqrt[:, i]

        # --- Predict step (random walk) ---
        chi_pred = chi
        x_pred = np.sum(Wm[:, None] * chi_pred, axis=0)

        # --- Measurement prediction ---
        Z_sigma = np.zeros((2 * n + 1, m))
        for i in range(2 * n + 1):
            T_i, log_V_r_i, log_I_r_i, received_angle_diff_i, sent_angle_diff_i = chi_pred[i]
            V_r_amp_i = np.exp(log_V_r_i)
            I_r_amp_i = np.exp(log_I_r_i)

            # Sending-end phasor calculation
            V_s_amp_i, V_s_angle_i, I_s_amp_i, I_s_angle_i = pi_model_sending_phasor(
                V_r_amp_i, 0.0,
                I_r_amp_i, -received_angle_diff_i,
                T_i,
                condParameter
            )

            # Convert sending-end amplitudes to log-space
            log_V_s_amp_i = np.log(V_s_amp_i)
            log_I_s_amp_i = np.log(I_s_amp_i)

            Z_sigma[i] = [log_V_r_i, log_I_r_i, log_V_s_amp_i, log_I_s_amp_i, received_angle_diff_i, sent_angle_diff_i]

        z_pred = np.sum(Wm[:, None] * Z_sigma, axis=0)
        P_zz = R + sum(Wc[i] * np.outer(Z_sigma[i] - z_pred, Z_sigma[i] - z_pred) for i in range(2 * n + 1))
        P_xz = sum(Wc[i] * np.outer(chi_pred[i] - x_pred, Z_sigma[i] - z_pred) for i in range(2 * n + 1))

        # --- Measurement vector in log-space ---
        z_meas = np.array([
            np.log(V_r_meas[k]),
            np.log(I_r_meas[k]),
            np.log(V_s_meas[k]),
            np.log(I_s_meas[k]),
            received_angle_meas[k],
            sent_angle_meas[k]
        ])

        # --- Update ---
        K = P_xz @ np.linalg.inv(P_zz)
        x_est = x_pred + K @ (z_meas - z_pred)

        # --- Adaptive Q ---
        innovation = z_meas - z_pred
        eta_k = innovation.T @ np.linalg.inv(P_zz) @ innovation
        Q_scaled = gamma * Q if eta_k > eta_th else Q

        # --- Predict covariance ---
        P_pred = Q_scaled + sum(Wc[i] * np.outer(chi_pred[i] - x_pred, chi_pred[i] - x_pred) for i in range(2 * n + 1))
        P = P_pred - K @ P_zz @ K.T

        # --- Store estimates ---
        X_est[k] = x_est
        T_est_list[k] = x_est[0]
        received_angle_est_list[k] = x_est[3]

        V_s_amp_i, V_s_angle_i, I_s_amp_i, I_s_angle_i = pi_model_sending_phasor(
            np.exp(x_est[1]), 0.0,
            np.exp(x_est[2]), -x_est[3],
            x_est[0], condParameter
        )
        V_s_est_list[k] = V_s_amp_i
        I_s_est_list[k] = I_s_amp_i
        sent_angle_est_list[k] = V_s_angle_i - I_s_angle_i

        sigma_list[k] = np.sqrt(np.diag(P))
        for i in range(n):
            K_state_norms[k, i] = np.linalg.norm(K[i, :])

        if (k + 1) % (steps // 10) == 0:
            print(f'[{label}] {100 * (k + 1) / steps:.1f}% complete')

    
    err = T_est_list - T_true
    sigma = sigma_list[:,0]

    return err, sigma


def plot_event_difference(diff, start_idx, label, mode, direction):

    start = max(0, start_idx - pre_event)
    end = min(len(diff), start_idx + post_event)

    diff_slice = diff[start:end]

    t_sec = (np.arange(start, end) - start_idx) / steps_psec
    plt.plot(t_sec, diff_slice, label=label, color = 'tab:blue' if direction == "dn" else 'tab:red', linestyle = '-' if mode == "curr" else '--')

from concurrent.futures import ProcessPoolExecutor

def run_wrapper(args):
    return args, run_scenario(*args)

if __name__ == "__main__":
    scenarios = [
        ("baseline",),
        ("conv", "up"),
        ("conv", "dn"),
        ("curr", "up"),
        ("curr", "dn"),
    ]

    results = {}

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_wrapper, s) for s in scenarios]

        for future in futures:
            args, (err, sigma) = future.result()
            results[args] = (err, sigma)

    # unpack results
    err_base, sigma_base = results[("baseline",)]
    err_conv_up, sigma_conv_up = results[("conv", "up")]
    err_conv_dn, sigma_conv_dn = results[("conv", "dn")]
    err_curr_up, sigma_curr_up = results[("curr", "up")]
    err_curr_dn, sigma_curr_dn = results[("curr", "dn")]

    delta_curr_up = err_curr_up - err_base
    delta_conv_up = err_conv_up - err_base

    delta_curr_dn = err_curr_dn - err_base
    delta_conv_dn = err_conv_dn - err_base

    # --------------------------
    # Plotting event comparisons
    # --------------------------
    folder = "results/figures/ukf/error_comparison"
    os.makedirs(folder, exist_ok=True)




    plt.figure(figsize=(8,5))

    # Current disturbance comparison
    plot_event_difference(
        delta_curr_dn,
        startCurr,
        "Current decrease",
        "curr",
        "dn"
    )

    # Convection disturbance comparison
    plot_event_difference(
        delta_conv_dn,
        startConv,
        "Convection increase",
        "conv",
        "dn"
    )

    # Current disturbance comparison
    plot_event_difference(
        delta_curr_up,
        startCurr,
        "Current increase",
        "curr",
        "up"
    )

    # Convection disturbance comparison
    plot_event_difference(
        delta_conv_up,
        startConv,
        "Convection decrease",
        "conv",
        "up"
    )


    plt.axvline(0, color='black', linestyle='--', linewidth=0.8, label="Event start")
    plt.axhline(0, color='black', linewidth=0.8)
    plt.xlabel('Time relative to event [s]')
    plt.ylabel('Temperature Error Difference [°C]')

    # Color legend (direction type)
    color_legend = [
        Line2D([0], [0], color='tab:blue', lw=2, label='Current Decrease'),
        Line2D([0], [0], color='tab:blue', lw=2, linestyle='--', label='Convection Increase')
    ]

    # Linestyle legend (mode)
    style_legend = [
        Line2D([0], [0], color='tab:red', lw=2, linestyle='-', label='Current Increase'),
        Line2D([0], [0], color='tab:red', lw=2, linestyle='--', label='Convection Decrease')
    ]

    # First legend (colors)
    legend1 = plt.legend(handles=color_legend, loc='upper right')

    # Second legend (linestyles)
    legend2 = plt.legend(handles=style_legend, loc='lower right')

    # Add the first legend back manually
    plt.gca().add_artist(legend1)

    plt.grid(False)
    plt.tight_layout()
    filename="comparison_vs_baseline"
    plt.savefig(os.path.join(folder, filename), dpi=300)
    plt.close()
