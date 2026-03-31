import numpy as np
import matplotlib.pyplot as plt
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
    "min": 100,
    "seed": 1,
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

parser.add_argument("--conv", type=int, default=0)
parser.add_argument("--curr", type=int, default=0)

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
np.random.seed(4)

n = 5  # [T, log(V_r_amp), log(I_r_amp), received_angle_diff, sent_angle_diff]
m = 6  # [log(V_r_amp), log(I_r_amp), log(V_s_amp), log(I_s_amp), sent_angle_diff, received_angle_diff]

# Chi-square threshold for 99% confidence (n_z = 6)
eta_th = 10.6  # for 90% confidence, use 10.6

# Scaling factor
gamma = 10
alpha, beta, kappa = 1e-3, 2, 0

# --------------------------
# Synthetic true states (receiving-end)
# --------------------------
x = np.linspace(1, steps, steps)
T_true = 55 + 2*np.sin(np.linspace(0,5*minutes/100,steps))
V_r_true_amp = 140_000/np.sqrt(3)
V_r_true_angle = 0.0
I_r_true_amp = 800 + 25*np.sin(np.linspace(0,4*minutes/100,steps))
I_r_true_angle = - np.arccos(0.8) + 0.05*np.sin(np.linspace(0,4*minutes/100,steps))

if args.conv == 1:
    startConv = steps // 4
    tauConv = steps_psec * 60 * 8
    T_true -= turn_on_exp(x, startConv, tau=tauConv, A=10)
if args.curr == 1:
    startCurr = steps // 2
    tauCurr = steps_psec * 60 * 13
    T_true += turn_on_exp(x, startCurr, tau=tauCurr, A=15)
    I_r_true_amp += 400 * np.concatenate([np.zeros(startCurr), np.ones(steps - startCurr)])

V_s_true_amp, V_s_true_angle, I_s_true_amp, I_s_true_angle = pi_model_sending_phasor(
    V_r_true_amp, V_r_true_angle, I_r_true_amp, I_r_true_angle, T_true, condParameter
)

# Angle differences
received_angle_diff_true = V_r_true_angle - I_r_true_angle
sent_angle_diff_true = V_s_true_angle - I_s_true_angle


# -------------------------------
# Apply measurement errors
# -------------------------------
V_r_meas, V_r_angle_meas, I_r_meas, I_r_angle_meas = apply_error_to_phasors(
    V_r_true_amp, V_r_true_angle, I_r_true_amp, I_r_true_angle,
    accuracy_r, steps, seed=1
)

V_s_meas, V_s_angle_meas, I_s_meas, I_s_angle_meas = apply_error_to_phasors(
    V_s_true_amp, V_s_true_angle, I_s_true_amp, I_s_true_angle,
    accuracy_s, steps, seed=2
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
        print(f'{100 * (k + 1) / steps:.1f}% complete')


rmse_T = np.sqrt(np.mean((T_est_list - T_true)**2))
mae_T = np.mean(np.abs(T_est_list - T_true))
var_T = np.mean(sigma_list[:,0])

print("===== Total =====")
print(f"RMSE of temperature estimates: {rmse_T:.4f} deg C")
print(f"MAE of temperature estimates: {mae_T:.4f} deg C")
print(f"Variance of temperature estimates: {var_T:.4f}")

rmse_T = np.sqrt(np.mean((T_est_list[0:60*steps_psec] - T_true[0:60*steps_psec])**2))
mae_T = np.mean(np.abs(T_est_list[0:60*steps_psec] - T_true[0:60*steps_psec]))
var_T = np.mean(sigma_list[0:60*steps_psec,0])


print("===== First minute =====")
print(f"RMSE of temperature estimates: {rmse_T:.4f} deg C")
print(f"MAE of temperature estimates: {mae_T:.4f} deg C")
print(f"Variance of temperature estimates: {var_T:.4f}")

# --------------------------
# Plotting
# --------------------------
folder = "results/figures/ukf"
if args.conv == 1 and args.curr == 1:
    folder = folder + "/Combination"
elif args.conv == 1:
    folder = folder + "/Convection"
elif args.curr == 1:
    folder = folder + "/Current"
else: 
    folder = folder + "/Baseline"
os.makedirs(folder, exist_ok=True)

def save_plot(true, est, sigma, ylabel, filename, steps_psec, confidence=95, color='tab:blue'):
    k_val = norm.ppf(0.5 + confidence/200)
    t_min = np.arange(len(est)) / (60 * steps_psec)

    plt.figure()
    plt.plot(t_min, est, label='est', color=color)
    plt.fill_between(
        t_min,
        est - k_val * sigma,
        est + k_val * sigma,
        color=color,
        alpha=0.3,
        label=f'{confidence}% CI'
    )
    plt.plot(t_min, true, label='true', color='black', linewidth=0.8)

    plt.xlabel('Time [minutes]')
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(os.path.join(folder, filename), dpi=300)
    plt.close()

def average_per_second(signal, steps_psec):
    n_blocks = len(signal) // steps_psec
    return signal[:n_blocks * steps_psec].reshape(n_blocks, steps_psec).mean(axis=1)


confidence = args.conf
k_val = norm.ppf(0.5 + confidence/200)
t_min = np.arange(len(T_est_list)) / (60 * steps_psec)

# --- Temperature with limits ---
plt.figure()
plt.plot(t_min, T_est_list, color='tab:blue', label='Estimated Temperature')
plt.fill_between(
    t_min,
    T_est_list - k_val * sigma_list[:, 0],
    T_est_list + k_val * sigma_list[:, 0],
    color='tab:blue', alpha=0.3, label=f'{confidence}% CI'
)
plt.plot(t_min, T_true, color='black', linewidth=0.8, label='True Temperature')
plt.xlabel('Time [minutes]')
plt.ylabel('Temperature [°C]')
plt.legend()
plt.grid(False)
plt.savefig(os.path.join(folder, "Temperature_with_limits.png"), dpi=300)
plt.close()

from matplotlib.patches import Rectangle, ConnectionPatch

sec = 30
first_minute_steps = min(sec * steps_psec, steps)

t_zoom = t_min[:first_minute_steps]
T_est_zoom = T_est_list[:first_minute_steps]
T_true_zoom = T_true[:first_minute_steps]
sigma_zoom = sigma_list[:first_minute_steps, 0]

fig, (ax_zoom, ax_main) = plt.subplots(
    2, 1,
    figsize=(12,6),
    gridspec_kw={"height_ratios":[1,1]}
)

# --------------------------
# TOP: first minute zoom
# --------------------------
ax_zoom.plot(60*t_zoom, T_est_zoom, color='tab:blue')
ax_zoom.fill_between(
    60*t_zoom,
    T_est_zoom - k_val * sigma_zoom,
    T_est_zoom + k_val * sigma_zoom,
    color='tab:blue',
    alpha=0.25
)
ax_zoom.plot(60*t_zoom, T_true_zoom, color='black', linewidth=1)

ax_zoom.set_xlim(0,sec)
ax_zoom.set_ylabel('Temperature [°C]')

# --------------------------
# BOTTOM: full simulation
# --------------------------
ax_main.plot(60*t_min, T_est_list, color='tab:blue', label='Estimated temperature')
ax_main.fill_between(
    60*t_min,
    T_est_list - k_val * sigma_list[:,0],
    T_est_list + k_val * sigma_list[:,0],
    color='tab:blue',
    alpha=0.25,
    label=f'{confidence}% CI'
)
ax_main.plot(60*t_min, T_true, color='black', linewidth=1)

ax_main.set_xlabel("Time [seconds]")
ax_main.set_ylabel("Temperature [°C]")
ax_main.legend()


# --------------------------
# Compute CI bounds for first minute
# --------------------------
ci_lower = T_est_zoom - k_val * sigma_zoom
ci_upper = T_est_zoom + k_val * sigma_zoom

ymin_zoom = min(ci_lower.min(), T_true_zoom.min())
ymax_zoom = max(ci_upper.max(), T_true_zoom.max())

# add small margin
margin = 0.05 * (ymax_zoom - ymin_zoom)

ymin_zoom -= margin
ymax_zoom += margin


# --------------------------
# Draw zoom rectangle
# --------------------------
rect = Rectangle(
    (0, ymin_zoom),              # x start, y start
    sec,                           # width = some sec
    ymax_zoom - ymin_zoom,       # height based on CI
    linewidth=1.2,
    edgecolor='0.35',
    facecolor='none',
    linestyle='--'
)

ax_main.add_patch(rect)

con1 = ConnectionPatch(
    xyA=(0, ymax_zoom),
    coordsA=ax_main.transData,
    xyB=(0, ax_zoom.get_ylim()[0]),
    coordsB=ax_zoom.transData,
    color="0.35",
    linewidth=1
)

con2 = ConnectionPatch(
    xyA=(sec, ymax_zoom),
    coordsA=ax_main.transData,
    xyB=(sec, ax_zoom.get_ylim()[0]),
    coordsB=ax_zoom.transData,
    color="0.35",
    linewidth=1
)

fig.add_artist(con1)
fig.add_artist(con2)

plt.tight_layout()

plt.savefig(
    os.path.join(folder, "Temperature_with_limits.png"),
    dpi=400,
    bbox_inches="tight"
)

plt.close()

# --- Temperature difference with limits ---
plt.figure()
plt.plot(t_min, T_est_list - T_true, color='tab:blue', label='Est - True')
plt.axhline(0, color='black', linewidth=0.8)
plt.axhline(5, color='red', linestyle=':', label='±5°C limit')
plt.axhline(-5, color='red', linestyle=':')
plt.xlabel('Time [minutes]')
plt.ylabel('Temperature Difference [°C]')
plt.legend()
plt.grid(False)
plt.savefig(os.path.join(folder, "Temperature_difference_with_limits.png"), dpi=300)
plt.close()

# --- Kalman gain per state ---
labels = [r'$T_{\mathrm{AV}}$', r'$\nu^{(r)}$', r'$\iota^{(r)}$', r'$\phi^{(r)}$', r'$\phi^{(s)}$']
colors = ['tab:blue', 'tab:green', 'tab:green', 'tab:orange', 'tab:orange']
linestyles = ['-', '-', ':', '-', ':']

plt.figure()
for i in range(n):
    plt.plot(np.linspace(steps//25, steps, steps-steps//25)/steps_psec/60,
             K_state_norms[steps//25:, i],
             label=labels[i], color=colors[i], linestyle=linestyles[i])
plt.xlabel("Time [minutes]")
plt.ylabel(r"$||K_i||$")
plt.legend()
plt.grid(False)
plt.savefig(os.path.join(folder, "kalman_gain_per_state.png"), dpi=300)
plt.close()

# --- Smoothed temperature estimates (optional) ---
T_est_smooth = average_per_second(T_est_list, steps_psec)
T_true_smooth = average_per_second(T_true, steps_psec)
var_T_smooth = average_per_second(sigma_list[:, 0]**2, steps_psec)
sigma_T_smooth = np.sqrt(var_T_smooth)
t_min_smooth = np.arange(len(T_est_smooth)) / 60

save_plot(T_true_smooth, T_est_smooth, sigma_T_smooth,
          'Temperature [°C]', 'Temperature_estimates.png',
          steps_psec, confidence, color='tab:blue')

# --------------------------
# First minute estimation error (independent plot)
# --------------------------
first_minute_steps = min(60 * steps_psec, steps)

t_sec = np.arange(first_minute_steps) / steps_psec
err = T_est_list[:first_minute_steps] - T_true[:first_minute_steps]
sigma = sigma_list[:first_minute_steps, 0]

plt.figure()
plt.plot(t_sec, err, color='tab:blue', label='Estimation error')
plt.fill_between(
    t_sec,
    err - k_val * sigma,
    err + k_val * sigma,
    color='tab:blue',
    alpha=0.3,
    label=f'{confidence}% CI'
)

plt.axhline(0, color='black', linewidth=0.8)

plt.xlabel('Time [s]')
plt.ylabel('Temperature Error [°C]')
plt.legend()
plt.grid(False)

plt.savefig(os.path.join(folder, "error_first_minute.png"), dpi=300)
plt.close()

# --------------------------
# Histogram of temperature estimation errors with Gaussian fit
# --------------------------


# --- Remove first 5% (filter warm-up) ---
start_idx = int(0.05 * steps)
temp_errors = T_est_list[start_idx:] - T_true[start_idx:]

# --- Gaussian fit ---
mu = np.mean(temp_errors)
sigma = np.std(temp_errors)

# --- Histogram ---
plt.figure()
counts, bins, _ = plt.hist(
    temp_errors,
    bins=60,
    density=True,
    label="Error histogram"
)

# --- Gaussian curve ---
x_vals = np.linspace(bins[0], bins[-1], 400)
gaussian = norm.pdf(x_vals, mu, sigma)

plt.plot(
    x_vals,
    gaussian,
    linewidth=2,
    label=f'Gaussian fit\nμ={mu:.3f}, σ={sigma:.3f}'
)

plt.xlabel('Temperature Error [°C]')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(False)

plt.savefig(os.path.join(folder, "temperature_error_histogram_gaussian.png"), dpi=300)
plt.close()

print("UKF simulation complete. Selected plots saved to files.")
