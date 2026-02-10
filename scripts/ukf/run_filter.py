import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
from dataclasses import dataclass

import argparse

# Import math functions
from phasor_dlr.utils.math import combined_uniforms_sigma

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
    "stand": "5P"
}

parser = argparse.ArgumentParser(description="Run power system script.")

parser.add_argument("--min", type=int, default=DEFAULTS["min"])
parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
parser.add_argument("--conf", type=int, default=DEFAULTS["conf"])
parser.add_argument("--CT_class", type=str, default="5P")
parser.add_argument("--VT_class", type=str, default="0.2")
parser.add_argument("--cond", type=str, default=DEFAULTS["cond"])

parser.add_argument("--conv", type=int, default=0)
parser.add_argument("--curr", type=int, default=0)

args = parser.parse_args()



# --------------------------
# Phasor dataclass
# --------------------------
@dataclass(frozen=False)
class phasor:
    amplitude: float
    angle: float

# --------------------------
# π-model sending phasor function
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
# Temperature from phasors
# --------------------------
def T_from_measurements(V_r: phasor, I_r: phasor, V_s: phasor, I_s: phasor, condParameter):
    I_AC = np.sqrt(0.5 * (I_s.amplitude**2 + I_r.amplitude**2))
    return condParameter["T_ref"] + 1 / condParameter["alpha"] * ((V_s.amplitude * I_s.amplitude * np.cos(V_s.angle - I_r.angle) - V_r.amplitude * I_r.amplitude * np.cos(V_r.angle - I_r.angle))/( 0.5 * (I_s.amplitude**2 + I_r.amplitude**2) * condParameter["R_20"] * condParameter["L"]) - 1)

# --------------------------
# Sigmoid for temperature
# --------------------------
def turn_on_exp(x, x0=0.0, tau=1.0, A=1.0):
    return A * (1 - np.exp(-(x - x0) / tau)) * (x > x0)

# --------------------------
# Conductor parameters
# --------------------------
condParameter = condParameters[args.cond]
# --------------------------
# Precision and Accuracy Standards
# --------------------------

# -------------------------------
# Measurement accuracy and variance
# -------------------------------
accuracy_r = make_error_intervals(args.CT_class, args.VT_class)
accuracy_s = make_error_intervals(args.CT_class, args.VT_class)

sigma_V = combined_uniforms_sigma(accuracy_r.r_V)
sigma_I = combined_uniforms_sigma(accuracy_r.r_I)
sigma_phiV = combined_uniforms_sigma(accuracy_r.theta_V)
sigma_phiI = combined_uniforms_sigma(accuracy_r.theta_I)

# --------------------------
# UKF parameters
# --------------------------
minutes = args.min
steps_psec = 100
steps = minutes * steps_psec * 60
np.random.seed(4)
n = 5  # [T, V_r_amp, I_r_amp, received_angle_diff, sent_angle_diff]
m = 6  # [V_r_amp, I_r_amp, V_s_amp, I_s_amp, sent_angle_diff, received_angle_diff]

# Chi-square threshold for 99% confidence (n_z = 6)
eta_th = 10.6  # for 90% confidence, use 10.6

# Scaling factor
gamma = 10


alpha, beta, kappa = 1e-3, 2, 0
Q = np.diag([0.1, 0.1, 0.01, 0.00001, 0.00001])
Q = Q / 500
R = np.diag([138_000**2*sigma_V**2, 900*sigma_I**2, 138_000**2*sigma_V**2, 900*sigma_I**2, (sigma_phiV**2 + sigma_phiI**2), (sigma_phiV**2 + sigma_phiI**2)])
lambda_ = alpha**2*(n+kappa) - n
Wm = np.full(2*n+1, 0.5/(n+lambda_))
Wc = Wm.copy()
Wm[0] = lambda_/(n+lambda_)
Wc[0] = Wm[0] + (1 - alpha**2 + beta)

# --------------------------
# Synthetic true states (receiving-end)
# --------------------------
x = np.linspace(1, steps, steps)
T_true = 55 + 2*np.sin(np.linspace(0,5*minutes/100,steps))
V_r_true = phasor(138_000,0)
I_r_true = phasor(800 + 25*np.sin(np.linspace(0,4*minutes/100,steps)), - np.arccos(0.8) + 0.05*np.sin(np.linspace(0,4*minutes/100,steps)))

if args.conv == 1:
    startConv = steps // 4
    tauConv = steps_psec * 60 * 8
    T_true -= turn_on_exp(x, startConv, tau=tauConv, A=10)
if args.curr == 1:
    startCurr = steps // 2
    tauCurr = steps_psec * 60 * 13
    T_true += turn_on_exp(x, startCurr, tau = tauCurr, A=15)
    I_r_true.amplitude += 400 * np.concatenate([np.zeros(startCurr), np.ones(steps - startCurr)])

V_s_true_amp, V_s_true_angle, I_s_true_amp, I_s_true_angle = pi_model_sending_phasor(
    V_r_true.amplitude, V_r_true.angle, I_r_true.amplitude, I_r_true.angle, T_true, condParameter
)

# Angle differences
received_angle_diff_true = (V_r_true.angle - I_r_true.angle)
sent_angle_diff_true = (V_s_true_angle - I_s_true_angle)

# -------------------------------
# Apply measurement errors using previously defined functions
# -------------------------------

# Receiving end
V_r_meas, V_r_angle_meas, I_r_meas, I_r_angle_meas = apply_error_to_phasors(
    V_r_true.amplitude, V_r_true.angle, I_r_true.amplitude, I_r_true.angle,
    accuracy_r, steps, seed=1
)

# Sending end
V_s_meas, V_s_angle_meas, I_s_meas, I_s_angle_meas = apply_error_to_phasors(
    V_s_true_amp, V_s_true_angle, I_s_true_amp, I_s_true_angle,
    accuracy_s, steps, seed=2
)


sent_angle_meas = V_s_angle_meas - I_s_angle_meas
received_angle_meas = V_r_angle_meas - I_r_angle_meas


T_meas = np.array([T_from_measurements(
    phasor(V_r_meas[k],0.0), phasor(I_r_meas[k],sent_angle_meas[k]),
    phasor(V_s_meas[k],0.0), phasor(I_s_meas[k],received_angle_meas[k]),
    condParameter
) for k in range(steps)])

# --------------------------
# UKF initialization
# --------------------------
x_est = np.array([55.0, 138_000, 800, np.arccos(0.8), np.arccos(0.8)])
P = np.diag([160.0, 138_000**2 * sigma_V**2, 900**2 * sigma_I**2, sigma_phiV**2 + sigma_phiI**2, sigma_phiV**2 + sigma_phiI**2])

X_est = np.zeros((steps,n))
V_s_est_list = np.zeros(steps)
I_s_est_list = np.zeros(steps)
T_est_list = np.zeros(steps)
sent_angle_est_list = np.zeros(steps)
received_angle_est_list = np.zeros(steps)
sigma_list = np.zeros((steps,n))

K_norm_list = np.zeros(steps)        # overall Kalman gain size
K_state_norms = np.zeros((steps, n)) # per-state gain size

# --------------------------
# UKF loop
# --------------------------
for k in range(steps):
    # --- Sigma points ---
    P_sqrt = np.linalg.cholesky((n+lambda_)*P)
    chi = np.zeros((2*n+1,n))
    chi[0] = x_est
    for i in range(n):
        chi[i+1] = x_est + P_sqrt[:,i]
        chi[i+1+n] = x_est - P_sqrt[:,i]

    # --- Predict step ---
    chi_pred = chi  # random walk, may be defined differently. 
    x_pred = np.sum(Wm[:,None]*chi_pred, axis=0)


    # --- Measurement prediction ---
    Z_sigma = np.zeros((2*n+1,6))
    for i in range(2*n+1):
        T_i, V_r_amp_i, I_r_amp_i, received_angle_diff_i, sent_angle_diff_i = chi_pred[i]
        V_r_i = phasor(V_r_amp_i, 0.0)
        I_r_i = phasor(I_r_amp_i, -received_angle_diff_i)

        V_s_amp_i, V_s_angle_i, I_s_amp_i, I_s_angle_i = pi_model_sending_phasor(
            V_r_i.amplitude, V_r_i.angle, I_r_i.amplitude, I_r_i.angle, T_i, condParameter
        )

        Z_sigma[i] = [V_r_i.amplitude, I_r_i.amplitude, V_s_amp_i, I_s_amp_i, received_angle_diff_i, sent_angle_diff_i]

    z_pred = np.sum(Wm[:,None]*Z_sigma, axis=0)
    P_zz = R + sum(Wc[i]*np.outer(Z_sigma[i]-z_pred, Z_sigma[i]-z_pred) for i in range(2*n+1))
    P_xz = sum(Wc[i]*np.outer(chi_pred[i]-x_pred, Z_sigma[i]-z_pred) for i in range(2*n+1))

    # --- Update ---
    z_meas = np.array([V_r_meas[k], I_r_meas[k], V_s_meas[k], I_s_meas[k], received_angle_meas[k], sent_angle_meas[k]])
    
    K = P_xz @ np.linalg.inv(P_zz)
    # Overall Kalman gain magnitude
    K_norm_list[k] = np.linalg.norm(K, ord='fro')
    # Per-state gain magnitude
    for i in range(n):
        K_state_norms[k, i] = np.linalg.norm(K[i, :])
    
    x_est = x_pred + K @ (z_meas - z_pred)
    # --- Adaptive Q ---
    innovation = z_meas - z_pred
    eta_k = innovation.T @ np.linalg.inv(P_zz) @ innovation
    Q_scaled = gamma * Q if eta_k > eta_th else Q
    # --- Predict covariance ---
    P_pred = Q_scaled + sum(Wc[i]*np.outer(chi_pred[i]-x_pred, chi_pred[i]-x_pred) for i in range(2*n+1))
    P = P_pred - K @ P_zz @ K.T

    # --- Store ---
    X_est[k] = x_est
    T_est_list[k] = x_est[0]
    received_angle_est_list[k] = x_est[3]
    V_s_amp_i, V_s_angle_i, I_s_amp_i, I_s_angle_i = pi_model_sending_phasor(
        x_est[1], 0.0,
        x_est[2], -x_est[3],
        x_est[0], condParameter
    )
    V_s_est_list[k] = V_s_amp_i
    I_s_est_list[k] = I_s_amp_i
    sent_angle_est_list[k] = V_s_angle_i - I_s_angle_i
    sigma_list[k] = np.sqrt(np.diag(P))
    if (k + 1) % (steps // 10) == 0:
        print(f'{100 * (k + 1) / steps}% complete')


rmse_T = np.sqrt(np.mean((T_est_list - T_true)**2))
mae_T = np.mean(np.abs(T_est_list - T_true))
var_T = np.mean(sigma_list[:,0])

print(f"RMSE of temperature estimates: {rmse_T:.4f} deg C")
print(f"MAE of temperature estimates: {mae_T:.4f} deg C")
print(f"Variance of temperature estimates: {var_T:.4f}")

# --------------------------
# Plotting function
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

def plot_true_only(true, ylabel, filename, steps_psec, color='black', label='true'):
    """
    Plot only the true/nominal signal over time.

    Parameters
    ----------
    true : array-like
        True/nominal signal to plot.
    ylabel : str
        Label for y-axis.
    filename : str
        Output file name (saved in `folder`).
    steps_psec : int
        Number of steps per second (for time conversion).
    color : str
        Line color.
    label : str
        Label for the legend.
    """
    t_min = np.arange(len(true)) / (60 * steps_psec)

    plt.figure()
    plt.plot(t_min, true, color=color, linewidth=0.8, label=label)
    plt.xlabel('Time [minutes]')
    plt.ylabel(ylabel)
    plt.ylim(40, 70)
    plt.title(f'Nominal Temperature')
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, filename), dpi=300)
    plt.close()

def average_per_second(signal, steps_psec):
    """
    Block-average a signal over one-second intervals.
    """
    n_blocks = len(signal) // steps_psec
    return signal[:n_blocks * steps_psec].reshape(n_blocks, steps_psec).mean(axis=1)


confidence = args.conf
k_val = norm.ppf(0.5 + confidence/200)

save_plot(V_r_true.amplitude*np.ones(steps), X_est[:,1], sigma_list[:,1], 'Voltage [V]', 'V_r_estimate.png', steps_psec, confidence, color='tab:green')
save_plot(I_r_true.amplitude*np.ones(steps), X_est[:,2], sigma_list[:,2], 'Current [A]', 'I_r_estimate.png', steps_psec, confidence, color='tab:green')
save_plot(V_s_true_amp*np.ones(steps), V_s_est_list, sigma_list[:,1], 'Voltage [V]', 'V_s_estimate.png', steps_psec, confidence, color='tab:green')
save_plot(I_s_true_amp*np.ones(steps), I_s_est_list, sigma_list[:,2], 'Current [A]', 'I_s_estimate.png', steps_psec, confidence, color='tab:green')

# Input angle difference (V_r - I_r)
save_plot(
    received_angle_diff_true * np.ones(steps),
    X_est[:, 3],
    sigma_list[:, 3],
    'Received angle difference [rad]',
    'received_angle_diff_estimate.png',
    steps_psec,
    confidence,
    color='tab:orange'
)

# Output angle difference (V_s - I_s)
save_plot(
    sent_angle_diff_true * np.ones(steps),
    X_est[:, 4],
    sigma_list[:, 4],
    'Sent angle difference [rad]',
    'sent_angle_diff_estimate.png',
    steps_psec,
    confidence,
    color='tab:orange'
)

t_min = np.arange(len(T_est_list)) / (60 * steps_psec)

plt.figure()

# Estimated temperature
plt.plot(t_min, T_est_list, color='tab:blue', label=r'$T_{\mathrm{EST}}$')
plt.fill_between(
    t_min,
    T_est_list - k_val * sigma_list[:, 0],
    T_est_list + k_val * sigma_list[:, 0],
    color='tab:blue',
    alpha=0.3,
    label=f'{confidence}% CI'
)

# Nominal temperature
plt.plot(t_min, T_true, linewidth=0.8, color='black', label=r'$T_{\mathrm{NOM}}$')

plt.xlabel('Time [minutes]')
plt.ylabel(r'Temperature [$^\circ$C]')
plt.title('')
plt.legend()
plt.grid(False)

plt.savefig(os.path.join(folder, "Temperature_with_limits.png"), dpi=300)
plt.close()

plt.figure()
# Difference
plt.plot(
    t_min,
    T_est_list - T_true,
    color='tab:blue',
    label=r'$T_{\mathrm{EST}} - T_{\mathrm{NOM}}$'
)

# Zero and limits
plt.axhline(0, color='black', linewidth=0.8)
plt.axhline(5, color='red', linestyle=':', label=r'$\pm 5^\circ\mathrm{C}$ limit')
plt.axhline(-5, color='red', linestyle=':')

plt.xlabel('Time [minutes]')
plt.ylabel(r'Temperature Difference [$^\circ$C]')
plt.title('')
plt.legend()
plt.grid(False)

plt.savefig(os.path.join(folder, "Temperature_difference_with_limits.png"), dpi=300)
plt.close()

plt.figure()
plt.plot(np.linspace(steps//25,steps,steps-steps//25)/steps_psec/60, K_norm_list[steps//25:])
plt.xlabel("Time [minutes]")
plt.ylabel(r"$||K||_F$")
plt.title("")
plt.grid(False)
plt.savefig(os.path.join(folder, "kalman_gain_overall.png"), dpi=300)
plt.close()



labels = [r'$T_{\mathrm{AV}}$', r'$\nu^{(r)}$', r'$\iota^{(r)}$', r'$\phi^{(r)}$', r'$\phi^{(s)}$']
colors = ['tab:blue', 'tab:green', 'tab:green', 'tab:orange', 'tab:orange']
linestyles = ['-', '-', ':', '-', ':']

# Temperature nominal
if args.curr * args.conv == 1:
    label='Combination'
elif args.curr == 1:
    label='Current'
elif args.conv == 1:
    label='Convection'
else:
    label='Baseline'
plot_true_only(
    T_true,
    ylabel=r'Temperature [$^\circ$C]',
    filename='temperature_nominal.png',
    steps_psec=steps_psec,
    label=label
)

plt.figure()
for i in range(n):
    plt.plot(np.linspace(steps//25,steps,steps-steps//25)/steps_psec/60, K_state_norms[steps//25:, i], label=labels[i], color=colors[i], linestyle=linestyles[i])
plt.xlabel("Time [minutes]")
plt.ylabel(r"$||K_i||$")
plt.title("")
plt.legend()
plt.grid(False)
plt.savefig(os.path.join(folder, "kalman_gain_per_state.png"), dpi=300)
plt.close()

# --- Smoothed (1-second averaged) temperature ---
T_est_smooth = average_per_second(T_est_list, steps_psec)
T_true_smooth = average_per_second(T_true, steps_psec)

# Variance = sigma^2 --> average
var_T_smooth = average_per_second(sigma_list[:, 0]**2, steps_psec)
sigma_T_smooth = np.sqrt(var_T_smooth)

# Time vector in minutes
t_min_smooth = np.arange(len(T_est_smooth)) / 60


plt.figure()
plt.plot(t_min_smooth, T_est_smooth, color='tab:blue', label='Smoothed estimate')
plt.fill_between(
    t_min_smooth,
    T_est_smooth - k_val * sigma_T_smooth,
    T_est_smooth + k_val * sigma_T_smooth,
    color='tab:blue',
    alpha=0.3,
    label=f'{confidence}% CI (smoothed)'
)
plt.plot(t_min_smooth, T_true_smooth, linewidth=0.8, color='black', label='Smoothed true')
plt.xlabel('Time [minutes]')
plt.ylabel('Temperature [deg C]')
plt.legend()
plt.savefig(os.path.join(folder, "temperature_estimate_smoothed.png"), dpi=300)
plt.close()

plt.figure()
plt.plot(
    t_min_smooth,
    T_est_smooth - T_true_smooth,
    '--',
    color='tab:blue',
    label=r'$T_{\mathrm{EST}} - T_{\mathrm{TRUE}}$ (smoothed)'
)
plt.fill_between(
    t_min_smooth,
    (T_est_smooth - T_true_smooth) - k_val * sigma_T_smooth,
    (T_est_smooth - T_true_smooth) + k_val * sigma_T_smooth,
    color='tab:blue',
    alpha=0.3,
    label=f'{confidence}% CI'
)
plt.axhline(0, color='black', linewidth=0.8)

plt.xlabel('Time [minutes]')
plt.ylabel('Temperature [deg C]')
plt.legend()
plt.savefig(os.path.join(folder, "temperature_difference_smoothed.png"), dpi=300)
plt.close()

print("UKF simulation complete. Plots saved to files.")
