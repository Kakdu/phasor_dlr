import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
import argparse

# Import synthetic data-generation
from phasor_dlr.synthetic_data.generators import generate_synthetic_data

# Import physical relations
from phasor_dlr.models.pi_model_static import pi_model_sending_phasor

# Import mathematical functions
from phasor_dlr.utils.math import turn_on_exp
from phasor_dlr.utils.math import combined_uniforms_sigma

# Import conductor parameters and error intervals
from phasor_dlr.config.defaults import condParameters
from phasor_dlr.config.standards import ErrorIntervals, make_error_intervals

# Import UKF modules
from phasor_dlr.estimation.ukf.config import ukf_initialize
from phasor_dlr.estimation.ukf.sigma_points import ukf_sigma_points
from phasor_dlr.estimation.ukf.filtering import ukf_predict_measurement, ukf_update

from phasor_dlr.plotting.ukf import save_plot, plot_temperature, plot_kalman_gain, plot_smoothed_temperature

DEFAULTS = {
    "min": 100,
    "seed": 1,
    "cond": "bohus",
    "conf": 95,
}

parser = argparse.ArgumentParser(description="Run power system script.")

parser.add_argument("--min", type=int, default=DEFAULTS["min"])
parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
parser.add_argument("--conf", type=int, default=DEFAULTS["conf"])
parser.add_argument("--CT_class", type=str, default="5P")
parser.add_argument("--VT_class", type=str, default="0.2")
parser.add_argument("--cond", type=str, default=DEFAULTS["cond"])

parser.add_argument("--conv", type=bool, default=False)
parser.add_argument("--curr", type=bool, default=False)

args = parser.parse_args()



# --------------------------
# Conductor parameters
# --------------------------
condParameter = condParameters[args.cond]


# --------------------------
# Precision and Accuracy Standards
# --------------------------
intervals = make_error_intervals(args.CT_class, args.VT_class)

sigma_V = combined_uniforms_sigma(intervals.r_V)
sigma_I = combined_uniforms_sigma(intervals.r_I)
sigma_phiV = combined_uniforms_sigma(intervals.theta_V)
sigma_phiI = combined_uniforms_sigma(intervals.theta_I)


# --------------------------
# UKF parameters
# --------------------------
minutes = args.min
steps_psec = 100
steps = minutes * steps_psec * 60
np.random.seed(4)
n = 5  # [T, V_r_amp, I_r_amp, received_angle_diff, sent_angle_diff]
m = 6  # [V_r_amp, I_r_amp, V_s_amp, I_s_amp, sent_angle_diff, received_angle_diff]


# --------------------------
# Synthetic measurements
# --------------------------
data = generate_synthetic_data(
    steps,
    condParameter,
    sigma_V, sigma_I, sigma_phiV, sigma_phiI,
    conv=args.conv,
    curr=args.curr,
    steps_psec=100
)


# --------------------------
# Initialization
# --------------------------
folder = "results/figures/ukf"
if args.conv == 1 and args.curr == 1:
    folder += "/Combination"
elif args.conv == 1:
    folder += "/Convection"
elif args.curr == 1:
    folder += "/Current"
os.makedirs(folder, exist_ok=True)

ukf_params = ukf_initialize(n, m, sigma_V, sigma_I, sigma_phiV, sigma_phiI)
x0 = np.array([55.0, 138_000, 800, np.arccos(0.8), np.arccos(0.8)])
P0 = np.diag([160.0, 138_000**2 * sigma_V**2, 900**2 * sigma_I**2, sigma_phiV**2 + sigma_phiI**2, sigma_phiV**2 + sigma_phiI**2])

# Allocate storage
T_est_list = np.zeros(steps)
V_s_est_list = np.zeros(steps)
I_s_est_list = np.zeros(steps)
X_est = np.zeros((steps, n))
sigma_list = np.zeros((steps, n))
K_norm_list = np.zeros(steps)
K_state_norms = np.zeros((steps, n))

x_est = x0.copy()   # initial state
P = P0.copy()       # initial covariance


# --------------------------
# UKF Loop
# --------------------------
for k in range(steps):
    # Sigma points
    chi = ukf_sigma_points(x_est, P, n, ukf_params['lambda_'])
    
    # State prediction
    x_pred = np.sum(ukf_params['Wm'][:, None] * chi, axis=0)
    
    # Predict measurement
    Z_sigma, z_pred, P_zz = ukf_predict_measurement(chi, ukf_params['Wm'], ukf_params['Wc'], condParameter)
    
    # Measured values
    z_meas = np.array([
        data["V_r_meas"][k],
        data["I_r_meas"][k],
        data["V_s_meas"][k],
        data["I_s_meas"][k],
        data["received_angle_meas"][k],
        data["sent_angle_meas"][k]
    ])
    
    # UKF update
    x_est, P, K = ukf_update(x_pred, chi, z_pred, Z_sigma, P_zz, z_meas, ukf_params['Wc'], ukf_params['Q'], ukf_params['gamma'])
    
    # Store results
    X_est[k, :] = x_est
    sigma_list[k, :] = np.sqrt(np.diag(P))
    T_est_list[k] = x_est[0]          # temperature
    V_s_est_list[k] = x_est[1]        # V_s
    I_s_est_list[k] = x_est[2]        # I_s
    K_norm_list[k] = np.linalg.norm(K, 'fro')
    K_state_norms[k, :] = np.linalg.norm(K, axis=1)

# --------------------------
# Post-processing metrics
# --------------------------
rmse_T = np.sqrt(np.mean((T_est_list - data["T_true"])**2))
mae_T = np.mean(np.abs(T_est_list - data["T_true"]))
var_T = np.mean(sigma_list[:,0])

print(f"RMSE of temperature estimates: {rmse_T:.4f} °C")
print(f"MAE of temperature estimates: {mae_T:.4f} °C")
print(f"Variance of temperature estimates: {var_T:.4f}")

# --------------------------
# Plots using functions
# --------------------------
confidence = args.conf

# Voltage & current plots
save_plot(data["V_r_true"].amplitude*np.ones(steps), X_est[:,1], sigma_list[:,1], 'Voltage [V]', 'V_r_estimate.png', folder, steps_psec, confidence, color='tab:green')
save_plot(data["I_r_true"].amplitude*np.ones(steps), X_est[:,2], sigma_list[:,2], 'Current [A]', 'I_r_estimate.png', folder, steps_psec, confidence, color='tab:green')
save_plot(data["V_s_true"]*np.ones(steps), V_s_est_list, sigma_list[:,1], 'Voltage [V]', 'V_s_estimate.png', folder, steps_psec, confidence, color='tab:green')
save_plot(data["I_s_true_amp"]*np.ones(steps), I_s_est_list, sigma_list[:,2], 'Current [A]', 'I_s_estimate.png', folder, steps_psec, confidence, color='tab:green')

# Angle differences
save_plot(received_angle_diff_true*np.ones(steps), X_est[:,3], sigma_list[:,3], 'Received angle difference [rad]', 'received_angle_diff_estimate.png', folder, steps_psec, confidence, color='tab:orange')
save_plot(sent_angle_diff_true*np.ones(steps), X_est[:,4], sigma_list[:,4], 'Sent angle difference [rad]', 'sent_angle_diff_estimate.png', folder, steps_psec, confidence, color='tab:orange')

# Temperature plots
plot_temperature(T_est_list, data["T_true"], sigma_list[:,0], folder, steps_psec, confidence)
plot_smoothed_temperature(T_est_list, data["T_true"], sigma_list[:,0], steps_psec, folder, confidence)

# Kalman gain plots
labels = [r'$T_{\mathrm{AV}}$', r'$\nu^{(r)}$', r'$\iota^{(r)}$', r'$\phi^{(r)}$', r'$\phi^{(s)}$']
plot_kalman_gain(K_norm_list, K_state_norms, n, folder, steps_psec, labels)

print("UKF simulation complete. Plots saved to folder.")