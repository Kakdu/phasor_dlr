import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def save_plot(true, est, sigma, ylabel, filename, folder, steps_psec=1, confidence=95, color='tab:blue'):
    """
    Generic function to save a plot with confidence intervals.
    """
    os.makedirs(folder, exist_ok=True)
    k_val = norm.ppf(0.5 + confidence / 200)
    t_min = np.arange(len(est)) / (60 * steps_psec)

    plt.figure()
    plt.plot(t_min, est, label='Estimate', color=color)
    plt.fill_between(
        t_min,
        est - k_val * sigma,
        est + k_val * sigma,
        color=color,
        alpha=0.3,
        label=f'{confidence}% CI'
    )
    plt.plot(t_min, true, label='True', color='black', linewidth=0.8)
    plt.xlabel('Time [minutes]')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(False)
    plt.savefig(os.path.join(folder, filename), dpi=300)
    plt.close()

def average_per_second(signal, steps_psec):
    """
    Block-average a signal over one-second intervals.
    """
    n_blocks = len(signal) // steps_psec
    return signal[:n_blocks * steps_psec].reshape(n_blocks, steps_psec).mean(axis=1)

def plot_temperature(T_est, T_true, sigma, folder, steps_psec=1, confidence=95):
    """
    Plot temperature estimate with confidence intervals and differences.
    """
    os.makedirs(folder, exist_ok=True)
    k_val = norm.ppf(0.5 + confidence / 200)
    t_min = np.arange(len(T_est)) / (60 * steps_psec)

    # Estimated vs nominal temperature
    plt.figure()
    plt.plot(t_min, T_est, color='tab:blue', label='Estimated')
    plt.fill_between(t_min, T_est - k_val * sigma, T_est + k_val * sigma,
                     color='tab:blue', alpha=0.3, label=f'{confidence}% CI')
    plt.plot(t_min, T_true, color='black', linewidth=0.8, label='True')
    plt.xlabel('Time [minutes]')
    plt.ylabel('Temperature [°C]')
    plt.title('Estimated Temperature')
    plt.legend()
    plt.savefig(os.path.join(folder, "Temperature_with_limits.png"), dpi=300)
    plt.close()

    # Temperature difference
    plt.figure()
    plt.plot(t_min, T_est - T_true, color='tab:blue', label='Estimate - True')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.axhline(5, color='red', linestyle=':', label='±5°C limit')
    plt.axhline(-5, color='red', linestyle=':')
    plt.xlabel('Time [minutes]')
    plt.ylabel('Temperature Difference [°C]')
    plt.title('Deviation from Nominal Temperature')
    plt.legend()
    plt.savefig(os.path.join(folder, "Temperature_difference_with_limits.png"), dpi=300)
    plt.close()

def plot_kalman_gain(K_norm_list, K_state_norms, n, folder, steps_psec, labels=None):
    """
    Plot overall and per-state Kalman gain magnitudes.
    """
    os.makedirs(folder, exist_ok=True)
    # Overall gain
    plt.figure()
    plt.plot(np.linspace(len(K_norm_list)//25, len(K_norm_list), len(K_norm_list)-len(K_norm_list)//25)/steps_psec/60, 
             K_norm_list[len(K_norm_list)//25:])
    plt.xlabel("Time [minutes]")
    plt.ylabel(r"$||K||_F$")
    plt.title("Overall Kalman Gain Magnitude")
    plt.grid(False)
    plt.savefig(os.path.join(folder, "kalman_gain_overall.png"), dpi=300)
    plt.close()

    # Per-state gain
    if labels is None:
        labels = [f'State {i}' for i in range(n)]
    colors = ['tab:blue', 'tab:green', 'tab:green', 'tab:orange', 'tab:orange']
    linestyles = ['-', '-', ':', '-', ':']

    plt.figure()
    for i in range(n):
        plt.plot(np.linspace(len(K_state_norms)//25, len(K_state_norms), len(K_state_norms)-len(K_state_norms)//25)/steps_psec/60,
                 K_state_norms[len(K_state_norms)//25:, i],
                 label=labels[i],
                 color=colors[i % len(colors)],
                 linestyle=linestyles[i % len(linestyles)])
    plt.xlabel("Time [minutes]")
    plt.ylabel(r"$||K_i||$")
    plt.title("Kalman Gain Magnitude per State")
    plt.legend()
    plt.grid(False)
    plt.savefig(os.path.join(folder, "kalman_gain_per_state.png"), dpi=300)
    plt.close()

def plot_smoothed_temperature(T_est, T_true, sigma, steps_psec, folder, confidence=95):
    """
    Plot smoothed (1-second averaged) temperature estimate and difference.
    """
    os.makedirs(folder, exist_ok=True)
    k_val = norm.ppf(0.5 + confidence / 200)
    T_est_smooth = average_per_second(T_est, steps_psec)
    T_true_smooth = average_per_second(T_true, steps_psec)
    var_smooth = average_per_second(sigma**2, steps_psec)
    sigma_smooth = np.sqrt(var_smooth)
    t_min_smooth = np.arange(len(T_est_smooth)) / 60

    # Smoothed estimate with CI
    plt.figure()
    plt.plot(t_min_smooth, T_est_smooth, color='tab:blue', label='Smoothed estimate')
    plt.fi
