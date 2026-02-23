import numpy as np
from concurrent.futures import ProcessPoolExecutor

from phasor_dlr.utils.math import combined_uniforms_sigma, combined_uniforms_logsigma
from phasor_dlr.config.defaults import condParameters
from phasor_dlr.config.standards import make_error_intervals
from phasor_dlr.synthetic_data.noise import apply_error_to_phasors


# --------------------------
# π-model sending phasor
# --------------------------
def pi_model_sending_phasor(V_r_amp, V_r_angle, I_r_amp, I_r_angle, T, condParameter):
    R = condParameter["L"] * condParameter["R_20"] * (1 + condParameter["alpha"] * (T - condParameter["T_ref"]))
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
    return np.abs(V_s), np.angle(V_s), np.abs(I_s_total), np.angle(I_s_total)


# --------------------------
# Sigmoid for temperature events
# --------------------------
def turn_on_exp(x, x0=0.0, tau=1.0, A=1.0):
    return A * (1 - np.exp(-(x - x0) / tau)) * (x > x0)


# --------------------------
# UKF single case
# --------------------------
def run_case(minutes=100, steps_psec=100, seed=4,
             conv=False, curr=False, cond_name="bohus",
             CT_class="5P", VT_class="0.2", confidence=95):
    # --------------------------
    # Conductor parameters
    # --------------------------
    condParameter = condParameters[cond_name]


    # -------------------------------
    # Measurement accuracy and variance
    # -------------------------------
    accuracy_r = make_error_intervals(CT_class, VT_class)
    accuracy_s = make_error_intervals(CT_class, VT_class)

    sigma_V = combined_uniforms_logsigma(accuracy_r.r_V)
    sigma_I = combined_uniforms_logsigma(accuracy_r.r_I)
    sigma_phiV = combined_uniforms_sigma(accuracy_r.theta_V)
    sigma_phiI = combined_uniforms_sigma(accuracy_r.theta_I)



    # --------------------------
    # UKF parameters with log amplitudes
    # --------------------------
    minutes = minutes
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

    if conv == 1:
        startConv = steps // 4
        tauConv = steps_psec * 60 * 8
        T_true -= turn_on_exp(x, startConv, tau=tauConv, A=10)
    if curr == 1:
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


    rmse_T = np.sqrt(np.mean((T_est_list - T_true)**2))
    mae_T = np.mean(np.abs(T_est_list - T_true))
    var_T = np.mean(sigma_list[:,0])

    return rmse_T, mae_T, var_T



# --------------------------
# Helper to submit one stand
# --------------------------
def run_stand(curr, conv, minutes, stand):
    """
    Run a single combination and return RMSE, MAE, variance for temperature.
    """
    rmse, mae, var = run_case(
        minutes=minutes,
        conv=conv,
        curr=curr,
        cond_name="bohus",
        CT_class=stand,
    )

    return (curr, conv, minutes, stand), (rmse, mae, var)


# --------------------------
# Sweep function with efficient parallelization
# --------------------------
def sweep_all_cases(cases, minutes_list, stands):
    tasks = [(curr, conv, minutes, stand)
             for _, curr, conv in cases
             for minutes in minutes_list
             for stand in stands]

    results_dict = {}
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_stand, *task) for task in tasks]
        for future in futures:
            key, metrics = future.result()
            results_dict[key] = metrics

    # Print LaTeX-style table
    print("\n--- Sweep Results ---\n")
    for case_name, curr, conv in cases:
        for minutes in minutes_list:
            results = {stand: results_dict[(curr, conv, minutes, stand)] for stand in stands}
            print(
                f"{case_name} & {minutes} & "
                f"{results['5P'][0]:.2f} & {results['5P'][1]:.2f} & {results['5P'][2]:.2f} & "
                f"{results['0.1'][0]:.2f} & {results['0.1'][1]:.2f} & {results['0.1'][2]:.2f} \\\\"
            )

# --------------------------
# Run sweep
# --------------------------
cases = [
    ("Baseline", 0, 0),
    ("Convection", 0, 1),
    ("Current", 1, 0),
    ("Combination", 1, 0),
]

minutes_list = [100, 10, 1]
stands = ["5P", "0.1"]


if __name__ == "__main__":
    sweep_all_cases(cases, minutes_list, stands)
