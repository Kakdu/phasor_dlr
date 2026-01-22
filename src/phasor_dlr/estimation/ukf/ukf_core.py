import numpy as np
from dataclasses import dataclass

from phasor_dlr.estimation.ukf.sigma_points import generate_sigma_points
from phasor_dlr.models.phasors import Phasor
from phasor_dlr.models.pi_model_static import sending_end_phasor


# ============================
# Result container
# ============================

@dataclass
class UKFResult:
    X_est: np.ndarray
    P_diag: np.ndarray
    K_norm: np.ndarray
    K_state_norms: np.ndarray


# ============================
# Core UKF routine
# ============================

def run_ukf(
    measurements: dict,
    cond: dict,
    ukf_params: dict,
    x0: np.ndarray,
    P0: np.ndarray,
):
    """
    Run Unscented Kalman Filter for conductor temperature estimation.

    State vector:
        x = [T, V_r_amp, I_r_amp, phi_r, phi_s]

    Measurement vector:
        z = [V_r, I_r, V_s, I_s, phi_r, phi_s]
    """

    # --- Unpack ---
    z_meas = measurements["z"]
    R = measurements["R"]

    Q = ukf_params["Q"]
    gamma = ukf_params["gamma"]
    eta_th = ukf_params["eta_threshold"]

    alpha = ukf_params["alpha"]
    beta = ukf_params["beta"]
    kappa = ukf_params["kappa"]

    n = len(x0)
    steps = z_meas.shape[0]

    # --- UKF weights ---
    lambda_ = alpha**2 * (n + kappa) - n
    Wm = np.full(2*n + 1, 0.5 / (n + lambda_))
    Wc = Wm.copy()
    Wm[0] = lambda_ / (n + lambda_)
    Wc[0] = Wm[0] + (1 - alpha**2 + beta)

    # --- Storage ---
    X_est = np.zeros((steps, n))
    P_diag = np.zeros((steps, n))
    K_norm = np.zeros(steps)
    K_state_norms = np.zeros((steps, n))

    # --- Initialize ---
    x = x0.copy()
    P = P0.copy()

    # ============================
    # Time loop
    # ============================
    for k in range(steps):

        # ------------------------
        # Sigma points
        # ------------------------
        chi = generate_sigma_points(x, P, lambda_)

        # ------------------------
        # Prediction (random walk)
        # ------------------------
        chi_pred = chi.copy()
        x_pred = np.sum(Wm[:, None] * chi_pred, axis=0)

        # ------------------------
        # Measurement prediction
        # ------------------------
        Z_sigma = np.zeros((2*n + 1, 6))

        for i in range(2*n + 1):
            T_i, V_r_amp_i, I_r_amp_i, phi_r_i, phi_s_i = chi_pred[i]

            V_r = Phasor(V_r_amp_i, 0.0)
            I_r = Phasor(I_r_amp_i, -phi_r_i)

            V_s_amp, V_s_ang, I_s_amp, I_s_ang = sending_end_phasor(
                V_r, I_r, T_i, cond
            )

            Z_sigma[i] = [
                V_r_amp_i,
                I_r_amp_i,
                V_s_amp,
                I_s_amp,
                phi_r_i,
                phi_s_i,
            ]

        z_pred = np.sum(Wm[:, None] * Z_sigma, axis=0)

        P_zz = R.copy()
        P_xz = np.zeros((n, 6))

        for i in range(2*n + 1):
            dz = Z_sigma[i] - z_pred
            dx = chi_pred[i] - x_pred
            P_zz += Wc[i] * np.outer(dz, dz)
            P_xz += Wc[i] * np.outer(dx, dz)

        # ------------------------
        # Update
        # ------------------------
        z = z_meas[k]
        K = P_xz @ np.linalg.inv(P_zz)

        innovation = z - z_pred
        x = x_pred + K @ innovation

        # Diagnostics
        K_norm[k] = np.linalg.norm(K, ord="fro")
        for i in range(n):
            K_state_norms[k, i] = np.linalg.norm(K[i, :])

        # ------------------------
        # Adaptive process noise
        # ------------------------
        eta_k = innovation.T @ np.linalg.inv(P_zz) @ innovation
        Q_eff = gamma * Q if eta_k > eta_th else Q

        # ------------------------
        # Covariance update
        # ------------------------
        P_pred = Q_eff.copy()
        for i in range(2*n + 1):
            dx = chi_pred[i] - x_pred
            P_pred += Wc[i] * np.outer(dx, dx)

        P = P_pred - K @ P_zz @ K.T

        # ------------------------
        # Store
        # ------------------------
        X_est[k] = x
        P_diag[k] = np.diag(P)

    return UKFResult(
        X_est=X_est,
        P_diag=P_diag,
        K_norm=K_norm,
        K_state_norms=K_state_norms,
    )
