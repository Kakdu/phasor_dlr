import numpy as np

from phasor_dlr.models.pi_model_static import pi_model_sending_phasor
from phasor_dlr.models.phasors import phasor

def ukf_predict_measurement(chi_pred, Wm, Wc, condParameter):
    """
    Map sigma points to measurement space and compute predicted mean and covariance.
    """
    n_sigma, n = chi_pred.shape
    m = 6
    Z_sigma = np.zeros((n_sigma, m))

    for i in range(n_sigma):
        T_i, V_r_amp_i, I_r_amp_i, received_angle_diff_i, sent_angle_diff_i = chi_pred[i]
        V_r_i = phasor(V_r_amp_i, 0.0)
        I_r_i = phasor(I_r_amp_i, -received_angle_diff_i)

        V_s_amp_i, V_s_angle_i, I_s_amp_i, I_s_angle_i = pi_model_sending_phasor(
            V_r_i.amplitude, V_r_i.angle, I_r_i.amplitude, I_r_i.angle, T_i, condParameter
        )

        Z_sigma[i] = [V_r_i.amplitude, I_r_i.amplitude, V_s_amp_i, I_s_amp_i, received_angle_diff_i, sent_angle_diff_i]

    z_pred = np.sum(Wm[:,None]*Z_sigma, axis=0)
    P_zz = sum(Wc[i]*np.outer(Z_sigma[i]-z_pred, Z_sigma[i]-z_pred) for i in range(n_sigma))
    return Z_sigma, z_pred, P_zz


def ukf_update(x_pred, chi_pred, z_pred, Z_sigma, P_zz, z_meas, Wc, Q, gamma, eta_th=10.6):
    """
    UKF update with adaptive Q scaling.
    """
    n = len(x_pred)
    P_xz = sum(Wc[i]*np.outer(chi_pred[i]-x_pred, Z_sigma[i]-z_pred) for i in range(2*n+1))
    K = P_xz @ np.linalg.inv(P_zz)

    x_est = x_pred + K @ (z_meas - z_pred)

    # Adaptive Q
    innovation = z_meas - z_pred
    eta_k = innovation.T @ np.linalg.inv(P_zz) @ innovation
    Q_scaled = gamma*Q if eta_k > eta_th else Q

    # Predict covariance
    P_pred = Q_scaled + sum(Wc[i]*np.outer(chi_pred[i]-x_pred, chi_pred[i]-x_pred) for i in range(2*n+1))
    P = P_pred - K @ P_zz @ K.T

    return x_est, P, K
