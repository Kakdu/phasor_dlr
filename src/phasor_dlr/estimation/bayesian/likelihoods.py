import pymc as pm
import numpy as np

def define_likelihoods(V1_true, I1_true, phiV1_true, phiI1_true,
                       V2_pred, I2_pred, phiV2_pred, phiI2_pred,
                       V_r_meas_array, I_r_meas_array,
                       V_s_meas_array, I_s_meas_array,
                       V_r_angle_meas_array, I_r_angle_meas_array,
                       V_s_angle_meas_array, I_s_angle_meas_array,
                       sigma_logV, sigma_logI, sigma_phiV, sigma_phiI):
    
    # Receiving-end
    pm.Normal("V1_obs", mu=pm.math.log(V1_true), sigma=sigma_logV, observed=np.log(V_r_meas_array))
    pm.Normal("I1_obs", mu=pm.math.log(I1_true), sigma=sigma_logI, observed=np.log(I_r_meas_array))
    pm.Normal("phiV1_obs", mu=phiV1_true, sigma=sigma_phiV, observed=V_r_angle_meas_array)
    pm.Normal("phiI1_obs", mu=phiI1_true, sigma=sigma_phiI, observed=I_r_angle_meas_array)

    # Sending-end
    pm.Normal("V2_obs", mu=pm.math.log(V2_pred), sigma=sigma_logV, observed=np.log(V_s_meas_array))
    pm.Normal("I2_obs", mu=pm.math.log(I2_pred), sigma=sigma_logI, observed=np.log(I_s_meas_array))
    pm.Normal("phiV2_obs", mu=phiV2_pred, sigma=sigma_phiV, observed=V_s_angle_meas_array)
    pm.Normal("phiI2_obs", mu=phiI2_pred, sigma=sigma_phiI, observed=I_s_angle_meas_array)