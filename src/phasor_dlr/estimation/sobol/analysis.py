import numpy as np

def T_A_model(samples, params):
    """
    Temperature model for Sobol evaluation
    """
    n_x = params["n_x"]

    r_s     = samples[:, :n_x]
    r_r     = samples[:, n_x:2*n_x]
    theta_s = samples[:, 2*n_x:3*n_x]
    theta_r = samples[:, 3*n_x:4*n_x]

    A_s = np.prod(1 + r_s, axis=1)
    A_r = np.prod(1 + r_r, axis=1)

    Theta_s = np.sum(theta_s, axis=1)
    Theta_r = np.sum(theta_r, axis=1)

    P_s = params["nu_s"] * params["iota_s"] * A_s * np.cos(params["phi_s"] + Theta_s)
    P_r = params["nu_r"] * params["iota_r"] * A_r * np.cos(params["phi_r"] + Theta_r)
    P_meas = P_s - P_r

    scale = (1 / (params["k_j"] * params["R_AC"] * params["I_AC"]**2) - 1) / params["alpha"]
    return params["T_ref"] + scale * P_meas

def build_model_params(condParam, V_s_amp, V_s_angle, I_s_amp, I_s_angle,
                       V_r_amp, V_r_angle, I_r_amp, I_r_angle, n_x=7, k_j=1.0):
    """
    Prepare parameter dictionary for T_A_model
    """
    I_AC = (I_r_amp + I_s_amp) / 2
    return {
        "T_ref": condParam["T_ref"],
        "alpha": condParam["alpha"],
        "k_j": k_j,
        "R_AC": condParam["R_20"] * condParam["L"],
        "I_AC": I_AC,
        "nu_s": V_s_amp,
        "iota_s": I_s_amp,
        "phi_s": V_s_angle - I_s_angle,
        "nu_r": V_r_amp,
        "iota_r": I_r_amp,
        "phi_r": V_r_angle - I_r_angle,
        "n_x": n_x
    }


