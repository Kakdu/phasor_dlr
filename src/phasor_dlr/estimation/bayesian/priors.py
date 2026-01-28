import pymc as pm
import pytensor.tensor as pt
import numpy as np

from phasor_dlr.models.pi_model_dynamic import pi_model_sending_phasor_pytensor

def engineer_priors(V_r_nom, I_r_nom, V_s_nom, I_s_nom, V_r_angle_nom, I_r_angle_nom, V_s_angle_nom, I_s_angle_nom,  sigma_phiV, sigma_phiI):
    """
    Return dict of priors for the model.
    """
    priors = {}

    # Non-physics-informed priors
    priors["Vr_true"] = pm.Uniform("Vr_true", lower=0.90 * V_r_nom, upper=1.1 * V_r_nom)
    priors["Ir_true"] = pm.Uniform("Ir_true", lower=0.90 * I_r_nom, upper=1.1 * I_r_nom)
    priors["Vs_pred"] = pm.Uniform("Vs_pred", lower=0.90 * V_s_nom, upper=1.1 * V_s_nom)
    priors["Is_pred"] = pm.Uniform("Is_pred", lower=0.90 * I_s_nom, upper=1.1 * I_s_nom)
    priors["thetaVr_true"] = pm.Normal("thetaVr_true", mu=V_r_angle_nom, sigma=sigma_phiV)
    priors["thetaIr_true"] = pm.Normal("thetaIr_true", mu=I_r_angle_nom, sigma=sigma_phiI)
    priors["thetaVs_pred"] = pm.Normal("phiVs_pred", mu=V_s_angle_nom, sigma=sigma_phiV)
    priors["thetaIs_pred"] = pm.Normal("phiIs_pred", mu=I_s_angle_nom, sigma=sigma_phiI)

    return priors

def phys_priors(V_r_nom, I_r_nom, V_r_angle_nom, I_r_angle_nom, sigma_phiV, sigma_phiI, condParameter, w): 

    priors = {}

    # Latent temperature
    priors["T_model"] = pm.SkewNormal("T_model", mu=70, sigma=10.0, alpha=-5)

    # Receiving-end phasor priors
    priors["Vr_true"] = pm.Uniform("Vr_true", lower=0.95 * V_r_nom, upper=1.05 * V_r_nom)
    priors["Ir_true"] = pm.Uniform("Ir_true", lower=0.90 * I_r_nom, upper=1.10 * I_r_nom)
    priors["thetaVr_true"] = pm.Normal("thetaVr_true", mu=V_r_angle_nom, sigma=sigma_phiV)
    priors["thetaIr_true"] = pm.Normal("thetaIr_true", mu=I_r_angle_nom, sigma=sigma_phiI)

    # --- Temperature-dependent impedance ---
    alpha = condParameter["alpha"]
    R_20 = condParameter["R_20"]
    L = condParameter["L"]
    X = condParameter["X"] * L
    T_ref = condParameter["T_ref"]
    C = condParameter["C"]

    R_T = L * R_20 * (1 + alpha * (priors["T_model"] - T_ref))

    priors["Vs_pred"], priors["thetaVs_pred"], priors["Is_pred"], priors["thetaIs_pred"] = pi_model_sending_phasor_pytensor(
        priors["Vr_true"], priors["thetaVr_true"], 
        priors["Ir_true"], priors["thetaIr_true"], 
        R_T, X, 
        0, w * C
    )

    return priors
