import pymc as pm


from phasor_dlr.estimation.bayesian.priors import phys_priors,  engineer_priors
from phasor_dlr.estimation.bayesian.likelihoods import define_likelihoods
from phasor_dlr.estimation.bayesian.deterministic import define_deterministic
from phasor_dlr.estimation.bayesian.soft_penalty import soft_temperature_bounds


def build_temperature_model(
    condParameter, V_r_nom, I_r_nom, V_r_angle_nom, I_r_angle_nom,
    V_s_nom, I_s_nom, V_s_angle_nom, I_s_angle_nom,
    V_r_meas_array, I_r_meas_array, V_r_angle_meas_array, I_r_angle_meas_array,
    V_s_meas_array, I_s_meas_array, V_s_angle_meas_array, I_s_angle_meas_array,
    sigma_logV, sigma_logI, sigma_phiV, sigma_phiI, w, C, tP=False, phys_informed=True, seed=None
):
    with pm.Model() as model:
        # 1. Priors
        if phys_informed == True:
            priors = phys_priors(V_r_nom, I_r_nom, V_r_angle_nom, I_r_angle_nom,  sigma_phiV, sigma_phiI, condParameter, w)

            # Access priors by name
            T_model = priors["T_model"]
            Vr_true = priors["Vr_true"]
            Ir_true = priors["Ir_true"]
            Vs_pred = priors["Vs_pred"]
            Is_pred = priors["Is_pred"]
            thetaVr_true = priors["thetaVr_true"]
            thetaIr_true = priors["thetaIr_true"]
            thetaVs_pred = priors["thetaVs_pred"]
            thetaIs_pred = priors["thetaIs_pred"]

        else:
            priors = engineer_priors(V_r_nom, I_r_nom, V_s_nom, I_s_nom, V_r_angle_nom, I_r_angle_nom, V_s_angle_nom, I_s_angle_nom,  sigma_phiV, sigma_phiI)

            # Access priors by name
            Vr_true = priors["Vr_true"]
            Ir_true = priors["Ir_true"]
            Vs_pred = priors["Vs_pred"]
            Is_pred = priors["Is_pred"]
            thetaVr_true = priors["thetaVr_true"]
            thetaIr_true = priors["thetaIr_true"]
            thetaVs_pred = priors["thetaVs_pred"]
            thetaIs_pred = priors["thetaIs_pred"]

        # 3. Likelihoods
        define_likelihoods(
            Vr_true, Ir_true, thetaVr_true, thetaIr_true,
            Vs_pred, Is_pred, thetaVs_pred, thetaIs_pred,
            V_r_meas_array, I_r_meas_array,
            V_s_meas_array, I_s_meas_array,
            V_r_angle_meas_array, I_r_angle_meas_array,
            V_s_angle_meas_array, I_s_angle_meas_array,
            sigma_logV, sigma_logI, sigma_phiV, sigma_phiI
        )

        # 4. Deterministic relations
        C_factor = 1.0 / (condParameter["alpha"] * condParameter["L"] * condParameter["R_20"])
        T_AV, P_loss, I_AC = define_deterministic(
            Vr_true, Ir_true, thetaVr_true, thetaIr_true,
            Vs_pred, Is_pred, thetaVs_pred, thetaIs_pred,
            C_factor, condParameter["T_ref"], condParameter["alpha"]
        )

        soft_temperature_bounds(T_AV, lower=30.0, upper=70.0, scale=5.0, name="T_A_soft_bounds", enabled=tP)
    return model


def run_model_for_N(
    N,
    seed,
    condParam,
    V_r_amp,
    V_r_angle,
    I_r_amp,
    I_r_angle,
    Z,
    Y,
    accuracy_r,
    accuracy_s,
    sigma_logV,
    sigma_logI,
    sigma_phiV,
    sigma_phiI,
    w,
    C,
    T_nom,
    tP=False,
    phys_informed=True,
    levels=[0.95]
):
    from phasor_dlr.synthetic_data.generators import generate_static_measurements

    # -------------------------------
    # Generate measurements
    # -------------------------------
    (
        V_r_amp_meas, V_r_angle_meas,
        I_r_amp_meas, I_r_angle_meas,
        V_s_amp_meas, V_s_angle_meas,
        I_s_amp_meas, I_s_angle_meas,
        V_s_amp, V_s_angle,
        I_s_amp, I_s_angle
    ) = generate_static_measurements(
        V_r_amp, V_r_angle,
        I_r_amp, I_r_angle,
        Z, Y,
        N,
        accuracy_r,
        accuracy_s,
        seed_r=seed,
        seed_s=seed+1
    )

    # -------------------------------
    # Build model
    # -------------------------------
    model = build_temperature_model(
        condParameter=condParam,
        V_r_nom=V_r_amp,
        I_r_nom=I_r_amp,
        V_r_angle_nom=V_r_angle,
        I_r_angle_nom=I_r_angle,
        V_s_nom=V_s_amp,
        I_s_nom=I_s_amp,
        V_s_angle_nom=V_s_angle,
        I_s_angle_nom=I_s_angle,
        V_r_meas_array=V_r_amp_meas,
        I_r_meas_array=I_r_amp_meas,
        V_r_angle_meas_array=V_r_angle_meas,
        I_r_angle_meas_array=I_r_angle_meas,
        V_s_meas_array=V_s_amp_meas,
        I_s_meas_array=I_s_amp_meas,
        V_s_angle_meas_array=V_s_angle_meas,
        I_s_angle_meas_array=I_s_angle_meas,
        sigma_logV=sigma_logV,
        sigma_logI=sigma_logI,
        sigma_phiV=sigma_phiV,
        sigma_phiI=sigma_phiI,
        w=w,
        C=C,
        tP=tP,
        phys_informed=phys_informed,
        seed=seed
    )

    # -------------------------------
    # Sample
    # -------------------------------

    with model:
        trace = pm.sample(
            draws=4000,
            tune=1000,
            chains=4,
            target_accept=0.95,
            random_seed=seed,
            progressbar=False
        )

    samples = trace.posterior["T_AV"].values.reshape(-1)

    import numpy as np
    rmse = np.sqrt(np.mean((samples - T_nom) ** 2))

    import arviz as az
    hdi_widths = {}
    for l_idx, level in enumerate(levels):
        hdi_vals = az.hdi(trace, var_names=["T_AV"], hdi_prob=level)["T_AV"].values
        hdi_widths[level] = 0.5 * (hdi_vals[1] - hdi_vals[0])


    return {
        "samples": samples,
        "rmse": rmse,
        "hdi": hdi_widths,
        "trace": trace
    }

