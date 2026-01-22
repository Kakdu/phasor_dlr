import numpy as np
from phasor_dlr.estimation.error_propagation.sampling import sample_error_intervals

def monte_carlo_multi(
    V_amp,
    V_phi,
    I_amp,
    I_phi,
    *,
    intervals_r_I,
    intervals_r_V,
    intervals_theta_I,
    intervals_theta_V,
    NSAMP,
    seed,
    equipment_error_mode=False,
    r_I_var=None,
    r_V_var=None,
    th_I_var=None,
    th_V_var=None,
):
    rng = np.random.default_rng(seed)
    M = len(V_amp)

    r_I = sample_error_intervals(
        intervals_r_I, M, NSAMP,
        equipment_error_mode=equipment_error_mode,
        measurement_variances=r_I_var,
        multiplicative=True,
        rng=rng,
    )
    r_V = sample_error_intervals(
        intervals_r_V, M, NSAMP,
        equipment_error_mode=equipment_error_mode,
        measurement_variances=r_V_var,
        multiplicative=True,
        rng=rng,
    )
    th_I = sample_error_intervals(
        intervals_theta_I, M, NSAMP,
        equipment_error_mode=equipment_error_mode,
        measurement_variances=th_I_var,
        rng=rng,
    )
    th_V = sample_error_intervals(
        intervals_theta_V, M, NSAMP,
        equipment_error_mode=equipment_error_mode,
        measurement_variances=th_V_var,
        rng=rng,
    )

    A_I = I_amp[:, None] * r_I
    A_V = V_amp[:, None] * r_V
    phi_I_eff = I_phi[:, None] + th_I
    phi_V_eff = V_phi[:, None] + th_V

    S_samples = A_V * A_I * np.cos(phi_V_eff - phi_I_eff)
    S0 = (V_amp * I_amp * np.cos(V_phi - I_phi))[:, None]

    return {
        "S_samples": S_samples,
        "DeltaS_abs": S_samples - S0,
        "DeltaS_rel": (S_samples - S0) / S0,
        "I_amp": A_I,
        "V_amp": A_V,
        "I_theta": phi_I_eff,
        "V_theta": phi_V_eff
    }

