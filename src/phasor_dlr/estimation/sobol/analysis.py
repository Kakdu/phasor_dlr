import numpy as np

def build_index_map(problem):
    """
    Maps variable names to column indices in samples.
    """
    return {name: i for i, name in enumerate(problem["names"])}


def T_A_model(samples, params, problem):
    """
    Temperature model for Sobol evaluation (name-based, order-safe)
    """
    n_x = params["n_x"]
    index = build_index_map(problem)

    # -------------------------------------------------
    # Collect variables explicitly by name
    # -------------------------------------------------
    def collect(var_type, suffix):
        """
        var_type: "r" or "theta"
        suffix: "_s" or "_r"
        """
        keys = [
            name for name in problem["names"]
            if name.startswith(var_type) and name.endswith(suffix)
        ]

        # Stable ordering inside block
        keys = sorted(keys)

        assert len(keys) == n_x, (
            f"Expected {n_x} variables for {var_type}{suffix}, got {len(keys)}"
        )

        cols = [index[k] for k in keys]
        return samples[:, cols]

    r_s     = collect("r", "_s")
    r_r     = collect("r", "_r")
    theta_s = collect("theta", "_s")
    theta_r = collect("theta", "_r")

    # -------------------------------------------------
    # Physics
    # -------------------------------------------------
    A_s = np.prod(1 + r_s, axis=1)
    A_r = np.prod(1 + r_r, axis=1)

    Theta_s = np.sum(theta_s, axis=1)
    Theta_r = np.sum(theta_r, axis=1)

    P_s = params["nu_s"] * params["iota_s"] * A_s * np.cos(params["phi_s"] + Theta_s)
    P_r = params["nu_r"] * params["iota_r"] * A_r * np.cos(params["phi_r"] + Theta_r)
    P_meas = P_s - P_r

    scale = (
        (1 / (params["k_j"] * params["R_AC"] * params["I_AC"]**2) - 1)
        / params["alpha"]
    )

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


