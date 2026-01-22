import numpy as np

def temperature_from_phasors(V_r, I_r, V_s, I_s, cond):
    I_ac = np.sqrt(0.5 * (I_s.amplitude**2 + I_r.amplitude**2))

    num = (
        V_s.amplitude * I_s.amplitude * np.cos(V_s.angle - I_r.angle)
        - V_r.amplitude * I_r.amplitude * np.cos(V_r.angle - I_r.angle)
    )

    den = I_ac**2 * cond["R_20"] * cond["L"]

    return cond["T_ref"] + (num / den - 1) / cond["alpha"]


def temperature_distribution(P_loss, I_AC, *, cond, kj=1.0):
    alpha = cond["alpha"]
    L = cond["L"]
    R20 = cond["R_20"]

    T = cond["T_ref"] + (1 / alpha) * (P_loss / (kj * L * R20 * I_AC**2) - 1)
    return T, T.max(axis=0), T.mean(axis=0)
