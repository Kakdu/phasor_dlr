def pi_model_sending_phasor(V_r_amp, V_r_angle, I_r_amp, I_r_angle, T, condParameter):
    R = (
        condParameter["L"]
        * condParameter["R_20"]
        * (1 + condParameter["alpha"] * (T - condParameter["T_ref"]))
    )
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
    V_s_amp, V_s_angle = np.abs(V_s), np.angle(V_s)
    I_s_amp, I_s_angle = np.abs(I_s_total), np.angle(I_s_total)
    return V_s_amp, V_s_angle, I_s_amp, I_s_angle
