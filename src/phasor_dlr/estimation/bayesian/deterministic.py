import pymc as pm

def define_deterministic(V1_true, I1_true, phiV1_true, phiI1_true,
                         V2_pred, I2_pred, phiV2_pred, phiI2_pred,
                         C_factor, T_ref, alpha, tP=0):
    
    dphi1 = phiV1_true - phiI1_true
    dphi2 = phiV2_pred - phiI2_pred

    # Power loss
    P_loss = pm.Deterministic("P_loss", V2_pred * I2_pred * pm.math.cos(dphi2) - V1_true * I1_true * pm.math.cos(dphi1))

    # AC current
    I_AC = pm.math.sqrt(0.5 * (I1_true**2 + I2_pred**2))

    # Temperature
    T_AV = pm.Deterministic("T_AV", T_ref + C_factor * (P_loss / I_AC**2) - 1 / alpha)

    return T_AV, P_loss, I_AC