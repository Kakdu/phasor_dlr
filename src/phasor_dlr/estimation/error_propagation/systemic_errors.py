from scipy.optimize import root_scalar
from phasor_dlr.estimation.error_propagation.temperature import temperature_distribution

def calibrate_kj(P_loss, I_AC, T_nom, *, cond, T_ref=20.0):
    def f(kj):
        _, _, T_avg = temperature_distribution(
            P_loss, I_AC, cond=cond, kj=kj, T_ref=T_ref
        )
        return T_avg.mean() - T_nom

    sol = root_scalar(f, bracket=[1e-6, 10.0], method="bisect")
    if not sol.converged:
        raise RuntimeError("kj calibration failed")
    return sol.root
