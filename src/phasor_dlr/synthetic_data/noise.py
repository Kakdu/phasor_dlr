import numpy as np

def relative_noise(attr_dict, N):
    """Multiplicative amplitude noise from dict of (emin, emax) tuples."""
    factor = np.ones(N)
    for emin, emax in attr_dict.values():
        factor *= 1.0 + np.random.uniform(emin, emax, N)
    return factor

def additive_noise(attr_dict, N):
    """Additive angle noise from dict of (emin, emax) tuples."""
    total = np.zeros(N)
    for emin, emax in attr_dict.values():
        total += np.random.uniform(emin, emax, N)
    return total


def apply_error_to_phasors(
    V_amp, V_angle,
    I_amp, I_angle,
    STANDARDS,
    N=None,
    seed=1
):
    """
    Apply measurement errors to voltage and current phasors for a single side (sending or receiving).

    Parameters
    ----------
    V_amp, I_amp : np.ndarray
        Voltage and current amplitudes
    V_angle, I_angle : np.ndarray
        Voltage and current angles [rad]
    STANDARDS : ErrorIntervals
        ErrorIntervals object for this side (r or s)
    N : int
        Number of samples (defaults to length of V_amp)
    seed : int
        Random seed

    Returns
    -------
    V_amp_n, V_angle_n, I_amp_n, I_angle_n : np.ndarray
        Phasors with measurement errors applied
    """

    rng = np.random.default_rng(seed)

    if N is None:
        N = len(V_amp)

    V_amp_n = V_amp * relative_noise(STANDARDS.r_V, N)
    V_angle_n = V_angle + additive_noise(STANDARDS.theta_V, N)

    I_amp_n = I_amp * relative_noise(STANDARDS.r_I, N)
    I_angle_n = I_angle + additive_noise(STANDARDS.theta_I, N)

    return V_amp_n, V_angle_n, I_amp_n, I_angle_n
