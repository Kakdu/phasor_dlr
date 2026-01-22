import numpy as np


def combined_uniforms_sigma(intervals_dict, eps=1e-12):
    """
    Compute combined standard deviation for uniform distributions given as a dictionary.

    Parameters
    ----------
    intervals_dict : dict
        Dictionary of intervals, e.g. {'CU': (-0.01, 0.01), 'CS': (-0.02, 0.02)}
    eps : float
        Small number for numerical stability

    Returns
    -------
    float
        Combined standard deviation of the uniform distributions
    """
    var_sum = sum(((b - a) ** 2) / 12.0 for a, b in intervals_dict.values())
    return float(np.sqrt(var_sum) + eps)


def combined_uniforms_logsigma(rel_intervals_dict, eps=1e-12):
    """
    Compute combined standard deviation of log(1 + u_i),
    where u_i ~ Uniform(a_i, b_i) for each key in the dictionary.

    Parameters
    ----------
    rel_intervals_dict : dict
        Dictionary of relative error bounds, e.g. {'CU': (-0.01, 0.01)}
    eps : float
        Numerical stabilizer

    Returns
    -------
    float
        sqrt(sum Var(log(1 + u_i)))
    """
    var_sum = 0.0

    for a_rel, b_rel in rel_intervals_dict.values():
        # Convert relative → multiplicative
        a = 1.0 + a_rel
        b = 1.0 + b_rel

        if a <= 0 or b <= 0:
            raise ValueError("Relative bounds must satisfy 1 + a > 0 and 1 + b > 0.")

        loga = np.log(a)
        logb = np.log(b)

        # Expected value of log(epsilon) for uniform
        mu = (b * logb - b - a * loga + a) / (b - a)

        # Second moment
        second = (b * (logb**2 - 2 * logb + 2) - a * (loga**2 - 2 * loga + 2)) / (b - a)

        var_log = second - mu**2
        var_sum += var_log

    return float(np.sqrt(max(var_sum, 0.0)) + eps)


def turn_on_exp(x, x0=0.0, tau=1.0, A=1.0):
    return A * (1 - np.exp(-(x - x0) / tau)) * (x > x0)