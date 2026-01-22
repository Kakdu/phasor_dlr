import numpy as np
from SALib.sample import sobol as sl
from phasor_dlr.config.standards import make_error_intervals

def flatten_error_intervals(error_intervals):
    """
    Convert ErrorIntervals dataclass into a flat list of intervals in SALib order:
    [r_s..., r_r..., theta_s..., theta_r...]
    """
    # Order of variables for s and r
    r_s = list(error_intervals.r_I.values())
    r_r = list(error_intervals.r_V.values())
    theta_s = list(error_intervals.theta_I.values())
    theta_r = list(error_intervals.theta_V.values())

    all_intervals = r_s + r_r + theta_s + theta_r
    return all_intervals

def build_names(error_intervals):
    """
    Build plain variable names for SALib in same order as flatten_error_intervals
    """
    # r and theta keys
    r_keys = list(error_intervals.r_I.keys())
    r_keys_r = list(error_intervals.r_V.keys())
    theta_keys = list(error_intervals.theta_I.keys())
    theta_keys_r = list(error_intervals.theta_V.keys())

    names_plain = (
        [f"r{k}_s" for k in r_keys] +
        [f"r{k}_r" for k in r_keys_r] +
        [f"theta{k}_s" for k in theta_keys] +
        [f"theta{k}_r" for k in theta_keys_r]
    )
    return names_plain

def generate_sobol_samples_from_standards(CT_class, VT_class, N_samples, calc_second_order):
    # 1) Build base error intervals
    error_intervals = make_error_intervals(CT_class, VT_class)

    # 2) Build ordered intervals for Sobol: r_s, r_r, theta_s, theta_r
    # r_I: CU, CS
    # r_V: VU, VS1, VS2
    intervals_r_s = list(error_intervals.r_I.values()) + list(error_intervals.r_V.values())
    intervals_r_r = intervals_r_s.copy()  # duplicate for received

    intervals_theta_s = list(error_intervals.theta_I.values()) + list(error_intervals.theta_V.values())
    intervals_theta_r = intervals_theta_s.copy()

    all_intervals = intervals_r_s + intervals_r_r + intervals_theta_s + intervals_theta_r

    # 3) Build names plain (matching the old order)
    names_plain = []
    # Magnitudes
    for suffix in ["_s", "_r"]:
        for k in ["VU", "VT", "VS1", "VS2", "CU", "CT", "CS"]:
            if k in error_intervals.r_V:
                names_plain.append(f"r{k}{suffix}")
            elif k in error_intervals.r_I:
                names_plain.append(f"r{k}{suffix}")
    # Angles
    for suffix in ["_s", "_r"]:
        for k in ["VU", "VT", "VS1", "VS2", "CU", "CT", "CS"]:
            if k in error_intervals.theta_V:
                names_plain.append(f"theta{k}{suffix}")
            elif k in error_intervals.theta_I:
                names_plain.append(f"theta{k}{suffix}")

    # 4) Build SALib problem
    problem = {
        "num_vars": len(all_intervals),
        "names": names_plain,
        "bounds": all_intervals
    }

    # 5) Generate Sobol samples
    param_values = sl.sample(problem, N_samples, calc_second_order=calc_second_order)

    return param_values, problem, error_intervals
