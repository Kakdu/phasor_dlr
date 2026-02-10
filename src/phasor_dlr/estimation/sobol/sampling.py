import numpy as np
from dataclasses import fields
from SALib.sample import sobol as sl
from phasor_dlr.config.standards import make_error_intervals

def iter_error_intervals(error_intervals):
    """
    Yields (var_type, subkey, interval_dict) tuples
    e.g. ("r", "VU", {...})
    """
    for f in fields(error_intervals):
        var_type, _ = f.name.split("_", 1)  # "r" or "theta"
        interval_dict = getattr(error_intervals, f.name)
        for subkey, interval in interval_dict.items():
            yield var_type, subkey, interval

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

def generate_sobol_samples_from_standards(
    CT_class,
    VT_class,
    N_samples,
    calc_second_order,
):
    error_intervals = make_error_intervals(CT_class, VT_class)

    bounds = []
    names = []

    # Explicit semantic ordering
    for var_type in ["r", "theta"]:
        for suffix in ["_s", "_r"]:
            for f in fields(error_intervals):
                f_var_type, _ = f.name.split("_", 1)
                if f_var_type != var_type:
                    continue

                interval_dict = getattr(error_intervals, f.name)

                # IMPORTANT: stable key order
                for subkey in sorted(interval_dict.keys()):
                    bounds.append(interval_dict[subkey])
                    names.append(f"{var_type}{subkey}{suffix}")

    # Safety checks
    assert len(bounds) == len(names)
    assert len(set(names)) == len(names)

    problem = {
        "num_vars": len(bounds),
        "names": names,
        "bounds": bounds,
    }

    param_values = sl.sample(
        problem,
        N_samples,
        calc_second_order=calc_second_order,
    )

    return param_values, problem, error_intervals