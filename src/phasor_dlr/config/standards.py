import numpy as np
from dataclasses import dataclass

# ------------------------------
# Dataclass for error intervals
# ------------------------------
@dataclass(frozen=True)
class ErrorIntervals:
    """
    r_*      : dimensionless ratio errors
    theta_*  : phase errors in radians
    """
    r_I: dict
    r_V: dict
    theta_I: dict
    theta_V: dict

# ------------------------------
# Base limits for other tuples
# ------------------------------
BASE_LIMITS = {
    "r_I": {
        "CU": (-0.0004, 0.0004),
        "CS": (-0.01, 0.01),
    },
    "r_V": {
        "VU": (-0.0004, 0.0004),
        "VS1": (-0.0005, 0),
        "VS2": (-0.01, 0.01),
    },
    "theta_I": {
        "CU": (-2, 2),
        "CS": (-12, 12),
    },
    "theta_V": {
        "VU": (-2, 2),
        "VS1": (0, 0),
        "VS2": (-12, 12),
    },
}

# ------------------------------
# CT accuracy class limits
# ------------------------------
CT_LIMITS = {
    "5P": {"r": (-0.01, 0.01), "theta": (-60, 60)},
    "0.2": {"r": (-0.002, 0.002), "theta": (-10, 10)},
    "0.1": {"r": (-0.001, 0.001), "theta": (-5, 5)},
}

# ------------------------------
# VT accuracy class limits
# ------------------------------
VT_LIMITS = {
    "0.5": {"r": (-0.005, 0.005), "theta": (-20, 20)},
    "0.2": {"r": (-0.002, 0.002), "theta": (-10, 10)},
    "0.1": {"r": (-0.001, 0.001), "theta": (-5, 5)},
}

# ------------------------------
# Utility functions
# ------------------------------
def constrain(interval, limit):
    """Constrain a base interval to max(abs(limit))."""
    a, b = interval
    L = max(abs(limit[0]), abs(limit[1]))
    return (max(a, -L), min(b, L))

def minToRad_dict(theta_dict_min):
    """Convert dict of intervals from minutes to radians."""
    minToRad_constant = np.pi / 180 / 60
    return {k: (a * minToRad_constant, b * minToRad_constant)
            for k, (a, b) in theta_dict_min.items()}

def clip_zero_interval(interval, eps=1e-12):
    """Ensure interval is never exactly zero."""
    a, b = interval
    if a == b:
        return (a - eps, b + eps)
    return (a, b)

# ------------------------------
# Main factory function
# ------------------------------
def make_error_intervals(CT_class: str, VT_class: str) -> ErrorIntervals:
    """Return fully processed error intervals given CT and VT classes."""
    
    # Get CT/VT limits
    ct_r, ct_t = CT_LIMITS[CT_class]["r"], CT_LIMITS[CT_class]["theta"]
    vt_r, vt_t = VT_LIMITS[VT_class]["r"], VT_LIMITS[VT_class]["theta"]
    
    # Max envelopes for other tuples
    max_r = (min(ct_r[0], vt_r[0]), max(ct_r[1], vt_r[1]))
    max_t = (min(ct_t[0], vt_t[0]), max(ct_t[1], vt_t[1]))
    
    # Compose linear (r) intervals
    r_I = {"CT": ct_r, **{k: constrain(v, max_r) for k, v in BASE_LIMITS["r_I"].items()}}
    r_V = {"VT": vt_r, **{k: constrain(v, max_r) for k, v in BASE_LIMITS["r_V"].items()}}
    
    # Compose angular (theta) intervals in minutes
    theta_I_min = {"CT": ct_t, **{k: constrain(v, max_t) for k, v in BASE_LIMITS["theta_I"].items()}}
    theta_V_min = {"VT": vt_t, **{k: constrain(v, max_t) for k, v in BASE_LIMITS["theta_V"].items()}}
    
    # Convert to radians
    theta_I = minToRad_dict(theta_I_min)
    theta_V = minToRad_dict(theta_V_min)
    
    # Clip zero-length intervals
    r_I = {k: clip_zero_interval(v) for k, v in r_I.items()}
    r_V = {k: clip_zero_interval(v) for k, v in r_V.items()}
    theta_I = {k: clip_zero_interval(v) for k, v in theta_I.items()}
    theta_V = {k: clip_zero_interval(v) for k, v in theta_V.items()}
    
    return ErrorIntervals(r_I=r_I, r_V=r_V, theta_I=theta_I, theta_V=theta_V)
