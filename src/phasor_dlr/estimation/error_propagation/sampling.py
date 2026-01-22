# sampling.py
import numpy as np

def sample_error_intervals(
    error_intervals,
    M,
    NSAMP,
    *,
    equipment_error_mode=False,
    measurement_variances=None,
    multiplicative=False,
    rng=np.random,
):
    """
    Parameters
    ----------
    error_intervals :
        One of:
        - dict[str, (a, b)]      (ErrorIntervals field)
        - iterable[(a, b)]       (list/tuple)
    """

    # --------------------------------------------------
    # Normalize ErrorIntervals input → list[(a, b)]
    # --------------------------------------------------
    if isinstance(error_intervals, dict):
        intervals = list(error_intervals.values())
    else:
        try:
            intervals = list(error_intervals)
        except TypeError:
            raise TypeError(
                "error_intervals must be a dict or iterable of (a, b) tuples "
                "(e.g. accuracy_r.r_I)"
            )

    if not intervals:
        raise ValueError("error_intervals must not be empty")

    for i, interval in enumerate(intervals):
        if not (
            isinstance(interval, (tuple, list))
            and len(interval) == 2
        ):
            raise ValueError(
                f"Invalid interval at index {i}: {interval}. "
                "Each interval must be a (min, max) tuple."
            )

    n_intervals = len(intervals)

    # --------------------------------------------------
    # Non-equipment-error mode
    # --------------------------------------------------
    if not equipment_error_mode:
        if multiplicative:
            total = np.ones((M, NSAMP))
            for a, b in intervals:
                total *= rng.uniform(1 + a, 1 + b, size=(M, NSAMP))
            return total
        else:
            total = np.zeros((M, NSAMP))
            for a, b in intervals:
                total += rng.uniform(a, b, size=(M, NSAMP))
            return total

    # --------------------------------------------------
    # Equipment-error mode
    # --------------------------------------------------
    if measurement_variances is None:
        raise ValueError("measurement_variances required in equipment_error_mode")

    measurement_variances = np.asarray(measurement_variances)

    if measurement_variances.shape != (M, n_intervals):
        raise ValueError(
            f"measurement_variances must have shape (M, {n_intervals})"
        )

    # One equipment bias per interval
    eq_errors = np.array([rng.uniform(a, b) for a, b in intervals])

    total = np.zeros((M, NSAMP))
    for k in range(n_intervals):
        std = np.sqrt(measurement_variances[:, k])[:, None]
        noise = rng.normal(0.0, std, size=(M, NSAMP))
        total += eq_errors[k] + noise

    return total
