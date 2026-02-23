import os
import arviz as az
from datetime import datetime
import numpy as np
from scipy.stats import gaussian_kde

from phasor_dlr.models.temperature import temperature_distribution

def log_monte_carlo_run(samples_r, samples_s, condParam,
                        folder="results/logs",
                        filename_prefix="mc_run"):

    import os
    import numpy as np
    from datetime import datetime

    os.makedirs(folder, exist_ok=True)

    log_lines = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_lines.append(f"Monte Carlo Run Log - {timestamp}\n")
    log_lines.append("="*60 + "\n")

    # -------------------------------------------------
    # Helper summary
    # -------------------------------------------------
    def summarize(name, data_r, data_s):

        mean_r, mean_s = np.mean(data_r), np.mean(data_s)
        std_r, std_s   = np.std(data_r), np.std(data_s)
        var_r, var_s   = np.var(data_r), np.var(data_s)
        min_r, min_s   = np.min(data_r), np.min(data_s)
        max_r, max_s   = np.max(data_r), np.max(data_s)

        return [
            f"{name} Summary:",
            f"  Received -> mean: {mean_r:.6g}, std: {std_r:.6g}, min: {min_r:.6g}, max: {max_r:.6g}",
            f"  Sent     -> mean: {mean_s:.6g}, std: {std_s:.6g}, min: {min_s:.6g}, max: {max_s:.6g}",
            "-"*60
        ]

    # -------------------------------------------------
    # Base variables
    # -------------------------------------------------
    variables = ["I_amp", "V_amp", "I_theta", "V_theta", "S_samples"]

    for var in variables:
        log_lines.extend(
            summarize(var,
                      samples_r[var].flatten(),
                      samples_s[var].flatten())
        )

    # -------------------------------------------------
    # Phase difference (custom definition)
    # -------------------------------------------------
    Vt_r = samples_r["V_theta"].flatten()
    It_r = samples_r["I_theta"].flatten()

    Vt_s = samples_s["V_theta"].flatten()
    It_s = samples_s["I_theta"].flatten()

    # Mean / min / max → subtraction
    mean_r = np.mean(Vt_r) - np.mean(It_r)
    mean_s = np.mean(Vt_s) - np.mean(It_s)

    min_r  = np.min(Vt_r) - np.min(It_r)
    min_s  = np.min(Vt_s) - np.min(It_s)

    max_r  = np.max(Vt_r) - np.max(It_r)
    max_s  = np.max(Vt_s) - np.max(It_s)

    # Std → sqrt(sum of variances)
    std_r = np.sqrt(np.std(Vt_r)**2 + np.std(It_r)**2)
    std_s = np.sqrt(np.std(Vt_s)**2 + np.std(It_s)**2)

    var_r = std_r**2
    var_s = std_s**2

    log_lines.extend([
        "PhaseDiff (V_theta - I_theta) Summary:",
        f"  Received -> mean: {mean_r:.6g}, std: {std_r:.6g}, min: {min_r:.6g}, max: {max_r:.6g}",
        f"  Sent     -> mean: {mean_s:.6g}, std: {std_s:.6g}, min: {min_s:.6g}, max: {max_s:.6g}",
        "-"*60
    ])

    # -------------------------------------------------
    # Derived quantities
    # -------------------------------------------------
    P_loss = samples_s["S_samples"] - samples_r["S_samples"]
    I_AC   = 0.5 * (samples_s["I_amp"] + samples_r["I_amp"])

    log_lines.extend([
        "P_loss Summary:",
        f"  mean: {np.mean(P_loss):.6g}, std: {np.std(P_loss):.6g}, "
        f"min: {np.min(P_loss):.6g}, max: {np.max(P_loss):.6g}",
        "-"*60
    ])

    log_lines.extend([
        "I_AC Summary:",
        f"  mean: {np.mean(I_AC):.6g}, std: {np.std(I_AC):.6g}, "
        f"min: {np.min(I_AC):.6g}, max: {np.max(I_AC):.6g}",
        "-"*60
    ])

    # -------------------------------------------------
    # Temperature distribution
    # -------------------------------------------------
    T, T_max, T_avg = temperature_distribution(
        P_loss, I_AC, cond=condParam
    )

    log_lines.extend([
        "Temperature Distribution Summary:",
        "-"*60,
        f"T(x) overall Summary:",
        f"  mean: {np.mean(T):.6g}, std: {np.std(T):.6g}, "
        f"min: {np.min(T):.6g}, max: {np.max(T):.6g}",
        "-"*60
    ])

    # -------------------------------------------------
    # Print + save
    # -------------------------------------------------
    for line in log_lines:
        print(line)

    log_file = os.path.join(
        folder,
        f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    with open(log_file, "w") as f:
        for line in log_lines:
            f.write(line + "\n")

    print(f"\nLog saved to {log_file}")




def log_sobol_indices(sobol_results, folder="results/logs", filename_prefix="sobol_run"):
    """
    Logs Sobol sensitivity indices (S1, ST) with summary statistics.

    Parameters
    ----------
    sobol_results : dict
        Dictionary with keys "S1" and "ST", each containing dicts of parameter: samples.
    folder : str
        Folder to save the log file.
    filename_prefix : str
        Prefix for the log file.
    """
    os.makedirs(folder, exist_ok=True)
    log_lines = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_lines.append(f"Sobol Sensitivity Indices Log - {timestamp}\n")
    log_lines.append("="*60 + "\n")
    
    for order in ["S1", "ST"]:
        log_lines.append(f"{order} indices summary:\n")
        for param, samples in sobol_results[order].items():
            mean_val = np.mean(samples)
            var_val = np.var(samples)
            min_val = np.min(samples)
            max_val = np.max(samples)
            lines = [
                f"  {param}: mean={mean_val:.6g}, var={var_val:.6g}, min={min_val:.6g}, max={max_val:.6g}"
            ]
            log_lines.extend(lines)
        log_lines.append("-"*60)
    
    # Print and save
    for line in log_lines:
        print(line)
    
    log_file = os.path.join(folder, f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    with open(log_file, "w") as f:
        for line in log_lines:
            f.write(line + "\n")
    
    print(f"\nLog saved to {log_file}")


def log_bayesian_run(trace, T_nom=None, folder="results/logs", filename_prefix="bayesian_run"):
    """
    Logs summary statistics for Bayesian posterior inference.

    Parameters
    ----------
    trace : arviz.InferenceData
        Trace object containing posterior samples.
    T_nom : float or None
        Nominal reference temperature, optional.
    folder : str
        Folder to save the log file.
    filename_prefix : str
        Prefix for the log file.
    """
    os.makedirs(folder, exist_ok=True)
    log_lines = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_lines.append(f"Bayesian Inference Log - {timestamp}\n")
    log_lines.append("="*60 + "\n")
    
    # Extract posterior samples
    samples = trace.posterior["T_AV"].values.flatten()
    
    mean_val = np.mean(samples)
    var_val = np.var(samples)
    min_val = np.min(samples)
    max_val = np.max(samples)

    kde = gaussian_kde(samples)
    x_grid = np.linspace(min_val, max_val, 2000)
    mode_val = x_grid[np.argmax(kde(x_grid))]
    
    hdi_95 = az.hdi(trace, var_names=["T_AV"], hdi_prob=0.95)["T_AV"].values
    hdi_width = 0.5 * (hdi_95[1] - hdi_95[0])
    

    lines = [
        f"T_AV Posterior Summary:",
        f"  Mean: {mean_val:.6g}",
        f"  Mode (KDE): {mode_val:.6g}",
        f"  Variance: {var_val:.6g}",
        f"  Min: {min_val:.6g}, Max: {max_val:.6g}",
        f"  95% HDI: [{hdi_95[0]:.6g}, {hdi_95[1]:.6g}] (Width: {hdi_width:.6g})",
    ]
    
    if T_nom is not None:
        rmse_val = np.sqrt(np.mean((samples - T_nom)**2))
        lines.append(f"  Nominal T_nom: {T_nom:.6g}")
        lines.append(f"  Mean deviation from T_nom: {mean_val - T_nom:.6g}")
        lines.append(f"  RMSE w.r.t. T_nom: {rmse_val:.6g}")
    
    log_lines.extend(lines)
    
    # Print and save
    for line in log_lines:
        print(line)
    
    log_file = os.path.join(folder, f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    with open(log_file, "w") as f:
        for line in log_lines:
            f.write(line + "\n")
    
    print(f"\nLog saved to {log_file}")




def log_bayesian_runs(
    traces,
    labels=None,
    T_nom=None,
    folder="results/logs",
    filename_prefix="bayesian_comparison",
):
    """
    Logs summary statistics for multiple Bayesian posterior inferences
    into a single log file.

    Parameters
    ----------
    traces : list of arviz.InferenceData
        List of trace objects containing posterior samples.
    labels : list of str or None
        Labels describing each run (e.g., ["tP=0, phys=0", ...]).
        If None, runs will be numbered.
    T_nom : float or None
        Nominal reference temperature, optional.
    folder : str
        Folder to save the log file.
    filename_prefix : str
        Prefix for the log file.
    """

    os.makedirs(folder, exist_ok=True)

    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(traces))]

    if len(labels) != len(traces):
        raise ValueError("Length of labels must match length of traces.")

    log_lines = []

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_lines.append(f"Bayesian Multi-Run Inference Log - {timestamp}")
    log_lines.append("=" * 70)
    log_lines.append("")

    # -------------------------------------------------
    # Loop over traces
    # -------------------------------------------------
    for trace, label in zip(traces, labels):

        log_lines.append(f"{label}")
        log_lines.append("-" * 70)

        # Extract posterior samples
        samples = trace.posterior["T_AV"].values.flatten()

        mean_val = np.mean(samples)
        var_val = np.var(samples)
        min_val = np.min(samples)
        max_val = np.max(samples)

        # KDE mode
        kde = gaussian_kde(samples)
        x_grid = np.linspace(min_val, max_val, 2000)
        mode_val = x_grid[np.argmax(kde(x_grid))]

        # HDI
        hdi_95 = az.hdi(trace, var_names=["T_AV"], hdi_prob=0.95)["T_AV"].values
        hdi_width = 0.5 * (hdi_95[1] - hdi_95[0])

        # RMSE vs nominal
        if T_nom is not None:
            rmse_val = np.sqrt(np.mean((samples - T_nom) ** 2))

        # Build lines
        lines = [
            f"T_AV Posterior Summary:",
            f"  Mean: {mean_val:.6g}",
            f"  Mode (KDE): {mode_val:.6g}",
            f"  Variance: {var_val:.6g}",
            f"  Min: {min_val:.6g}, Max: {max_val:.6g}",
            f"  95% HDI: [{hdi_95[0]:.6g}, {hdi_95[1]:.6g}] "
            f"(Half-width: {hdi_width:.6g})",
        ]

        if T_nom is not None:
            lines.extend(
                [
                    f"  Nominal T_nom: {T_nom:.6g}",
                    f"  Mean deviation from T_nom: {mean_val - T_nom:.6g}",
                    f"  RMSE w.r.t. T_nom: {rmse_val:.6g}",
                ]
            )

        log_lines.extend(lines)
        log_lines.append("")

    # -------------------------------------------------
    # Optional cross-run comparison
    # -------------------------------------------------
    log_lines.append("=" * 70)
    log_lines.append("Cross-Run Comparison")
    log_lines.append("=" * 70)

    means = [
        np.mean(t.posterior["T_AV"].values.flatten()) for t in traces
    ]

    for label, mean_val in zip(labels, means):
        log_lines.append(f"{label} mean: {mean_val:.6g}")

    log_lines.append(
        f"Mean spread (max − min): {np.max(means) - np.min(means):.6g}"
    )
    log_lines.append("")

    # -------------------------------------------------
    # Print to console
    # -------------------------------------------------
    for line in log_lines:
        print(line)

    # -------------------------------------------------
    # Save log file
    # -------------------------------------------------
    log_file = os.path.join(
        folder,
        f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )

    with open(log_file, "w") as f:
        for line in log_lines:
            f.write(line + "\n")

    print(f"\nLog saved to {log_file}")