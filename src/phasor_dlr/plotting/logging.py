import os
import arviz as az
from datetime import datetime
import numpy as np

def log_monte_carlo_run(samples_r, samples_s, folder="results/logs", filename_prefix="mc_run"):
    """
    Logs summary statistics (mean, variance, min, max) for sent and received signals.
    
    Parameters
    ----------
    samples_r : dict
        Received samples from monte_carlo_multi
    samples_s : dict
        Sent samples from monte_carlo_multi
    folder : str
        Folder to save the log file.
    filename_prefix : str
        Prefix for the log file.
    """
    os.makedirs(folder, exist_ok=True)
    
    log_lines = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_lines.append(f"Monte Carlo Run Log - {timestamp}\n")
    log_lines.append("="*60 + "\n")
    
    def summarize(name, data_r, data_s):
        mean_r = np.mean(data_r)
        mean_s = np.mean(data_s)
        var_r = np.var(data_r)
        var_s = np.var(data_s)
        min_r = np.min(data_r)
        min_s = np.min(data_s)
        max_r = np.max(data_r)
        max_s = np.max(data_s)
        
        lines = [
            f"{name} Summary:",
            f"  Received -> mean: {mean_r:.6g}, var: {var_r:.6g}, min: {min_r:.6g}, max: {max_r:.6g}",
            f"  Sent     -> mean: {mean_s:.6g}, var: {var_s:.6g}, min: {min_s:.6g}, max: {max_s:.6g}",
            "-"*60
        ]
        return lines
    
    # Variables to log
    variables = ["I_amp", "V_amp", "I_theta", "V_theta", "S_samples"]
    
    for var in variables:
        log_lines.extend(summarize(var, samples_r[var].flatten(), samples_s[var].flatten()))
    
    # Print to console
    for line in log_lines:
        print(line)
    
    # Save to file
    log_file = os.path.join(folder, f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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
    
    hdi_95 = az.hdi(trace, var_names=["T_AV"], hdi_prob=0.95)["T_AV"].values
    hdi_width = 0.5 * (hdi_95[1] - hdi_95[0])
    
    lines = [
        f"T_AV Posterior Summary:",
        f"  Mean: {mean_val:.6g}",
        f"  Variance: {var_val:.6g}",
        f"  Min: {min_val:.6g}, Max: {max_val:.6g}",
        f"  95% HDI: [{hdi_95[0]:.6g}, {hdi_95[1]:.6g}] (Width: {hdi_width:.6g})",
    ]
    
    if T_nom is not None:
        lines.append(f"  Nominal T_nom: {T_nom:.6g}")
        lines.append(f"  Mean deviation from T_nom: {mean_val - T_nom:.6g}")
    
    log_lines.extend(lines)
    
    # Print and save
    for line in log_lines:
        print(line)
    
    log_file = os.path.join(folder, f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    with open(log_file, "w") as f:
        for line in log_lines:
            f.write(line + "\n")
    
    print(f"\nLog saved to {log_file}")
