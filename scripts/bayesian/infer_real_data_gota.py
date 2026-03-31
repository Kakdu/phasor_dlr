import os
import pymc as pm
import argparse
import numpy as np
import multiprocessing as mp
import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import skewnorm
import pandas as pd
import sys

# Import bayesian model
from phasor_dlr.estimation.bayesian.models import build_temperature_model

# Import measurement generation
from phasor_dlr.synthetic_data.generators import generate_static_measurements
from phasor_dlr.config.standards import make_error_intervals
from phasor_dlr.config.defaults import condParameters

# Import measurement variance
from phasor_dlr.utils.math import combined_uniforms_sigma, combined_uniforms_logsigma

# Import logging
from phasor_dlr.plotting.logging import log_bayesian_run

def parse_excel_time(series):

    # Convert to string
    s = series.astype(str)

    # Trim fractional seconds to 6 digits
    s = s.str.replace(
        r"(\.\d{6})\d+",
        r"\1",
        regex=True
    )

    # Convert to datetime
    return pd.to_datetime(
        s,
        format="%Y-%m-%d %H:%M:%S.%f",
        errors="coerce"
    )

def corrected_angle_diff(a_deg, b_deg):
    """
    Computes angle difference a - b (in degrees), correcting any
    ±180° global offset so that the mean difference is closer to 0°.
    
    Parameters:
        a_deg, b_deg : np.ndarray
            Angle arrays in degrees

    Returns:
        diff : np.ndarray
            Corrected angle difference
    """
    # Raw difference
    diff = a_deg - b_deg

    # Wrap to [-180, 180)
    diff = (diff + 180) % 360 - 180

    # Check global offset using mean
    mean_diff = np.mean(diff)

    if mean_diff > 90:
        # Shift entire array down by 180
        diff = (diff - 180 + 180) % 360 - 180
    elif mean_diff < -90:
        # Shift entire array up by 180
        diff = (diff + 180 + 180) % 360 - 180

    return diff

def remove_spikes(values, window=5, threshold=3):
    """
    Remove spikes from a 1D NumPy array.

    Parameters:
        values : np.ndarray
            The original signal
        window : int
            Window size for rolling median (odd number)
        threshold : float
            Spike threshold in units of standard deviations

    Returns:
        cleaned : np.ndarray
            Array with spikes replaced by median values
    """
    series = pd.Series(values)
    rolling_median = series.rolling(window=window, center=True, min_periods=1).median()
    diff = np.abs(series - rolling_median)
    std = np.std(series)

    # Identify spikes
    spikes = diff > threshold * std

    # Replace spikes with rolling median
    cleaned = series.copy()
    cleaned[spikes] = rolling_median[spikes]

    return cleaned.to_numpy()


# -------------------------------
# Parser arguments
# -------------------------------
parser = argparse.ArgumentParser(description="Run power system script.")

parser.add_argument("--f", type=float, default=50)
parser.add_argument("--T_nom", type=float, default=50)
parser.add_argument("--N", type=int, default=1)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--hdi", type=int, default=95)
parser.add_argument("--tP", type=int, default=0)
parser.add_argument("--phys_informed", type=int, default=1)

parser.add_argument("--CT_class", type=str, default="5P")
parser.add_argument("--VT_class", type=str, default="0.2")


parser.add_argument("--cond", type=str, default="gota")
parser.add_argument("--VrAmp", type=float, default=140000.0)
parser.add_argument("--VrAngle", type=float, default=0.8)
parser.add_argument("--IrAmp", type=float, default=None)

args = parser.parse_args()



# -------------------------------
# Data import
# -------------------------------

# Determine folder of the executable
if getattr(sys, "frozen", False):
    # Running as executable
    exe_dir = os.path.dirname(sys.executable)
    data_dir = os.path.abspath(os.path.join(exe_dir, "..", "data", args.cond))
    results_fig_dir = os.path.join(exe_dir, "..", "results", "figures", args.cond)
    results_log_dir = os.path.join(exe_dir, "..", "results", "logs", args.cond)
    meipass = sys._MEIPASS
    os.environ["ARVIZ_DATA_DIR"] = os.path.join(meipass, "arviz", "static")
else:
    # Running as script
    exe_dir = os.getcwd()
    data_dir = os.path.abspath(os.path.join(exe_dir, "data"))
    results_fig_dir = os.path.join(exe_dir, "results", "figures")
    results_log_dir = os.path.join(exe_dir, "results", "logs")

data_variables_dir = os.path.abspath(os.path.join(data_dir, "variablePlots"))
os.makedirs(results_fig_dir, exist_ok=True)
os.makedirs(results_log_dir, exist_ok=True)
os.makedirs(data_variables_dir, exist_ok=True)


file_path_send = os.path.join(data_dir, "send-station.xlsx")
file_path_rec  = os.path.join(data_dir, "rec-station.xlsx")

# Read the sheets
key_sheet_send   = pd.read_excel(file_path_send, sheet_name=0, header=None)
value_sheet_send = pd.read_excel(file_path_send, sheet_name=1)

key_sheet_rec    = pd.read_excel(file_path_rec, sheet_name=0, header=None)
value_sheet_rec  = pd.read_excel(file_path_rec, sheet_name=1)



# -------------------------------
# Format value sheets
# -------------------------------

value_sheet_send = value_sheet_send.iloc[:, [0, 1, 2]]
value_sheet_send.columns = ["Time", "ID", "Value"]

value_sheet_rec = value_sheet_rec.iloc[:, [0, 1, 2]]
value_sheet_rec.columns = ["Time", "ID", "Value"]


# -------------------------------
# Format key sheets
# -------------------------------

key_sheet_send = key_sheet_send.iloc[:, [0, 1]]
key_sheet_send.columns = ["Variable", "ID"]
key_sheet_send = key_sheet_send.dropna(subset=["Variable", "ID"])

key_sheet_rec = key_sheet_rec.iloc[:, [0, 1]]
key_sheet_rec.columns = ["Variable", "ID"]
key_sheet_rec = key_sheet_rec.dropna(subset=["Variable", "ID"])


# -------------------------------
# Ensure matching ID types
# -------------------------------

value_sheet_send["ID"] = value_sheet_send["ID"].astype(str)
value_sheet_rec["ID"]  = value_sheet_rec["ID"].astype(str)

key_sheet_send["ID"] = key_sheet_send["ID"].astype(str)
key_sheet_rec["ID"]  = key_sheet_rec["ID"].astype(str)


# -------------------------------
# Build ID --> Variable mappings
# -------------------------------

id_to_var_send = dict(zip(key_sheet_send["ID"],
                          key_sheet_send["Variable"]))

id_to_var_rec = dict(zip(key_sheet_rec["ID"],
                         key_sheet_rec["Variable"]))


# -------------------------------
# Map variable names onto values
# -------------------------------

value_sheet_send["Variable"] = \
    value_sheet_send["ID"].map(id_to_var_send)

value_sheet_rec["Variable"] = \
    value_sheet_rec["ID"].map(id_to_var_rec)


# -------------------------------
# Group into NumPy arrays
# -------------------------------

value_sheet_send["Time"] = parse_excel_time(
    value_sheet_send["Time"]
)

value_sheet_rec["Time"] = parse_excel_time(
    value_sheet_rec["Time"]
)

arrays_send = {}
time_send   = {}

for var, group in value_sheet_send.groupby("Variable"):

    # Sort by time
    group = group.sort_values("Time")

    arrays_send[var] = group["Value"].to_numpy()
    time_send[var]   = group["Time"].to_numpy()

arrays_rec = {}
time_rec   = {}

for var, group in value_sheet_rec.groupby("Variable"):

    group = group.sort_values("Time")

    arrays_rec[var] = group["Value"].to_numpy()
    time_rec[var]   = group["Time"].to_numpy()


phases = [1, 2, 3]


phi_send = {}
time_phi_send = {}

for ph in phases:

    ul_key = f"UL{ph}ANG"
    il_key = f"IL{ph}ANG"

    if ul_key in arrays_send and il_key in arrays_send:

        phi_send[f"PHI{ph}"] = corrected_angle_diff(
            arrays_send[ul_key],
            arrays_send[il_key]
        )

        # time (same for both --> take one)
        time_phi_send[f"PHI{ph}"] = \
            time_send[ul_key]


phi_rec = {}
time_phi_rec = {}

for ph in phases:

    ul_key = f"UL{ph}ANG"
    il_key = f"IL{ph}ANG"

    if ul_key in arrays_rec and il_key in arrays_rec:

        phi_rec[f"PHI{ph}"] = corrected_angle_diff(
            arrays_rec[ul_key],
            arrays_rec[il_key]
        )

        time_phi_rec[f"PHI{ph}"] = \
            time_rec[ul_key]

arrays_send.update(phi_send)
arrays_rec.update(phi_rec)

time_send.update(time_phi_send)
time_rec.update(time_phi_rec)

arrays_send_clean = {}
arrays_rec_clean = {}

for var, vals in arrays_send.items():
    arrays_send_clean[var] = remove_spikes(vals, window=100, threshold=2)

for var, vals in arrays_rec.items():
    arrays_rec_clean[var] = remove_spikes(vals, window=100, threshold=2)


# Combine all variable names from sent and received
all_vars = set(arrays_send_clean.keys()) | set(arrays_rec_clean.keys())

for var in all_vars:
    plt.figure(figsize=(6,4))
    
    # Plot sent if available
    if var in arrays_send_clean:
        plt.plot(time_send[var], arrays_send_clean[var], 
                 label="Sent", linestyle='-')
    
    # Plot received if available
    if var in arrays_rec_clean:
        plt.plot(time_rec[var], arrays_rec_clean[var], 
                 label="Received", linestyle='-')
    
    plt.title(f"{var}")
    plt.xlabel("Time")
    plt.ylabel(var)
    plt.grid(True)
    plt.legend()
    
    # Save plot
    filename = f"{var}.png"
    filepath = os.path.join(data_variables_dir, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close()

# -------------------------------
# Conductor Parameters
# -------------------------------
condParam = condParameters[args.cond]

# -------------------------------
# Measurement accuracy and variance
# -------------------------------
accuracy_r = make_error_intervals(args.CT_class, args.VT_class)
accuracy_s = make_error_intervals(args.CT_class, args.VT_class)

sigma_logV = combined_uniforms_logsigma(accuracy_r.r_V)
sigma_logI = combined_uniforms_logsigma(accuracy_r.r_I)
sigma_phiV = combined_uniforms_sigma(accuracy_r.theta_V)
sigma_phiI = combined_uniforms_sigma(accuracy_r.theta_I)


# -------------------------------
# Physical properties
# -------------------------------
f = args.f
T_nom = args.T_nom

R = condParam["L"] * condParam["R_20"] * (1 + condParam["alpha"] * (T_nom - condParam["T_ref"]))
X = condParam["L"] * condParam["X"]
C = condParam["C"]

w = 2 * np.pi * f
Z = R + 1j * X
Y = 1j * w * C

def run_phase_model(phase, arrays_send, arrays_rec, time_send, time_rec,
                    condParam, sigma_logV, sigma_logI, sigma_phiV, sigma_phiI,
                    w, C, args, seed=1234):
    """
    Build and run Bayesian temperature model for a single phase.

    Parameters:
        phase : int
            Phase number (1,2,3)
        arrays_send/arrays_rec : dict
            Measurement arrays keyed by variable name
        time_send/time_rec : dict
            Time arrays
        condParam, sigma_logV, sigma_logI, sigma_phiV, sigma_phiI, w, C, args
            Model parameters
        seed : int
            Random seed

    Returns:
        trace : arviz.InferenceData
            Posterior trace for T_AV
    """
    # -------------------------------
    # Variable names for this phase
    # -------------------------------
    V_r_key       = f"UL{phase}MAG"
    I_r_key       = f"IL{phase}MAG"
    V_angle_key   = f"PHI{phase}"
    I_angle_key   = None
    V_s_key       = f"UL{phase}MAG"
    I_s_key       = f"IL{phase}MAG"
    V_s_angle_key = f"PHI{phase}" 
    I_s_angle_key = None 

    # -------------------------------
    # Nominal values: first measurement
    # -------------------------------
    V_r_nom = arrays_send[V_r_key][0]
    I_r_nom = arrays_send[I_r_key][0]
    V_r_angle_nom = np.deg2rad(arrays_send[V_angle_key][0])
    I_r_angle_nom = 0.0

    V_s_nom = arrays_rec[V_s_key][0]
    I_s_nom = arrays_rec[I_s_key][0]
    V_s_angle_nom = np.deg2rad(arrays_rec[V_s_angle_key][0])
    I_s_angle_nom = 0.0

    # -------------------------------
    # Measurement arrays
    # -------------------------------
    V_r_meas_array = arrays_send[V_r_key]
    I_r_meas_array = arrays_send[I_r_key]
    V_r_angle_meas_array = np.deg2rad(arrays_send[V_angle_key])
    I_r_angle_meas_array = np.zeros_like(V_r_meas_array)

    V_s_meas_array = arrays_rec[V_s_key]
    I_s_meas_array = arrays_rec[I_s_key]
    V_s_angle_meas_array = np.deg2rad(arrays_rec[V_s_angle_key])
    I_s_angle_meas_array = np.zeros_like(V_s_meas_array)


    # -------------------------------
    # Build model
    # -------------------------------
    model = build_temperature_model(
        condParameter=condParam,
        V_r_nom=V_r_nom,
        I_r_nom=I_r_nom,
        V_r_angle_nom=V_r_angle_nom,
        I_r_angle_nom=I_r_angle_nom,
        V_s_nom=V_s_nom,
        I_s_nom=I_s_nom,
        V_s_angle_nom=V_s_angle_nom,
        I_s_angle_nom=I_s_angle_nom,
        V_r_meas_array=V_r_meas_array,
        I_r_meas_array=I_r_meas_array,
        V_r_angle_meas_array=V_r_angle_meas_array,
        I_r_angle_meas_array=I_r_angle_meas_array,
        V_s_meas_array=V_s_meas_array,
        I_s_meas_array=I_s_meas_array,
        V_s_angle_meas_array=V_s_angle_meas_array,
        I_s_angle_meas_array=I_s_angle_meas_array,
        sigma_logV=sigma_logV,
        sigma_logI=sigma_logI,
        sigma_phiV=sigma_phiV,
        sigma_phiI=sigma_phiI,
        w=w,
        C=C,
        tP=args.tP,
        phys_informed=args.phys_informed,
        seed=seed
    )

    # -------------------------------
    # Run model
    # -------------------------------


    def main():
        mp.set_start_method("fork", force=True)
        with model:
            trace = pm.sample(
                draws=4000,
                tune=1000,
                chains=4,
                target_accept=0.95,
                random_seed=seed
            )

        az.summary(trace, var_names=["T_AV"])

        fig, ax = plt.subplots(figsize=(6, 4))

        az.plot_posterior(
            trace,
            var_names=["T_AV"],
            hdi_prob=args.hdi / 100,
            point_estimate=None,
            ax=ax,
        )

        for txt in ax.texts:
            txt.set_visible(False)

        posterior_line = ax.lines[0]
        posterior_line.set_label(r"$T_{\mathrm{AV}}$ posterior")

        ax.set_xlabel("Degrees [$^\circ$C]")
        ax.set_ylabel(None)
        ax.legend()
        title = "Posterior of $T_{\\mathrm{AV}}$"
        ax.set_title(title)
        plt.tight_layout()

        hdi_handle = Line2D(
            [0],
            [0],
            color="black",
            linewidth=4,
            label=f"{args.hdi:.0f}%" + " $T_{\\mathrm{AV}}$ HDI",
        )

        handles = [posterior_line, hdi_handle]

        ax.legend(handles=handles)

        plt.savefig(os.path.join(results_fig_dir, f"temperature_posterior_phase{phase}.png"), dpi=300)
        plt.close()

        log_bayesian_run(trace, folder=results_log_dir, filename_prefix=f"bayes_run_phase{phase}")

    if __name__ == "__main__":
        main()

    return model


for ph in [1,2,3]:
    model = run_phase_model(
        phase=ph,
        arrays_send=arrays_send_clean,
        arrays_rec=arrays_rec_clean,
        time_send=time_send,
        time_rec=time_rec,
        condParam=condParam,
        sigma_logV=sigma_logV,
        sigma_logI=sigma_logI,
        sigma_phiV=sigma_phiV,
        sigma_phiI=sigma_phiI,
        w=w,
        C=C,
        args=args,
        seed=1234
    )
