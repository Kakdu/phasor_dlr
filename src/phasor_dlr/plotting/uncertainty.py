import matplotlib.pyplot as plt
import os
from scipy.stats import gaussian_kde, norm
import numpy as np
from itertools import cycle

from phasor_dlr.config.defaults import condParameters

## ------ Colours and styles ------
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
colors_rgb = [tuple(int(h[i : i + 2], 16) / 255 for i in (1, 3, 5)) for h in colors]
lines = ["-", "--", ":", "-."]
linecycler = cycle(lines)

styles = {
    "temperature": {"color": colors[0], "linestyle": "-", "linewidth": 2},
    "temperature_2": {"color": colors[0], "linestyle": "--", "linewidth": 1},
    "angle": {"color": colors[1], "linestyle": "-", "linewidth": 2},
    "angle_2": {"color": colors[1], "linestyle": "--", "linewidth": 2},
    "magnitude": {"color": colors[2], "linestyle": "-", "linewidth": 2},
    "magnitude_2": {"color": colors[2], "linestyle": "--", "linewidth": 2},
    "combined": {"color": colors[8], "linestyle": "-", "linewidth": 2},
    "combined_2": {"color": colors[8], "linestyle": "--", "linewidth": 2},
}


def plot_kde(
    samples,
    title,
    xlabel,
    style=None,
    confidence_interval=None,
    normal_confidence_interval=None,
    show=False,
    filename=None,
    interval=None,
    fit_normal=True,
):
    """
    Plot KDE of samples with optional confidence interval and custom style.

    Parameters
    ----------
    samples : array-like
        Monte Carlo samples.
    title : str
        Plot title.
    xlabel : str
        Label for x-axis.
    style : dict or None
        Dictionary with keys 'color', 'linestyle', 'linewidth'.
        If None, defaults to color='C0', linestyle='-', linewidth=2.
    confidence_interval : float or None
        Confidence level in percent (e.g., 95 for 95% CI). If None, no CI plotted.
    interval : tuple or None
        Defines x-axis limits. If None, automatically fitted to data
    """
    if style is None:
        style = {"color": "C0", "linestyle": "-", "linewidth": 2}

    kde = gaussian_kde(samples)
    grid = np.linspace(samples.min(), samples.max(), 500)
    f = kde(grid)

    plt.figure(figsize=(8, 5))

    # KDE
    plt.plot(
        grid,
        f,
        color=style.get("color", "C0"),
        linestyle=style.get("linestyle", "-"),
        linewidth=style.get("linewidth", 2),
        label="KDE",
    )

    # Normal fit
    if fit_normal:
        mu = np.mean(samples)
        sigma = np.std(samples, ddof=1)
        pdf = norm.pdf(grid, mu, sigma)

        plt.plot(
            grid,
            pdf,
            linestyle=style.get("linestyle", "-"),
            linewidth=0.8,
            color="black",
            label="Normal fit",
        )

        # Normal CI
        if normal_confidence_interval is not None:
            alpha = 1 - normal_confidence_interval / 100
            lower = norm.ppf(alpha / 2, mu, sigma)
            upper = norm.ppf(1 - alpha / 2, mu, sigma)

            pdf_lower = norm.pdf(lower, mu, sigma)
            pdf_upper = norm.pdf(upper, mu, sigma)
            line_height = 1.5 * max(pdf_lower, pdf_upper)

            plt.vlines(
                [lower, upper],
                ymin=0,
                ymax=line_height,
                color="black",
                linestyle=style.get("linestyle", "-"),
                linewidth=0.8,
                label=f"Normal {normal_confidence_interval:.1f}% CI",
            )

    # KDE CI
    if confidence_interval is not None:
        alpha = 1 - confidence_interval / 100
        lower = np.percentile(samples, 100 * alpha / 2)
        upper = np.percentile(samples, 100 * (1 - alpha / 2))

        f_lower = np.interp(lower, grid, f)
        f_upper = np.interp(upper, grid, f)
        line_height = 1.5 * max(f_lower, f_upper)

        plt.vlines(
            [lower, upper],
            ymin=0,
            ymax=line_height,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"{confidence_interval:.1f}% CI",
        )

    if interval is not None:
        def pi_formatter(x, pos):
            frac = Fraction(x / np.pi).limit_denominator()
            if frac == 0:
                return "0"
            elif frac == 1:
                return "π"
            elif frac == -1:
                return "-π"
            else:
                return f"{frac}π"

        plt.xlim(interval)
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(np.pi / 8))
        plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(pi_formatter))

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("KDE")
    plt.grid(False)
    plt.legend()
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300)
    if show:
        plt.show()


def plot_kdes(
    samples_list,
    labels=None,
    title="",
    xlabel="",
    styles=None,
    confidence_interval=None,
    normal_confidence_interval=None,
    filename=None,
    show=False,
    fit_normal=True,
):
    """
    Plot multiple KDEs on the same axes with optional confidence intervals and custom styles.

    Parameters
    ----------
    samples_list : list of array-like
        List of sample arrays to plot.
    labels : list of str or None
        Labels for each dataset. If None, no legend is shown.
    title : str
        Plot title.
    xlabel : str
        Label for x-axis.
    styles : list of dict or None
        List of style dicts (keys: 'color', 'linestyle', 'linewidth') for each dataset.
        If None, defaults are used.
    confidence_interval : float or None
        Confidence level in percent (e.g., 95 for 95% CI). If None, no CI plotted.
    """
    plt.figure(figsize=(8, 5))
    n = len(samples_list)

    if styles is None:
        colors = plt.cm.tab10.colors
        styles = [
            {"color": colors[i % 10], "linestyle": "-", "linewidth": 2}
            for i in range(n)
        ]

    for i, samples in enumerate(samples_list):
        kde = gaussian_kde(samples)
        grid = np.linspace(samples.min(), samples.max(), 500)
        f = kde(grid)

        label = None if labels is None else labels[i]

        # KDE
        plt.plot(
            grid,
            f,
            color=styles[i].get("color", "C0"),
            linestyle=styles[i].get("linestyle", "-"),
            linewidth=styles[i].get("linewidth", 2),
            label=label,
        )

        # Normal fit
        if fit_normal:
            mu = np.mean(samples)
            sigma = np.std(samples, ddof=1)
            pdf = norm.pdf(grid, mu, sigma)

            normal_label = None
            if labels is not None:
                normal_label = f"Normal fit ({labels[i]})"

            plt.plot(
                grid,
                pdf,
                linestyle=styles[i].get("linestyle", "-"),
                linewidth=0.8,
                color="black",
                label=normal_label,
            )

            # Normal CI
            if normal_confidence_interval is not None:
                alpha = 1 - normal_confidence_interval / 100
                lower = norm.ppf(alpha / 2, mu, sigma)
                upper = norm.ppf(1 - alpha / 2, mu, sigma)

                pdf_lower = norm.pdf(lower, mu, sigma)
                pdf_upper = norm.pdf(upper, mu, sigma)
                line_height = 1.5 * max(pdf_lower, pdf_upper)

                plt.vlines(
                    [lower, upper],
                    ymin=0,
                    ymax=line_height,
                    color="black",
                    linestyle=styles[i].get("linestyle", "-"),
                    linewidth=0.8,
                )

        # KDE CI
        if confidence_interval is not None:
            alpha = 1 - confidence_interval / 100
            lower = np.percentile(samples, 100 * alpha / 2)
            upper = np.percentile(samples, 100 * (1 - alpha / 2))

            f_lower = np.interp(lower, grid, f)
            f_upper = np.interp(upper, grid, f)
            line_height = 1.5 * max(f_lower, f_upper)

            plt.vlines(
                [lower, upper],
                ymin=0,
                ymax=line_height,
                color=styles[i].get("color", "C0"),
                linestyle="--",
                linewidth=2,
            )

    if labels is not None or fit_normal:
        plt.legend()

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("pdf")
    plt.grid(False)
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300)
    if show:
        plt.show()


def plot_sent_received(samples_r, samples_s, folder="figures", confidence_interval=None):
    os.makedirs(folder, exist_ok=True)

    # --- 1. Current amplitude ---
    plot_kdes(
        [samples_r["I_amp"].flatten(), samples_s["I_amp"].flatten()],
        labels=["Received", "Sent"],
        title="Current amplitude",
        xlabel="Current [A]",
        styles=[styles["magnitude_2"], styles["magnitude"]],
        confidence_interval=confidence_interval,
        filename=f"{folder}/current_amplitude.png",
    )

    # --- 2. Voltage amplitude ---
    plot_kdes(
        [samples_r["V_amp"].flatten() / 1e3, samples_s["V_amp"].flatten() / 1e3],
        labels=["Received", "Sent"],
        title="Voltage amplitude",
        xlabel="Voltage [kV]",
        styles=[styles["magnitude_2"], styles["magnitude"]],
        confidence_interval=confidence_interval,
        filename=f"{folder}/voltage_amplitude.png",
    )

    # --- 3. Phase difference (V - I) ---
    plot_kdes(
        [(samples_r["V_theta"] - samples_r["I_theta"]).flatten(),
         (samples_s["V_theta"] - samples_s["I_theta"]).flatten()],
        labels=["Received", "Sent"],
        title="Phase difference (V - I)",
        xlabel="Phase difference [rad]",
        styles=[styles["angle_2"], styles["angle"]],
        confidence_interval=confidence_interval,
        filename=f"{folder}/phase_difference.png",
    )

    # --- 4. Active power (S_samples) ---
    plot_kdes(
        [samples_r["S_samples"].flatten() / 1e6, samples_s["S_samples"].flatten() / 1e6],
        labels=["Received", "Sent"],
        title="Active Real Power",
        xlabel="Power [MW]",
        styles=[styles["combined_2"], styles["combined"]],
        confidence_interval=confidence_interval,
        filename=f"{folder}/S_samples.png",
    )

    print(f"All plots saved to folder: {folder}")


def plot_temperature_variance(I_r_sweep, temp_variance, folder="results/figures/error_propagation", filename="variance_over_current"):
    """
    Plot temperature standard deviation vs received current for multiple conductors.

    Parameters
    ----------
    I_r_sweep : array-like
        Array of received current amplitudes [A].
    temp_variance : dict
        Dictionary mapping conductor names to lists/arrays of temperature variance values.
    folder : str
        Folder to save the plot.
    filename : str
        Filename for the saved plot (without extension).
    """
    os.makedirs(folder, exist_ok=True)
    plt.figure()

    for cond_name, var_T in temp_variance.items():
        rateC = 0.8 * condParameters[cond_name]["rateC"]
        rateA = 1.3 * condParameters[cond_name]["rateA"]

        I_r = np.array(I_r_sweep)
        var_T = np.sqrt(np.array(var_T))  # variance -> std

        inside_mask = (I_r >= rateC) & (I_r <= rateA)
        outside_mask = ~inside_mask

        # Choose linestyle per conductor
        linestyle = "--" if cond_name == "bohus" else ":"

        # Outside operating range (black)
        if np.any(outside_mask):
            plt.plot(
                I_r[outside_mask],
                var_T[outside_mask],
                color="black",
                lw=1,
                linestyle=linestyle,
                label=cond_name,
                zorder=1,
            )

        # Inside operating range (blue)
        if np.any(inside_mask):
            plt.plot(
                I_r[inside_mask],
                var_T[inside_mask],
                color="tab:blue",
                lw=2,
                linestyle="-",
                label="Operational Currents" if cond_name == "gota" else "",
                zorder=2,
            )

    plt.xlabel("Received Current Amplitude $\iota^{(r)}$ [A]")
    plt.ylabel(
        "Temperature Standard Deviation $\\mathrm{Std}(T_{\\mathrm{AV}})$ [$^\circ$C]"
    )
    plt.title("Temperature Standard Deviation vs Current")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(False)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(folder, f"{filename}.png"))
    plt.close()