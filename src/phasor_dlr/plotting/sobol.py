# phasor_dlr/estimation/sobol/plotting.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

def _build_plot_labels(problem):
    """
    Build x_positions, LaTeX labels, and bar_colors from problem['names'],
    preserving both type (r/theta) and send/receiver (_s/_r) distinction.
    """
    names_plain = problem["names"]

    # Detect groups of consecutive same type and send/receiver
    groups = []
    current_group = []
    current_type_sr = None

    for name in names_plain:
        # name like "rVU_s" or "thetaCU_r"
        if "_" not in name:
            raise ValueError(f"Invalid variable name: {name}")
        base, sr = name.split("_")
        if base.startswith("theta"):
            vtype = "theta"
            sub = base[5:]  # e.g. "CU"
        else:
            vtype = "r"
            sub = base[1:]  # e.g. "VU"

        type_sr = f"{vtype}_{sr}"  # combine type + send/receiver

        if type_sr != current_type_sr:
            if current_group:
                groups.append((current_type_sr, current_group))
            current_group = [name]
            current_type_sr = type_sr
        else:
            current_group.append(name)

    if current_group:
        groups.append((current_type_sr, current_group))

    # Build positions and labels
    x_positions = []
    x_labels_grouped = []
    current_x = 0
    group_separation = 2.0

    def to_latex(varname):
        base, sr = varname.split("_")
        if base.startswith("theta"):
            base_letter = r"\theta"
            subscript = base[5:]
        else:
            base_letter = base[0]
            subscript = base[1:]
        return rf"${base_letter}_{{{subscript}}}^{{({sr})}}$"

    bar_colors = []
    for type_sr, group_vars in groups:
        local_positions = current_x + np.arange(len(group_vars))
        x_positions.extend(local_positions)
        x_labels_grouped.extend([to_latex(v) for v in group_vars])
        current_x = local_positions[-1] + group_separation
        vtype = type_sr.split("_")[0]
        if vtype == "r":
            bar_colors.extend(["tab:green"] * len(group_vars))
        elif vtype == "theta":
            bar_colors.extend(["tab:orange"] * len(group_vars))
        else:
            bar_colors.extend(["gray"] * len(group_vars))

    return np.array(x_positions), x_labels_grouped, bar_colors


def plot_sobol_first_order(S1, S1_conf, problem, filename, title=""):
    x_positions, x_labels_grouped, bar_colors = _build_plot_labels(problem)

    plt.figure(figsize=(8, 6))
    plt.bar(x_positions, S1, yerr=S1_conf, capsize=4, color=bar_colors, edgecolor="black")
    plt.xticks(x_positions, x_labels_grouped, rotation=45, ha='right')
    plt.ylabel("S1")
    plt.title(title)
    legend_elements = [
        Patch(facecolor="tab:green", edgecolor="black", label=r"$r$ variables"),
        Patch(facecolor="tab:orange", edgecolor="black", label=r"$\theta$ variables")
    ]
    plt.legend(handles=legend_elements)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_sobol_total_effect(ST, ST_conf, problem, filename, title=""):
    x_positions, x_labels_grouped, bar_colors = _build_plot_labels(problem)

    plt.figure(figsize=(8, 6))
    plt.bar(x_positions, ST, yerr=ST_conf, capsize=4, color=bar_colors, edgecolor="black")
    plt.xticks(x_positions, x_labels_grouped, rotation=45, ha='right')
    plt.ylabel("ST")
    plt.title(title)
    legend_elements = [
        Patch(facecolor="tab:green", edgecolor="black", label=r"$r$ variables"),
        Patch(facecolor="tab:orange", edgecolor="black", label=r"$\theta$ variables")
    ]
    plt.legend(handles=legend_elements)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_sobol_second_order(S2, problem, filename_base, title=""):
    _, x_labels, _ = _build_plot_labels(problem)
    names = problem["names"]

    # -------------------------------------------------
    # Identify indices by variable TYPE only
    # -------------------------------------------------
    idx_r = []
    idx_theta = []

    for i, name in enumerate(names):
        base, _ = name.split("_")
        if base.startswith("theta"):
            idx_theta.append(i)
        else:
            idx_r.append(i)

    labels_r = [x_labels[i] for i in idx_r]
    labels_theta = [x_labels[i] for i in idx_theta]

    # -------------------------------------------------
    # Global color scale (shared across all plots)
    # -------------------------------------------------
    vmin = np.nanmin(S2)
    vmax = np.nanmax(S2)

    def plot_heatmap(matrix, xlbls, ylbls, title, fname):
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            matrix,
            annot=False,
            cmap="Reds",
            xticklabels=xlbls,
            yticklabels=ylbls,
            vmin=vmin,
            vmax=vmax,
        )
        plt.title(title)
        plt.tight_layout()
        plt.savefig(fname, dpi=300)
        plt.close()

    # -------------------------------------------------
    # 1) Full matrix
    # -------------------------------------------------
    plot_heatmap(
        S2,
        x_labels,
        x_labels,
        title,
        filename_base,
    )

    # -------------------------------------------------
    # 2) r–r interactions
    # -------------------------------------------------
    plot_heatmap(
        S2[np.ix_(idx_r, idx_r)],
        labels_r,
        labels_r,
        title,
        filename_base.replace(".png", "_rr.png"),
    )

    # -------------------------------------------------
    # 3) θ–θ interactions
    # -------------------------------------------------
    plot_heatmap(
        S2[np.ix_(idx_theta, idx_theta)],
        labels_theta,
        labels_theta,
        title,
        filename_base.replace(".png", "_thetatheta.png"),
    )

    # -------------------------------------------------
    # 4) r–θ interactions
    # -------------------------------------------------
    plot_heatmap(
        S2[np.ix_(idx_r, idx_theta)],
        labels_theta,
        labels_r,
        title,
        filename_base.replace(".png", "_rtheta.png"),
    )



# phasor_dlr/estimation/sobol/plotting.py
import matplotlib.pyplot as plt

def plot_sobol_sweep(cos_angles, S1_r_sum, S1_r_conf,
                      S1_theta_sum, S1_theta_conf,
                      S1_inter_sum, S1_inter_conf,
                      save_path="Sobol/sobol_S1_sum_vs_cos_angle.png",
                      title=""):
    """
    Plot the first-order Sobol index sweep vs cos(angle) with error bars.
    
    Parameters
    ----------
    cos_angles : array_like
        Cosine of the voltage angles (x-axis)
    S1_r_sum : array_like
        Summed first-order Sobol indices for r variables
    S1_r_conf : array_like
        Confidence intervals for r variables
    S1_theta_sum : array_like
        Summed first-order Sobol indices for theta variables
    S1_theta_conf : array_like
        Confidence intervals for theta variables
    S1_inter_sum : array_like
        Summed interaction terms (1 - sum_r - sum_theta)
    S1_inter_conf : array_like
        Confidence intervals for interactions
    save_path : str
        Path to save the figure
    """
    plt.figure(figsize=(7,5))
    plt.errorbar(cos_angles, S1_r_sum, yerr=S1_r_conf, fmt='-o', capsize=4, lw=2, label=r"$\sum S_1(r)$", color='tab:green')
    plt.errorbar(cos_angles, S1_theta_sum, yerr=S1_theta_conf, fmt='-s', capsize=4, lw=2, label=r"$\sum S_1(\theta)$", color='tab:orange')
    plt.errorbar(cos_angles, S1_inter_sum, yerr=S1_inter_conf, fmt='-^', capsize=4, lw=2, label=r"$\mathrm{Interactions}$", color='tab:red')
    
    plt.xlabel(r"$\cos(\phi^{(r)})$")
    plt.ylabel("Summed first-order Sobol index")
    plt.title(title)
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
