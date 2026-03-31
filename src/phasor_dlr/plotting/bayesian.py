import matplotlib.pyplot as plt
import os

def prior_label(prior):
    return "SI" if prior else "WI"

def plot_rmse_vs_N(
    results,
    levels,
    keys,
    Nlist,
    folder,
    suffix="default",
    hdi_threshold=5,
):  
    l_idx = 1
    colors = plt.cm.tab10.colors
    linestyles = ["-", "--", ":"]
    lws = [1, 1.5, 2.25] if len(levels) > 1 else [1.5]
    markers = ["o", "s", "^"] if len(keys) > 1 else [""]

    fig, ax = plt.subplots(figsize=(12, 4))

    for i_idx, prior in enumerate(results.keys()):
        for k_idx, key in enumerate(keys):
            if key not in results[prior]:
                continue

            Nvals = sorted(results[prior][key].keys())

            rmse_vals = []
            N_plot = []
            for N in Nvals:
                rmse = results[prior][key][N]["rmse"]
                hdi  = results[prior][key][N]["hdi"][levels[-1]]
                N_plot.append(N)
                rmse_vals.append(rmse)
                if hdi <= hdi_threshold:
                    break
            ax.plot(
                N_plot,
                rmse_vals,
                linestyle=linestyles[i_idx % len(linestyles)],
                marker=markers[k_idx % len(markers)],
                color='tab:blue',
                lw=lws[l_idx % len(lws)],
                label=f"{prior_label(prior)}, CT:{key}",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (number of measurements)")
    ax.set_ylabel(r"Posterior RMSE of $T_{\mathrm{EST}}$ [C]")
    ax.set_title("RMSE vs number of measurements")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    filename = os.path.join(folder, f"rmse_trends_{suffix}.png")
    plt.savefig(filename, dpi=300)
    print(f"Saved RMSE trend figure to {filename}")


def plot_hdi_vs_N(
    results,
    levels,
    keys,
    Nlist,
    folder,
    suffix="default",
    hdi_threshold=5,
):
    colors = plt.cm.tab10.colors
    linestyles = ["-", "--", ":"]
    lws = [1, 1.5, 2.25] if len(levels) > 1 else [1.5]
    markers = ["o", "s", "^"] if len(keys) > 1 else [""]

    fig, ax = plt.subplots(figsize=(12, 4))

    for i_idx, prior in enumerate(results.keys()):
        for k_idx, key in enumerate(keys):
            if key not in results[prior]:
                continue

            Nvals = sorted(results[prior][key].keys())

            for l_idx, level in enumerate(levels):
                hdi_plot = []
                N_plot = []

                for N in Nvals:
                    hdi = results[prior][key][N]["hdi"][level]

                    N_plot.append(N)
                    hdi_plot.append(hdi)

                    if hdi <= hdi_threshold:
                        break

                ax.plot(
                    N_plot,
                    hdi_plot,
                    linestyle=linestyles[i_idx % len(linestyles)],
                    marker=markers[k_idx % len(markers)],
                    lw=lws[l_idx % len(lws)],
                    color='tab:blue',
                    label=f"{prior_label(prior)}, CT:{key}, {int(level*100)}% HDI",
                )

    ax.plot(
        [min(Nlist), max(Nlist)],
        [hdi_threshold, hdi_threshold],
        color="red",
        linestyle=":",
        label=rf"$T_{{\mathrm{{NOM}}}} \pm {hdi_threshold}\mathrm{{C}}$",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Measurements")
    ax.set_ylabel(r"HDI half-width [$^\circ$C]")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    filename = os.path.join(folder, f"hdi_trends_{suffix}.png")
    plt.savefig(filename, dpi=300)
    print(f"Saved HDI trend figure to {filename}")
