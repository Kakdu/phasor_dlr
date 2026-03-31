import numpy as np
import matplotlib.pyplot as plt


methods = ["PM-DLR", "WB-DLR\n(min)", "WB-DLR\n(max)"]
values = [0.8607, 1.57, 2.61]

plt.figure()
plt.bar(methods, values)
plt.ylabel("Temperature RMSE (°C)")
plt.tight_layout()
plt.show()



labels = ["Mounted sensing\n(95% CI)", "PM-DLR\n(95% CI)", "WB-DLR\n(75% CI)"]

# confidence interval half-widths
mounted_ci = 1.0              # ±1 °C
pm_ci = 1.96 * 0.751          # ±1.47 °C
wb_ci = 2.23                  # ±2.23 °C (75%)

lower = [-mounted_ci, -pm_ci, -wb_ci]
upper = [mounted_ci, pm_ci, wb_ci]

x = np.arange(len(labels))

plt.figure()

for i in range(len(labels)):
    plt.fill_between([x[i]-0.25, x[i]+0.25], lower[i], upper[i], alpha=0.4)

plt.axhline(0)

plt.xticks(x, labels)
plt.ylabel("Temperature deviation (°C)")

plt.tight_layout()
plt.show()