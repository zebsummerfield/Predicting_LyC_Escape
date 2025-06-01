import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from functions import *

# 0 for f_esc, 1 for n_esc
f_or_n = 1

# True if model is generated to predict for an observational catalogue 
obvs = False

folder = "final_graph_generation/"
file = 'cat.hdf5'
keys, log_vars, log_f_esc, log_n_esc = prepare_data(file, f_or_n=f_or_n, obvs=obvs, eps=False)

print(keys)
x_index = 6
log_x = log_vars[x_index]

plt.style.use('./MNRAS_Style.mplstyle')
mpl.rcParams.update({'font.size': 20})
# Create figure with GridSpec to have better control over subplots and colorbar
fig = plt.figure(figsize=(15, 5))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 0.2, 1, 0.05], wspace=0.2)  # The last slice is for colorbar
axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 2])]
cbar_ax = fig.add_subplot(gs[0, 3])  # Dedicated axes for colorbar

for ax_i in range(len(axes)):
    log_y = [log_f_esc, log_n_esc][ax_i]
    f_or_n_str = ['$\mathrm{Log}_{10}(f_{\mathrm{esc}})$',
                  '$\mathrm{Log}_{10}(\dot{n}_{\mathrm{ion,esc}} \; [\mathrm{s^{-1}}])$'][ax_i]
    y_limits = ((-5, 0), (46, 54))[ax_i]

    # plots a 2d histogram of log_x against log_y where the number of galaxies in a bin dictates it's colour
    nbins = 100
    hist, xedges, yedges = np.histogram2d(log_x, log_y, bins=nbins, range=((min(log_x), max(log_x)), y_limits))
    hist = hist.T
    hist = np.log10(hist)
    hist[hist == -np.inf] = 0
    h1 = axes[ax_i].imshow(hist, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                    origin='lower', aspect='auto', cmap='viridis', interpolation='nearest', vmin=0, vmax=2.3)
    axes[ax_i].set_xlabel("$M_{\mathrm{UV}}$")
    axes[ax_i].set_ylabel(f_or_n_str)

    # seperates the galaxies into bins of variable log_x with each containing equal numbers of galaxies 
    nbins = 50
    bins = np.quantile(log_x, np.linspace(0, 1, nbins + 1))
    bin_indices = np.digitize(log_x, bins)
    x_medians, y_medians = ([], [])
    for i in range(1, len(bins)):
        bin_mask = bin_indices == i
        x_medians.append(np.median(log_x[bin_mask]))
        y_medians.append(np.median(log_y[bin_mask]))

    # plots the median of log_x against the median of log_y for each bin
    axes[ax_i].plot(x_medians, y_medians, c='r', linewidth=3, alpha=0.9, label="median $f_{esc}$")
    axes[ax_i].set_xlim(min(x_medians), max(x_medians))
    axes[ax_i].set_ylim(y_limits)
    axes[ax_i].set_box_aspect(1)
    axes[ax_i].grid(False)
    #axes[ax_i].grid(True, alpha=0.6, linestyle='--')

cbar = fig.colorbar(h1, label="$\mathrm{Log}_{10}(\mathrm{N_{bin}})$",
                    cax=cbar_ax, fraction=0.046)

mpl.rcParams['figure.dpi'] = 500
fig.savefig(folder + "report_graphs/report_graph.png", bbox_inches='tight', dpi=500)
plt.show()