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
log_y = [log_f_esc, log_n_esc][f_or_n]
f_or_n_str = ['$\mathrm{Log}_{10}(f_{\mathrm{esc}})$',
                '$\mathrm{Log}_{10}(\dot{n}_{\mathrm{ion,esc}} \; [\mathrm{s^{-1}}])$'][f_or_n]
y_limits = ((-5, 0), (46, 54))[f_or_n]

f_strs = ['$\Delta\mathrm{MS}_{10}$', '$\mathrm{SFR}_{10}/\mathrm{SFR}_{100}$', '$M_*$',
            '$M_\mathrm{gas}/M_*$', '$M_*/M_\mathrm{vir}$', '$Z$',
            '$L_\mathrm{UV}$', '$L_\mathrm{UV}/L_\mathrm{H\\alpha}$', '$R_\mathrm{UV}$',
            '$R_\mathrm{H\\alpha}$', '$R_\mathrm{SFR}$', '$R_\mathrm{SFR}/R_{M_*}$',
            '$\Sigma_\mathrm{SFR_{10}}$', '$1+z$', 'Rand']
n_strs = ['$\mathrm{SFR}_{10}$', '$\mathrm{SFR}_{100}$', '$M_*$',
          '$M_\mathrm{gas}/M_*$', '$M_*/M_\mathrm{vir}$', '$Z$',
          '$L_\mathrm{UV}$', '$L_\mathrm{H\\alpha}$', '$1+z$',
          'Rand']
var_strs = [f_strs, n_strs][f_or_n]

plt.style.use('./MNRAS_Style.mplstyle')
mpl.rcParams.update({'font.size': 20})
# Create figure with GridSpec to have better control over subplots and colorbar
fig, axes = plt.subplots((3, 2)[f_or_n], 5, figsize=((24, 12), (24, 8))[f_or_n])
axes = axes.flatten()

for index in range((15, 10)[f_or_n]):
    log_x = log_vars[index]
    ax = axes[index]
    # plots a 2d histogram of log_x against log_y where the number of galaxies in a bin dictates it's colour
    nbins = 100
    hist, xedges, yedges = np.histogram2d(log_x, log_y, bins=nbins, range=((min(log_x), max(log_x)), y_limits))
    hist = hist.T
    hist = np.log10(hist)
    hist[hist == -np.inf] = 0
    h1 = ax.imshow(hist, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                    origin='lower', aspect='auto', cmap='viridis', interpolation='nearest', vmin=0, vmax=2.3)
    ax.set_xlabel('$\mathrm{log}_{10}$(' + var_strs[index] + ')')
    if index in [0, 5, 10]:
        ax.set_ylabel(f_or_n_str)

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
    ax.plot(x_medians, y_medians, c='r', linewidth=3, alpha=0.9, label="median $f_{esc}$")
    ax.set_xlim(min(x_medians), max(x_medians))
    ax.set_ylim(y_limits)
    ax.set_box_aspect(1)
    ax.grid(False)

fig.tight_layout(w_pad=2.5)
cbar = fig.colorbar(h1, ax=axes, orientation='vertical', aspect=(30, 20)[f_or_n])
cbar.set_label("$\mathrm{Log}_{10}(\mathrm{N_{bin}})$")
mpl.rcParams['figure.dpi'] = 500
fig.savefig(folder + "report_graphs/report_graph.png", bbox_inches='tight', dpi=500)
plt.show()