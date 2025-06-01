import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from functions import *

# 0 for f_esc, 1 for n_esc
f_or_n = 0

# True if model is generated to predict for an observational catalogue 
obvs = False

# True to make the graphs scatter plots, False to make the graphs histograms
scatter = False

folder = "final_graph_generation/"
file = 'cat.hdf5'
keys, log_vars, log_f_esc, log_n_esc = prepare_data(file, f_or_n=f_or_n, obvs=obvs, eps=False,
                                                    add_vars=['stellar_mass_full', 'gas_mass_full'])

print(keys)
x = log_vars[-2]
gas_mass = 10**log_vars[-1].astype('float64')
gas_mass = gas_mass / (0.76 / 1.6735575e-24)
gas_mass = gas_mass / 1.989e33
y = np.log10(gas_mass)
y = log_vars[-1]
x_str = '$\mathrm{Log}_{10}(M_{*} \; [\mathrm{M}_\odot])$'
y_str = '$\mathrm{Log}_{10}(M_\mathrm{gas} \; [\mathrm{M}_\odot])$'

plt.style.use('./MNRAS_Style.mplstyle')
mpl.rcParams.update({'font.size': 20})
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

for ax_i in range(len(axes)):

    log_target = [log_f_esc, log_n_esc][ax_i]
    target = 10**log_target.astype('float64')
    f_or_n_str = ['$\mathrm{Log}_{10}(f_\mathrm{esc})$',
                  '$\mathrm{Log}_{10}(\dot{n}_\mathrm{ion,esc} \; [\mathrm{s^{-1}}])$'][ax_i]

    if scatter:
        # plots a scatter of x against y where the colour and size of the points is determined by the target
        if ax_i == 1:
            target = log_n_esc
        sizes = np.where(target > (0.2, 52)[ax_i], 50, 5)
        sorted_indices = np.argsort(target)
        vmax = (1, 56)[ax_i]
        vmin = (0, 48)[ax_i]
        color_scatter = axes[ax_i].scatter(x[sorted_indices], y[sorted_indices], s=sizes[sorted_indices],
                                        c=target[sorted_indices], cmap='inferno', marker='.',
                                        vmax=vmax, vmin=vmin)
        scatter_label =  ['$f_{\mathrm{esc}}$',
                          '$\mathrm{Log}_{10}(\dot{n}_\mathrm{ion,esc} \; [\mathrm{s^{-1}}])$'][ax_i]
        plt.colorbar(color_scatter, label=scatter_label, ax=axes[ax_i], fraction=0.046, pad=0.04)
        axes[ax_i].set_xlabel(x_str)
        axes[ax_i].set_ylabel(y_str)

    else:
        # plots a 2d histogram of x against y where the colour of the bin is dictated by it's mean log_target
        nbins = 30
        hist, xedges, yedges = np.histogram2d(x, y, bins=nbins, weights=log_target)
        x_width = xedges[1] - xedges[0]
        y_width = yedges[1] - yedges[0]
        hist_norm, xedges, yedges = np.histogram2d(x, y, bins=nbins)
        hist = hist / hist_norm
        hist[hist_norm < 5] = np.nan
        vmax = (-0.5, 54)[ax_i]
        vmin = (-2.5, 48)[ax_i]
        hist = hist.T # transposes the histogram for proper orientation
        color_hist = axes[ax_i].imshow(hist, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                                    origin='lower', aspect='auto', cmap='viridis', 
                                    vmax=vmax, vmin=vmin, interpolation='nearest', zorder=2)
        plt.colorbar(color_hist, label=f_or_n_str, ax=axes[ax_i], fraction=0.046, pad=0.04)
        axes[ax_i].set_xlabel(x_str)
        axes[ax_i].set_ylabel(y_str)


        # contour overlay of the histogram
        A, B, pdf, levels = make_contours(x, y, nbins=nbins)
        axes[ax_i].contour(A, B, pdf.T, levels=levels, colors='black', linewidths=1.5, zorder=3)

        # PCA Direction Arrow
        vari_pcc = np.array([x, y, log_target])  
        theta, theta_err = pcc_err(vari_pcc, theta=True)

        # sets arrow appearance and position
        xmin, xmax = axes[ax_i].get_xlim()
        x_median_coord = (np.median(x) - xmin) / (xmax - xmin)
        ymin, ymax = axes[ax_i].get_ylim()
        y_median_coord = (np.median(y) - ymin) / (ymax - ymin)
        start = (x_median_coord, y_median_coord)
        r = 0.2
        width = 0.0125

        # calculates the end points of the arrow
        if theta < 0:
            (da, db) = (r * np.sin(-theta), -r * np.cos(-theta))
            angle_str = f"{round(theta * 180 / np.pi + 180, 1)}"
            angle_error_str = f"{round(theta_err * 180 / np.pi, 1)}"
        else:
            (da, db) = (r * np.sin(theta), r * np.cos(theta))
            angle_str = f"{round(theta * 180 / np.pi, 1)}"
            angle_error_str = f"{round(theta_err * 180 / np.pi, 1)}"
        
        angle_label = f"$\\theta$={angle_str}$\pm${angle_error_str}$^\circ$"

        # Step 3: Plot the arrow
        axes[ax_i].arrow(start[0], start[1], da, db,
                    width=width, ec='black', fc='red', alpha=0.8,
                    transform=axes[ax_i].transAxes, zorder=5)
        
        axes[ax_i].text(0.95, 0.10, angle_label,
                ha='right', va='bottom', fontsize=24,
                transform=axes[ax_i].transAxes)

    axes[ax_i].set_box_aspect(1)
    axes[ax_i]
    # add grid lines in background of graph
    axes[ax_i].grid(True, alpha=0.6, linestyle='--')
    axes[ax_i].set_axisbelow(True)
    for line in axes[ax_i].get_xgridlines() + axes[ax_i].get_ygridlines():
        line.set_zorder(0)  # lower z-order than image

mpl.rcParams['figure.dpi'] = 500
plt.tight_layout(w_pad=5)
fig.savefig(folder + "report_graphs/report_graph.png", bbox_inches='tight', dpi=500)
plt.show()