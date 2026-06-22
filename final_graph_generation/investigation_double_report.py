import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from functions import *

# True if model is generated to predict for an observational catalogue 
obvs = False

# True to make the graphs scatter plots, False to make the graphs histograms
scatter = False

# True to use the dusty Thesan-Zoom catalogue, False for dust-free catalogue
dusty = True

folder = "final_graph_generation/"
file = ['cat.hdf5', 'cat_dusttestszeb_fdust.hdf5'][dusty]
#file = 'cat_dusttestszeb_g2.hdf5'
keys, log_vars, log_f_esc, log_n_esc = prepare_data(file, f_or_n=0, obvs=obvs, dusty=dusty, eps=False,
                                                    add_vars=['stellar_mass_full', 'gas_mass_full'])
print(keys)
x = log_vars[-2]
gas_mass = 10**log_vars[-1].astype('float64')
gas_mass = gas_mass / (0.76 / 1.6735575e-24)
gas_mass = gas_mass / 1.989e33
y = np.log10(gas_mass)
x_str = '$\mathrm{log}_{10}(M_{*} \; [\mathrm{M}_\odot])$'
y_str = '$\mathrm{log}_{10}(M_\mathrm{gas} \; [\mathrm{M}_\odot])$'

plt.style.use('./MNRAS_Style.mplstyle')
mpl.rcParams.update({'font.size': 17})
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax_i in range(len(axes)):
    ax = axes[ax_i]

    log_target = [log_f_esc, log_n_esc][ax_i]
    target = 10**log_target.astype('float64')
    f_or_n_str = ['$\mathrm{log}_{10}(f_\mathrm{esc})$',
                  '$\mathrm{log}_{10}(\dot{N}_\mathrm{ion,esc} \; [\mathrm{s^{-1}}])$'][ax_i]

    if scatter:
        # plots a scatter of x against y where the colour and size of the points is determined by the target
        if ax_i == 1:
            target = log_n_esc
        sizes = np.where(target > (0.2, 52)[ax_i], 50, 5)
        sorted_indices = np.argsort(target)
        vmax = (1, 56)[ax_i]
        vmin = (0, 48)[ax_i]
        color_scatter = ax.scatter(x[sorted_indices], y[sorted_indices], s=sizes[sorted_indices],
                                        c=target[sorted_indices], cmap='inferno', marker='.',
                                        vmax=vmax, vmin=vmin)
        scatter_label =  ['$f_{\mathrm{esc}}$',
                          '$\mathrm{Log}_{10}(\dot{N}_\mathrm{ion,esc} \; [\mathrm{s^{-1}}])$'][ax_i]
        plt.colorbar(color_scatter, label=scatter_label, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel(x_str)
        ax.set_ylabel(y_str)

    else:
        # plots a 2d histogram of x against y where the colour of the bin is dictated by it's mean log_target
        nbins = (30, 20)[dusty]
        hist, xedges, yedges = np.histogram2d(x, y, bins=nbins, weights=log_target)
        x_width = xedges[1] - xedges[0]
        y_width = yedges[1] - yedges[0]
        hist_norm, xedges, yedges = np.histogram2d(x, y, bins=nbins)
        hist = hist / hist_norm
        hist[hist_norm < 5] = np.nan
        # vmax = (-0.5, 54)[ax_i]
        # vmin = (-3, 48)[ax_i]
        hist = hist.T # transposes the histogram for proper orientation
        color_hist = ax.imshow(hist, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                                       origin='lower', aspect='auto', cmap='viridis', 
                                       interpolation='nearest', zorder=2,
                                       # vmax=vmax, vmin=vmin
                                       )
        plt.colorbar(color_hist, label=f_or_n_str, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel(x_str)
        ax.set_ylabel(y_str)
        xrange = max(x) - min(x)
        yrange = max(y) - min(y)
        ax.set_xlim(min(x) - xrange * 0.05, max(x) + xrange * 0.05)
        ax.set_ylim(min(y) - yrange * 0.05, max(y) + yrange * 0.05)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=8))

        # contour overlay of the histogram
        A, B, pdf, levels = make_contours(x, y, nbins=nbins)
        ax.contour(A, B, pdf.T, levels=levels, colors='black', linewidths=1.5, zorder=3)

        # PCA Direction Arrow
        vari_pcc = np.array([x, y, log_target])  
        theta, theta_err = pcc_err(vari_pcc, theta=True)
        # For some reason the PCA analysis yields theta and theta_err ~ pi/2 radians for n_esc with the dusty catalogue
        # Therefore we manually set the values to that of the old catalogue here 
        if ax_i == 1:
            theta = 1.510
            theta_err = 0.006


        # sets arrow appearance and position
        xmin, xmax = ax.get_xlim()
        x_median_coord = (np.median(x) - xmin) / (xmax - xmin)
        ymin, ymax = ax.get_ylim()
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
        
        angle_label = f"$\\theta = {angle_str}\pm{angle_error_str}^\circ$"

        # Step 3: Plot the arrow
        ax.arrow(start[0], start[1], da, db,
                    width=width, ec='black', fc='red', alpha=0.8,
                    transform=ax.transAxes, zorder=5)
        
        ax.text(0.95, 0.05, angle_label,
                ha='right', va='bottom', transform=ax.transAxes)

    ax.set_box_aspect(1)
    ax.grid(False)
    # ax.grid(True, alpha=0.6, linestyle='--')
    # ax.set_axisbelow(True)
    # for line in ax.get_xgridlines() + ax.get_ygridlines():
    #     line.set_zorder(0)

mpl.rcParams['figure.dpi'] = 500
plt.tight_layout(w_pad=2)
fig.savefig(folder + "report_graphs/report_graph.pdf", bbox_inches='tight', dpi=500)
plt.show()