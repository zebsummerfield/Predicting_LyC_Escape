import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from functions import *

# 0 for f_esc, 1 for n_esc
f_or_n = 0

# True if model is generated to predict for an observational catalogue 
obvs = False

folder = "final_graph_generation/"
file = 'cat.hdf5'
ratio_vars = ['sfr_full_10', 'sfr_full_100',
              'gas_mass_full', 'stellar_mass_full',
              'stellar_mass_full', 'M_vir_full',
              'uv_lum_int_full', 'ha_lum_int_full',
              'sfr_size_full', 'stellar_size_full']
keys, log_vars, log_f_esc, log_n_esc = prepare_data(file, f_or_n=f_or_n, obvs=obvs, eps=False,
                                                    add_vars=ratio_vars)
f_or_n_str = ['$\mathrm{Log}_{10}(f_\mathrm{esc})$',
              '$\mathrm{Log}_{10}(\dot{n}_\mathrm{ion,esc} \; [\mathrm{s^{-1}}])$'][f_or_n]
log_target = [log_f_esc, log_n_esc][f_or_n]

y_vars = [log_vars[-10], log_vars[-8], log_vars[-6], log_vars[-4], log_vars[-2]]
x_vars = [log_vars[-9], log_vars[-7], log_vars[-5], log_vars[-3], log_vars[-1]]

gas_mass = 10**y_vars[1].astype('float64')
gas_mass = gas_mass / (0.76 / 1.6735575e-24)
gas_mass = gas_mass / 1.989e33
y = np.log10(gas_mass)
y = log_vars[1]

y_strs = ['$\mathrm{SFR}_{10}$',
          '$M_\mathrm{gas}$',
          '$M_*$',
          '$L_\mathrm{UV}$',
          '$R_\mathrm{SFR}$']
x_strs = ['$\mathrm{SFR}_{100}$',
          '$M_*$',
          '$M_\mathrm{vir}$',
          '$L_\mathrm{H\\alpha}$',
          '$R_{M_*}$']

plt.style.use('./MNRAS_Style.mplstyle')
mpl.rcParams.update({'font.size': 20})
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
fig.delaxes(axes[-1])

for index in range(5):
    print(f'Compiling figure : {index+1}')
    ratio = log_vars[index]
    ax = axes[index]
    ax = axes[index]
    x = x_vars[index]
    y = y_vars[index]
    x_str = '$\mathrm{log}_{10}$(' + x_strs[index] + ')'
    y_str = '$\mathrm{log}_{10}$(' + y_strs[index] + ')'

    # plots a 2d histogram of x against y where the colour of the bin is dictated by it's mean log_target
    nbins = 30
    hist, xedges, yedges = np.histogram2d(x, y, bins=nbins, weights=log_target)
    x_width = xedges[1] - xedges[0]
    y_width = yedges[1] - yedges[0]
    hist_norm, xedges, yedges = np.histogram2d(x, y, bins=nbins)
    hist = hist / hist_norm
    hist[hist_norm < 5] = np.nan
    hist = hist.T # transposes the histogram for proper orientation
    color_hist = ax.imshow(hist, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                                origin='lower', aspect='auto', cmap='viridis', 
                                vmax=-0.5, vmin=-2.5, interpolation='nearest', zorder=2)
    ax.set_xlabel(x_str)
    ax.set_ylabel(y_str)


    # contour overlay of the histogram
    A, B, pdf, levels = make_contours(x, y, nbins=nbins)
    ax.contour(A, B, pdf.T, levels=levels, colors='black', linewidths=1.5, zorder=3)

    # PCA Direction Arrow
    vari_pcc = np.array([x, y, log_target])  
    theta, theta_err = pcc_err(vari_pcc, theta=True)
    if index == 2:
        theta = theta * 5
    if index == 4:
        theta = - theta * 20
    print(f'Arrow Angle : {theta}')

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
    
    angle_label = f"$\\theta$={angle_str}$\pm${angle_error_str}$^\circ$"

    # Step 3: Plot the arrow
    ax.arrow(start[0], start[1], da, db,
                width=width, ec='black', fc='red', alpha=0.8,
                transform=ax.transAxes, zorder=5)
    
    ax.text(0.95, 0.05, angle_label,
            ha='right', va='bottom', fontsize=16,
            transform=ax.transAxes)

    ax.set_box_aspect(1)
    # add grid lines in background of graph
    ax.grid(True, alpha=0.6, linestyle='--')
    ax.set_axisbelow(True)
    for line in ax.get_xgridlines() + ax.get_ygridlines():
        line.set_zorder(0)  # lower z-order than image

fig.tight_layout(w_pad=3, h_pad=3)
cbar = fig.colorbar(color_hist, ax=axes, orientation='vertical', aspect=25)
cbar.set_label(f_or_n_str)
mpl.rcParams['figure.dpi'] = 500
fig.savefig(folder + "report_graphs/report_graph.png", bbox_inches='tight', dpi=500)
plt.show()