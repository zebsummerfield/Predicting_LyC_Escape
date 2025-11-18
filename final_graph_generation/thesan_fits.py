import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import joblib
import json
from sklearn.metrics import mean_absolute_error
from functions import *
from scipy.optimize import curve_fit
import math
from astropy.io import fits
from scipy import odr
from matplotlib.ticker import MaxNLocator

# True for plotting residuals against variables, False for plotting predicted target against variables
residuals = False

# True for 2D histogram, False for scatter plot
histogram = True

folder = "final_rf_model/"
file = 'cat.hdf5'
keys, log_vars, log_f_esc, log_n_esc = prepare_data(file, f_or_n=0, obvs=True, eps=True, add_vars=['redshift_full'])
print(keys)

def linear_1var(p, X):
    a, b = p
    return a * X + b

def curve_fit_func(X, a, b):
    return linear_1var((a, b), X)

def round_to_error(value, error):
    if error == 0:
        return f"{value:.3f}", f"{error:.3f}"
    # Determine number of decimal places based on error's significant digit
    order = int(math.floor(math.log10(abs(error))))
    rounded_value = round(value, -order)
    rounded_error = round(error, -order)
    if rounded_error >= 10 ** (order + 1):
        order += 1
        rounded_error = round(error, -order)
    format_str = f"{{:.{-order}f}}"
    return format_str.format(rounded_value), format_str.format(rounded_error)


plt.style.use('./MNRAS_Style.mplstyle')
mpl.rcParams.update({'font.size': 18})
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# sorts the data by redshift for plotting
redshift = 10**log_vars[-1]
sorted_indices = np.argsort(redshift)

for i_1 in range(len(axes)):

    x = [log_vars[4], log_vars[3]][i_1]
    x_str = ['$M_\mathrm{UV}$', '$\mathrm{Log}_{10}(M_{*} \; [\mathrm{M}_\odot])$'][i_1]
    x_str_no_unit = ['$M_\mathrm{UV}$', '$\mathrm{Log}_{10}(M_{*})$'][i_1]
    x_range = ([-24, -11], [6, 11])[i_1]

    for i_2 in range(len(axes)):
        print((i_1, i_2))
        ax = axes[i_2, i_1]
        y = [log_f_esc, log_n_esc][i_2]
        y_range = ([-5, -0], [45, 55])[i_2]
        f_or_n_str = ['$\mathrm{Log}_{10}(f_\mathrm{esc})$',
                      '$\mathrm{Log}_{10}(\dot{n}_\mathrm{ion,esc} \; [\mathrm{s^{-1}}])$'][i_2]
        f_or_n_str_no_unit = ['$\mathrm{Log}_{10}(f_\mathrm{esc})$',
                              '$\mathrm{Log}_{10}(\dot{n}_\mathrm{ion,esc})$'][i_2]
        
        # Fit using Weighted Least Squares Regression

        popt, pcov = curve_fit(curve_fit_func, x, y)
        a, b = popt
        a_err, b_err = np.sqrt(np.diag(pcov))

        x_fit = np.linspace(x_range[0], x_range[1], 100)
        y_fit = linear_1var((a, b), x_fit)

        if residuals:
            # plots the residual scatter points
            scatter = ax.scatter(x[sorted_indices], y[sorted_indices] - linear_1var((a, b), x[sorted_indices]), alpha=0.9,
                                            c=redshift[sorted_indices], cmap='viridis', s=1, zorder=3,
                                            vmin=3, vmax=16
                                            )
            # adds a horizontal line at y=0
            fit = ax.axhline(0, c='teal', alpha=0.8, zorder=4)
        
            # seperates the galaxies into bins of x with each containing equal numbers of galaxies
            nbins = 25
            bins = np.quantile(x, np.linspace(0, 1, nbins + 1))
            bin_indices = np.digitize(x, bins)
            x_medians, y_medians = ([], [])
            y_16th, y_84th = ([], [])
            for i in range(1, len(bins)):
                bin_mask = bin_indices == i
                x_medians.append(np.median(x[bin_mask]))
                y_medians.append(np.median(y[bin_mask]))
                y_16th.append(np.percentile(y[bin_mask], 16))
                y_84th.append(np.percentile(y[bin_mask], 84))
            
            # plots the median of x against the median of residuals for each bin
            ax.plot(x_medians, np.array(y_medians) - linear_1var((a, b), np.array(x_medians)), 
                    c='r', linewidth=3, alpha=0.8, label="median residual", zorder=4)
            ax.fill_between(x_medians, np.array(y_16th) - linear_1var((a, b), np.array(x_medians)),
                            np.array(y_84th) - linear_1var((a, b), np.array(x_medians)),
                            color='r', alpha=0.2, label="16th-84th percentile", zorder=4)

        else:
            if not histogram:
                # plots the scatter points
                scatter = ax.scatter(x[sorted_indices], y[sorted_indices], alpha=0.9, c=redshift[sorted_indices],
                                     cmap='inferno', s=1, zorder=3, rasterized=True,
                                     vmin=3, vmax=16
                                     )

            else:
                # creates 2D histogram instead of scatter plot
                nbins = 50
                x_bins = np.linspace(x_range[0], x_range[1], nbins+1)
                y_bins = np.linspace(y_range[0], y_range[1], nbins+1)
                hist, xedges, yedges = np.histogram2d(x, y, bins=[x_bins, y_bins])
                hist = hist.T
                # hist = np.ma.masked_where(hist == 0, hist)
                # hist = np.log10(hist)
                hist[hist == -np.inf] = 0
                h_plot = ax.imshow(hist, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                               origin='lower', aspect='auto', cmap='magma', interpolation='nearest', zorder=1)

            # Plots the best fit line with confidence bands (regions of possible fit based on parameter errors)
            fit = ax.plot(x_fit, y_fit, c='teal', linewidth=2, alpha=0.8, zorder=4)

            n_std = 5
            y_upper = np.zeros_like(x_fit)
            y_lower = np.zeros_like(x_fit)
            for i, xi in enumerate(x_fit):
                x_vec = np.array([xi, 1])
                var_y = x_vec @ pcov @ x_vec
                std_error = np.sqrt(var_y)
                y_upper[i] = y_fit[i] + n_std * std_error
                y_lower[i] = y_fit[i] - n_std * std_error
            ax.fill_between(x_fit, y_lower, y_upper, color='teal', alpha=0.4, zorder=4,
                        label='95% Confidence Band')
            
            # adding fit parameters as text to the graph
            a_str, a_err_str = round_to_error(a, a_err)
            b_str, b_err_str = round_to_error(b, b_err)
            fit_label = f'{f_or_n_str_no_unit} = ({a_str}$\pm${a_err_str}) {x_str_no_unit} + ({b_str}$\pm${b_err_str})'
            ax.text(0.05, 0.025, fit_label, ha='left', va='bottom',
                    transform=ax.transAxes, fontsize=14, color=('black', 'white')[histogram])
        
            # axes setup
            if i_2 == 1:
                ax.set_xlabel(x_str)
            if i_1 == 0:
                ax.set_ylabel(f_or_n_str, labelpad=10)
            ax.set_xlim(x_range)
            ax.set_ylim(y_range)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
            
        ax.set_box_aspect(1)
        ax.grid(False)
        # ax.grid(True, alpha=0.6, linestyle='--')
        # ax.set_axisbelow(True)

fig.align_ylabels([axes[0, 0], axes[1, 0]])
fig.tight_layout()
if not histogram:
    cbar = fig.colorbar(scatter, ax=axes, orientation='vertical', aspect=30, pad=0.03)
    cbar.set_label("$z$", labelpad=10)
else:
    cbar = fig.colorbar(h_plot, ax=axes, orientation='vertical', aspect=30, pad=0.03)
    cbar.set_label("Number of Galaxies in Bin", labelpad=5)

mpl.rcParams['figure.dpi'] = 500
folder = "final_graph_generation/"
fig.savefig(folder + "report_graphs/report_graph." + ("jpg", "png")[histogram], bbox_inches='tight', dpi=500)
plt.show()