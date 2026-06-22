import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from functions import *
from scipy.optimize import curve_fit
from matplotlib.ticker import MaxNLocator

# True for plotting residuals against variables, False for plotting predicted target against variables
residuals = False

# True for 2D histogram, False for scatter plot
histogram = True

# Whether to use the random forest models trained with the dusty or dust-free Thesan-Zoom data
dusty = True

folder = "final_rf_model/"
if dusty:
    file = 'cat_dusttestszeb_fdust.hdf5'
else:
    file = 'cat.hdf5'
#file = 'cat_dusttestszeb_g2.hdf5'
keys, log_vars, log_f_esc, log_n_esc = prepare_data(file, f_or_n=1, obvs=True, dusty=dusty, eps=True, ssfr50_cut=False, add_vars=['redshift_full'])
print(len(log_f_esc))
print(keys)

def linear_1var(p, X):
    a, b = p
    return a * X + b

def linear_2var(p, X):
    a, b, c = p
    x, z = X
    return a * x + b * z + c

def curve_fit_func_1var(X, a, b):
    return linear_1var((a, b), X)

def curve_fit_func_2var(X, a, b, c):
    return linear_2var((a, b, c), X)


plt.style.use('./MNRAS_Style.mplstyle')
mpl.rcParams.update({'font.size': 22})
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# sorts the data by redshift for plotting
uv_mag = log_vars[3]
log_star_mass = log_vars[2]
redshift = 10**log_vars[-1]
sorted_indices = np.argsort(redshift)

for i_1 in range(len(axes)):

    x = [uv_mag, log_star_mass][i_1]
    x_str = ['$M_\mathrm{UV}$', '$\mathrm{Log}_{10}(M_{*} \; [\mathrm{M}_\odot])$'][i_1]
    x_str_no_unit = ['$M_\mathrm{UV}$', '$\mathrm{Log}_{10}(M_{*})$'][i_1]
    # x_range = (([-27, -11], [5, 12])[i_1], ([-23, -13], [6, 11])[i_1])[histogram]
    x_range = ([-23, -13], [6, 11])[i_1]

    for i_2 in range(len(axes)):
        print('')
        ax = axes[i_2, i_1]
        y = [log_f_esc, log_n_esc][i_2]
        y_range = ([-5, -0], [44.5, 54.5])[i_2]
        f_or_n_str = ['$\mathrm{Log}_{10}(f_\mathrm{esc})$',
                      '$\mathrm{Log}_{10}(\dot{N}_\mathrm{ion,esc} \; [\mathrm{s^{-1}}])$'][i_2]
        f_or_n_str_no_unit = ['$\mathrm{Log}_{10}(f_\mathrm{esc})$',
                              '$\mathrm{Log}_{10}(\dot{N}_\mathrm{ion,esc})$'][i_2]
        print((x_str_no_unit, f_or_n_str_no_unit))

        # Fit using only galaxies from Thesan with M_UV <= -13 or log_10(M_*) >= 6
        selection = np.where(
            [(uv_mag <= -13), (log_star_mass >= 6.0)][i_1]
            )[0]
        print('thesan rows used in fitting:', len(selection))
        
        # Fit using Weighted Least Squares Regression
        popt, pcov = curve_fit(curve_fit_func_1var, x[selection], y[selection])
        a, b = popt
        a_err, b_err = np.sqrt(np.diag(pcov))

        # Print fit parameters for Weighted Least Squares Regression including redshift dependence
        popt_2, pcov_2 = curve_fit(curve_fit_func_2var, (x[selection], np.log10(1+redshift[selection])), y[selection])
        a_2, b_2, c_2 = popt_2
        a_2_err, b_2_err, c_2_err = np.sqrt(np.diag(pcov_2))
        a_2_str, a_2_err_str = round_to_error(a_2, a_2_err)
        b_2_str, b_2_err_str = round_to_error(b_2, b_2_err)
        c_2_str, c_2_err_str = round_to_error(c_2, c_2_err)
        print((
            rf'log10({["f_esc", "N_ion,esc"][i_2]}) = '
            rf'({a_2_str}±{a_2_err_str}){["Muv", "M*"][i_1]} + '
            rf'({b_2_str}±{b_2_err_str})log10(1+z) + '
            rf'({c_2_str}±{c_2_err_str})'
        ))

        if i_1 == 0 and i_2 == 1:
            total_emissivity = np.mean(10**y[selection].astype('float64'))
            predicted_emissivity = np.mean(10**linear_2var((a_2, b_2, c_2), (x[selection], np.log10(1+redshift[selection]))).astype('float64'))
            print(f"Mean emissivity from data: {total_emissivity:.3e}")
            print(f"Mean emissivity from fit: {predicted_emissivity:.3e}")

        y_residuals = y - linear_2var((a_2, b_2, c_2), (x, np.log10(1+redshift)))
        std_y = np.sqrt((1/ (len(x[selection])- 2)) * np.sum(y_residuals[selection]**2))
        print(f"Standard deviation of residuals: {std_y}")
        p16, p84 = np.percentile(y_residuals, [16, 84])
        sigma_16_84 = (p84 - p16) / 2
        print(f"16th-84th percentile range: {sigma_16_84}")

        popt_3, pcov_3 = curve_fit(curve_fit_func_2var, (y[selection], np.log10(1+redshift[selection])), x[selection])
        a_3, b_3, c_3 = popt_3
        x_residuals = x - linear_2var((a_3, b_3, c_3), (y, np.log10(1+redshift)))
        std_x = np.sqrt((1/ (len(x[selection])- 2)) * np.sum(x_residuals[selection]**2))
        print(f"Fit reveresd standard deviation of residuals: {std_x}")

        if i_2 == 1:
            if not residuals:
                ax.text((0.95, 0.05)[i_1], 0.95, r'$\sigma_{\dot{N}} = $' + f'{sigma_16_84:.3f}',
                    ha=('right', 'left')[i_1], va='top', transform=ax.transAxes, fontsize=19, color=('black', 'white')[histogram])
            else:
                ax.text((0.95, 0.905)[i_1], 0.05, r'$\sigma_{\dot{N}} = $' + f'{sigma_16_84:.3f}',
                    ha=('left', 'right')[i_1], va='bottom', transform=ax.transAxes, fontsize=19, color=('black', 'white')[histogram])

        if residuals:
            if not histogram:
                # plots the residual scatter points
                scatter = ax.scatter(x[sorted_indices], y_residuals[sorted_indices], alpha=0.9,
                                                c=redshift[sorted_indices], cmap='viridis', s=1, zorder=3,
                                                vmin=3, vmax=16
                                                )
            else:
                # creates 2D histogram instead of scatter plot
                nbins = 50
                x_bins = np.linspace(x_range[0], x_range[1], nbins+1)
                y_bins = np.linspace(-3, 3, nbins+1)
                hist, xedges, yedges = np.histogram2d(x, y_residuals, bins=[x_bins, y_bins])
                hist = hist.T
                # hist = np.ma.masked_where(hist == 0, hist)
                hist = np.log10(hist)
                hist[hist == -np.inf] = 0
                h_plot = ax.imshow(hist, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                                origin='lower', aspect='auto', cmap='viridis', interpolation='nearest', zorder=1)
                
            # adds a horizontal line at y=0
            fit = ax.axhline(0, c='teal', alpha=0.8, zorder=4)
        
            # seperates the galaxies into bins of x with each containing equal numbers of galaxies
            nbins = 20
            bins = np.quantile(x, np.linspace(0, 1, nbins + 1))
            bin_indices = np.digitize(x, bins)
            x_medians, y_medians, = ([], [])
            y_16th, y_84th = ([], [])
            for i in range(1, len(bins)):
                bin_mask = bin_indices == i
                x_medians.append(np.median(x[bin_mask]))
                y_medians.append(np.median(y[bin_mask]))
                redshift_medians = np.median(redshift[bin_mask])
                y_16th.append(np.percentile(y[bin_mask], 16))
                y_84th.append(np.percentile(y[bin_mask], 84))
            log_redshift_medians = np.log10(1 + redshift_medians)

            # # Calculate standard deviation across the Muv bins
            # if i_1 == 0 and i_2 == 1:
            #     y_residuals_bins = [y_residuals[bin_indices == i] for i in range(1, nbins)]
            #     stds = [np.sqrt((1/ (len(residuals)- 2)) * np.sum(residuals**2)) for residuals in y_residuals_bins]
            #     print("Standard deviations in Muv bins:")
            #     for i in range(len(stds)):
            #         print(f"[{bins[i]:.1f}, {bins[i+1]:.1f}): {stds[i]:.3f}")
            
            # plots the median of x against the median of residuals for each bin
            ax.plot(x_medians, np.array(y_medians) - linear_2var((a_2, b_2, c_2), (np.array(x_medians), log_redshift_medians)), 
                    c='r', linewidth=3, alpha=0.8, label="median residual", zorder=4)
            ax.fill_between(x_medians, np.array(y_16th) - linear_2var((a_2, b_2, c_2), (np.array(x_medians), log_redshift_medians)),
                            np.array(y_84th) - linear_2var((a_2, b_2, c_2), (np.array(x_medians), log_redshift_medians)),
                            color='r', alpha=0.2, label="16th-84th percentile", zorder=4)
            ax.set_ylim((-3, 3))
            if i_1 == 0:
                ax.set_ylabel(f_or_n_str + " Residuals", labelpad=10)

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
                               origin='lower', aspect='auto', cmap='inferno', interpolation='nearest', zorder=1)

            if i_2 == 1:
                # Plots the best fit line with confidence bands (regions of possible fit based on parameter errors)
                x_fit = np.linspace(x_range[0], x_range[1], 100)
                y_fit = linear_1var((a, b), x_fit)
                fit = ax.plot(x_fit, y_fit, c='darkcyan', linewidth=3, alpha=0.8, zorder=4)

                # n_std = 5
                # y_upper = np.zeros_like(x_fit)
                # y_lower = np.zeros_like(x_fit)
                # for i, xi in enumerate(x_fit):
                #     x_vec = np.array([xi, 1])
                #     var_y = x_vec @ pcov @ x_vec
                #     std_error = np.sqrt(var_y)
                #     y_upper[i] = y_fit[i] + n_std * std_error
                #     y_lower[i] = y_fit[i] - n_std * std_error
                # ax.fill_between(x_fit, y_lower, y_upper, color='teal', alpha=0.4, zorder=4,
                #             label='95% Confidence Band')
                
                # adding fit parameters as text to the graph
                A, B = a_2, b_2
                A_err, B_err = a_2_err, b_2_err
                if i_1 == 0:
                    # Muv fit
                    C = c_2 - 20*a_2 + np.log10(7) * b_2  # normalizing at Muv = -20 and z = 6
                    # TC = np.array([-20, np.log10(7), 1])
                    # C_err = np.sqrt(TC @ pcov_2 @ TC.T)
                    C_err = np.sqrt(c_2_err**2 + (-20 * a_2_err)**2 + (np.log10(7) * b_2_err)**2)
                else:
                    # stellar mass fit
                    C = c_2 + 10*a_2 + np.log10(7) * b_2  # normalizing at log10(M*) = 10 and z = 6
                    # TC = np.array([10, np.log10(7), 1])
                    # C_err = np.sqrt(TC @ pcov_2 @ TC.T)
                    C_err = np.sqrt(c_2_err**2 + (10 * a_2_err)**2 + (np.log10(7) * b_2_err)**2)

                A_str, A_err_str = round_to_error(A, A_err)
                B_str, B_err_str = round_to_error(B, B_err)
                C_str, C_err_str = round_to_error(C, C_err)
                # z_str = r'$\mathrm{log}_{10}\left(\frac{1+z}{7}\right)$'
                z_str = r'$\mathrm{log}_{10}((1+z)/7)$'
                fit_label = (
                    r"$\begin{aligned}"
                    f"{f_or_n_str_no_unit.replace('$', '')} &= ({A_str}\\pm{A_err_str}) \, ({x_str_no_unit.replace('$', '')} {['+ 20', '- 10'][i_1]}) \\\\"
                    f"&+ ({B_str}\\pm{B_err_str}) \, {z_str.replace('$', '')} \\\\"
                    f"&+ ({C_str}\\pm{C_err_str})"
                    r"\end{aligned}$"
                )
                ax.text(0.025, 0.025, fit_label, ha='left', va='bottom',
                        transform=ax.transAxes, fontsize=16, color=('black', 'white')[histogram])
        

        # axes setup
            if i_1 == 0:
                ax.set_ylabel(f_or_n_str, labelpad=10)
            ax.set_ylim(y_range)
            ax.tick_params(color='white')
        if i_2 == 1:
            ax.set_xlabel(x_str)
        ax.set_xlim(x_range)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
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
    cbar.set_label(("$N_\mathrm{gal}$", "$\log_{10}(N_\mathrm{gal})$")[residuals], labelpad=5)

mpl.rcParams['figure.dpi'] = 500
folder = "final_graph_generation/"
fig.savefig(folder + "report_graphs/report_graph." + ("jpg", "pdf")[histogram], bbox_inches='tight', dpi=500)
plt.show()