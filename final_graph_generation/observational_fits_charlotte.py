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

# True for weighted least squares regression, False for orthogonal distance regression
least_squares = False

# True to use Monte Carlo to estimate fit parameter uncertainties, False to use covariance matrix from fit
mc = False

# True for plotting residuals against variables, False for plotting predicted target against variables
residuals = False

# True for 2D histogram, False for scatter plot
histogram = False

# Whether to use the random forest models trained with the dusty or dust-free Thesan-Zoom data
dusty = True

folder = "final_rf_model/"
if not dusty:
    file5 = folder + 'f_esc_rf_observational_charlotte_test_train.json'
    file6 = folder + 'n_esc_rf_observational_charlotte_test_train.json'
else:
    file5 = folder + 'f_esc_rf_observational_charlotte_dusty_test_train.json'
    file6 = folder + 'n_esc_rf_observational_charlotte_dusty_test_train.json'

with fits.open("prosp_properties_GOODSS.fits") as hdul1:
    data1 = {name: hdul1[1].data[name] for name in hdul1[1].data.columns.names}
with fits.open("prosp_properties_GOODSN.fits") as hdul2:
    data2 = {name: hdul2[1].data[name] for name in hdul2[1].data.columns.names}
obvs_data = {key: np.concatenate([data1[key], data2[key]]) for key in data1}
print(obvs_data.keys())

ID = obvs_data['ID']
obvs_redshift = obvs_data['z']
obvs_star_mass = 10**(np.array(obvs_data['log(Mstar)']).astype('float64'))
obvs_sfr10 = np.array(obvs_data['SFR10'])
obvs_sfr100 = np.array(obvs_data['SFR100'])
obvs_uv_mag = np.array(obvs_data['M1500obs'])
# obvs_int_uv_mag = np.array(obvs_data['M1500int'])
# attenuation = obvs_uv_mag - obvs_int_uv_mag

obvs_ssfr10 = ssfr_func(obvs_sfr10, obvs_star_mass)
obvs_ssfr100 = ssfr_func(obvs_sfr100, obvs_star_mass)
log10_offset10 = np.log10(obvs_ssfr10) - sfms_func(np.array([obvs_redshift, np.log10(obvs_star_mass)]), s[0], b[0], u[0])

random_variable = np.random.uniform(low=0, high=1, size=(len(obvs_redshift)))

obvs_f_esc_vars = [obvs_ssfr10, obvs_ssfr10/obvs_ssfr100, obvs_star_mass, random_variable, 1+obvs_redshift, random_variable]
obvs_f_esc_keys = ['offset10', 'ssfr10/ssfr100', 'star_mass', 'uv_mag', '1 + redshift', 'random_variable']
obvs_n_esc_vars = [obvs_sfr10, obvs_sfr100, obvs_star_mass, random_variable, 1+obvs_redshift, random_variable]
obvs_n_esc_keys = ['sfr10', 'sfr100', 'star_mass', 'uv_mag', '1 + redshift', 'random_variable']

obvs_log_f_esc_vars = np.log10(obvs_f_esc_vars).astype('float32')
obvs_log_n_esc_vars = np.log10(obvs_n_esc_vars).astype('float32')

# replaces ssfr10 with offset from the star forming main sequence over 10Myrs
obvs_log_f_esc_vars[0] = log10_offset10.astype('float32')
# add the uv magnitude to the variables
obvs_log_f_esc_vars[3] = obvs_uv_mag.astype('float32')
obvs_log_n_esc_vars[3] = obvs_uv_mag.astype('float32')

# Creates a list of bad IDs for the diffraction spike galaxies 
with fits.open("masked_objects_gs.fits") as hdul1:
    bad_data1 = {name: hdul1[1].data[name] for name in hdul1[1].data.columns.names}
with fits.open("masked_objects_gn.fits") as hdul2:
    bad_data2 = {name: hdul2[1].data[name] for name in hdul2[1].data.columns.names}
bad_ID = np.array(bad_data1['ID'].tolist() + bad_data2['ID'].tolist())
bad_indices = np.where(np.isin(ID, bad_ID))[0].tolist()
print(f"diffraction spike rows: {len(bad_indices)}")

# removes any rows that have zero, nan, or infinity for the vars; signal to noise SN(F444W) < 3; and red_chi2(JWST) > 1
b_i = [index for index, val in enumerate(list(obvs_data['SN(F444W)'])) if val < 3]
print(f"SN(F444W) < 3 rows: {len(b_i)}")
bad_indices += b_i
b_i += [index for index, val in enumerate(list(obvs_data['red_chi2(JWST)'])) if val > 10]
print(f"red_chi2(JWST) > 10 rows: {len(b_i)}")
bad_indices += b_i
for i in range(len(obvs_log_f_esc_vars)):
    b_i = [index for index, val in enumerate(list((obvs_log_f_esc_vars)[i]))
                    if (val == 0 or val == np.inf or val== -np.inf or np.isnan(val))]
    print(f"feature {i+1} bad rows: {len(b_i)}")
    bad_indices += b_i
bad_indices = list(set(bad_indices))[::-1]
obvs_redshift = np.delete(obvs_redshift, bad_indices)
obvs_log_f_esc_vars = np.delete(obvs_log_f_esc_vars, bad_indices, axis=1)
obvs_log_n_esc_vars = np.delete(obvs_log_n_esc_vars, bad_indices, axis=1)
obvs_star_mass_high_error = np.delete(np.array(obvs_data['log(Mstar)_ehi']), bad_indices)
obvs_star_mass_low_error = np.delete(np.array(obvs_data['log(Mstar)_elo']), bad_indices)
obvs_uv_mag_high_error = np.delete(np.array(obvs_data['M1500obs_ehi']), bad_indices)
obvs_uv_mag_low_error = np.delete(np.array(obvs_data['M1500obs_elo']), bad_indices)
print(f'rows remaining: {len(obvs_redshift)}')

log_vars = [obvs_log_f_esc_vars, obvs_log_n_esc_vars]
keys = [obvs_f_esc_keys, obvs_n_esc_keys]
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
sorted_indices = np.argsort(obvs_redshift)

for i_1 in range(len(axes)):

    x = [obvs_log_f_esc_vars[3], obvs_log_f_esc_vars[2]][i_1]
    print(min(x), max(x))
    x_str = ['$M_\mathrm{UV}$', '$\mathrm{Log}_{10}(M_{*} \; [\mathrm{M}_\odot])$'][i_1]
    x_str_no_unit = ['$M_\mathrm{UV}$', '$\mathrm{Log}_{10}(M_{*})$'][i_1]
    x_range = (([-28, -10], [5, 12])[i_1], ([-24, -12], [6, 11])[i_1])[histogram]

    for i_2 in range(len(axes)):
        ax = axes[i_2, i_1]
        y_range = (([-3.5, -0.5], [48, 55])[i_2], ([-3.25, -0.75], [48.5, 54.5])[i_2])[histogram]

        # each row contains the data for a single galaxy
        X = np.transpose(log_vars[i_2])

        # run target prediction using loaded model
        f_or_n_str = ['$\mathrm{Log}_{10}(f_\mathrm{esc})$',
                      '$\mathrm{Log}_{10}(\dot{n}_\mathrm{ion,esc} \; [\mathrm{s^{-1}}])$'][i_2]
        f_or_n_str_no_unit = ['$\mathrm{Log}_{10}(f_\mathrm{esc})$',
                              '$\mathrm{Log}_{10}(\dot{n}_\mathrm{ion,esc})$'][i_2]
        print((x_str_no_unit, f_or_n_str_no_unit))

        if not dusty:
            loaded_model = joblib.load(folder + ['f_esc', 'n_esc'][i_2] + '_rf_observational_charlotte_model.pkl')
        else:
            loaded_model = joblib.load(folder + ['f_esc', 'n_esc'][i_2] + '_rf_observational_charlotte_dusty_model.pkl')
        predictions = loaded_model.predict(X)

        with open((file5, file6)[i_2], 'r') as json_data:
            data = json.load(json_data)
            test = np.array(data['f_esc_test'])
            test_pred = np.array(data['f_esc_test_pred'])

        # seperates the test sample galaxies into bins of predicted test target with each containing equal numbers of galaxies, 
        # then calculates the median of predicted test target and the mean_absolute_error of test target in each bin
        nbins = 50
        bins = np.quantile(test_pred, np.linspace(0, 1, nbins + 1))
        bin_indices = np.digitize(test_pred, bins)
        test_pred_medians = np.zeros(nbins) 
        test_pred_mae  = np.zeros(nbins)
        for i in range(1, len(bins)):
            bin_mask = bin_indices == i
            test_pred_medians[i-1] = np.median(test_pred[bin_mask])
            test_pred_mae[i-1] = mean_absolute_error(test[bin_mask], test_pred[bin_mask])
        
        # finds which mae bin a prediction of the target falls into so its error can be classified as the corresponding mae
        pred_mae = np.zeros(len(predictions))
        for i in range(len(predictions)):
            index = np.argmin(np.abs(test_pred_medians - predictions[i]))
            pred_mae[i] = (test_pred_mae[index])

        y = predictions
        y_err = pred_mae
        x_err_low = [np.abs(obvs_uv_mag_low_error), np.abs(obvs_star_mass_low_error)][i_1]
        x_err_high = [np.abs(obvs_uv_mag_high_error), np.abs(obvs_star_mass_high_error)][i_1]
        x_err_sym = (x_err_low + x_err_high) / 2

        if mc:
            # Monte Carlo to estimate fit parameter uncertainties
            x_err_median = np.median(x_err_sym[x_err_sym > 0]) * 1e-6
            x_err_sym[x_err_sym <= 0] = x_err_median
            mc_iters = 1000
            sub_size = len(x)//5
            betas = np.zeros((mc_iters, 2))

            for k in range(mc_iters):
                # bootstrap indices + perturb by measurement errors
                idx = np.random.choice(len(x), size=sub_size, replace=False)
                xb = x[idx] + np.random.normal(0, x_err_sym[idx])
                yb = y[idx] + np.random.normal(0, y_err[idx])

                if least_squares:
                    # Fit using Weighted Least Squares Regression, accounting only for y errors
                    popt, _ = curve_fit(curve_fit_func, xb, yb, sigma=y_err[idx], absolute_sigma=True)
                    betas[k] = popt[0], popt[1]
                
                else:
                    # Fit using Orthogonal Distance Regression (ODR) to account for errors in both x and y
                    model_mc = odr.Model(linear_1var)
                    data_mc = odr.RealData(xb, yb, sx=x_err_sym[idx], sy=y_err[idx])
                    out_mc = odr.ODR(data_mc, model_mc, beta0=[0, np.median(y)]).run()
                    betas[k] = out_mc.beta

            # MC medians and 16/84 percentiles
            a, b = np.median(betas, axis=0)
            a_16, a_84 = np.percentile(betas[:,0], [16, 84])
            b_16, b_84 = np.percentile(betas[:,1], [16, 84])
            a_err = 0.5 * (a_84 - a_16)
            b_err = 0.5 * (b_84 - b_16)
            pcov = np.cov(betas.T)        

        else:
            if least_squares:
                # Fit using Weighted Least Squares Regression, accounting only for y errors
                popt, pcov = curve_fit(curve_fit_func, x, y, sigma=y_err, absolute_sigma=True)
                a, b = popt
                a_err, b_err = np.sqrt(np.diag(pcov))

            else:
                # Fit using Orthogonal Distance Regression (ODR) to account for errors in both x and y
                x_err_median = np.nanmedian(x_err_sym[x_err_sym > 0]) * 1e-6
                x_err_sym[x_err_sym <= 0] = x_err_median
                model = odr.Model(linear_1var)
                data = odr.RealData(x, y, sx=x_err_sym, sy=y_err)
                odr_inst = odr.ODR(data, model, beta0=[0, np.median(y)])
                out = odr_inst.run()

                a, b = out.beta
                a_err, b_err = out.sd_beta 
                pcov = out.cov_beta

        x_fit = np.linspace(x_range[0], x_range[1], 100)
        y_fit = linear_1var((a, b), x_fit)

        if residuals:
            y_residuals = y - linear_1var((a, b), x)
            # plots the residual scatter points
            scatter = ax.scatter(x[sorted_indices], y_residuals[sorted_indices], alpha=0.9,
                                            c=obvs_redshift[sorted_indices], cmap='viridis', s=1, zorder=3,
                                            vmin=3, vmax=9
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
            
            std_uv = np.sqrt((1/ (len(x)- 2)) * np.sum(y_residuals**2))
            print(std_uv)

        else:
            if not histogram:
                # plots the error bars and the scatter points
                error_bars = ax.errorbar(x[sorted_indices], y[sorted_indices],
                                            xerr=(x_err_low[sorted_indices], x_err_high[sorted_indices]),
                                            yerr=y_err[sorted_indices], fmt='none',
                                            ecolor=(0.7, 0.7, 0.7, 1), # RGBAlpha
                                            elinewidth=0.2, zorder=2, rasterized=True)
                
                scatter = ax.scatter(x[sorted_indices], y[sorted_indices], alpha=0.9, c=obvs_redshift[sorted_indices],
                                     cmap='inferno', s=1, zorder=3, rasterized=True,
                                     vmin=3, vmax=9
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