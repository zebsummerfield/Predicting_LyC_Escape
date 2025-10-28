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
from scipy import stats

# True for weighted least squares regression, False for orthogonal distance regression
least_squares = False

folder = "final_rf_model/"
file5 = folder + 'f_esc_rf_observational_charlotte_test_train.json'
file6 = folder + 'n_esc_rf_observational_charlotte_test_train.json'

with fits.open("prosp_properties_GOODSN.fits") as hdul1:
    data1 = {name: hdul1[1].data[name] for name in hdul1[1].data.columns.names}
with fits.open("prosp_properties_GOODSS.fits") as hdul2:
    data2 = {name: hdul2[1].data[name] for name in hdul2[1].data.columns.names}
obvs_data = {key: np.concatenate([data1[key], data2[key]]) for key in data1}

obvs_redshift = obvs_data['z']
obvs_star_mass = 10**(np.array(obvs_data['log(Mstar)']).astype('float64'))
obvs_sfr10 = np.array(obvs_data['SFR10'])
obvs_sfr100 = np.array(obvs_data['SFR100'])
obvs_uv_mag = np.array(obvs_data['M1500int'])

obvs_ssfr10 = ssfr_func(obvs_sfr10, obvs_star_mass)
obvs_ssfr100 = ssfr_func(obvs_sfr100, obvs_star_mass)
log10_offset10 = np.log10(obvs_ssfr10) - sfms_func(np.array([obvs_redshift, np.log10(obvs_star_mass)]), s[0], b[0], u[0])

random_variable = np.random.uniform(low=0, high=1, size=(len(obvs_redshift)))

obvs_f_esc_vars = [obvs_ssfr10, obvs_ssfr100, obvs_ssfr10/obvs_ssfr100, obvs_star_mass, random_variable, 1+obvs_redshift, random_variable]
obvs_f_esc_keys = ['offset10', 'ssfr100', 'ssfr10/ssfr100', 'star_mass', 'uv_mag', '1 + redshift', 'random_variable']
obvs_n_esc_vars = [obvs_sfr10, obvs_sfr100, obvs_star_mass, random_variable, 1+obvs_redshift, random_variable]
obvs_n_esc_keys = ['sfr10', 'sfr100', 'star_mass', 'uv_mag', '1 + redshift', 'random_variable']

obvs_log_f_esc_vars = np.log10(obvs_f_esc_vars).astype('float32')
obvs_log_n_esc_vars = np.log10(obvs_n_esc_vars).astype('float32')

# replaces ssfr10 with offset from the star forming main sequence over 10Myrs
obvs_log_f_esc_vars[0] = log10_offset10.astype('float32')
# replaces the luminosities with magnitudes
obvs_log_f_esc_vars[4] = obvs_uv_mag.astype('float32')
obvs_log_n_esc_vars[3] = obvs_uv_mag.astype('float32')

# removes any rows that have zero, nan or infinity for the vars
bad_indices = []
for i in range(len(obvs_f_esc_vars)):
    b_i = [index for index, val in enumerate(list((obvs_f_esc_vars)[i]))
                    if (val == 0 or val == np.inf or val== -np.inf or val == np.nan)]
    print(f"feature {i+1} bad rows: {len(b_i)}")
    bad_indices += b_i
bad_indices = list(set(bad_indices))[::-1]
obvs_redshift = np.delete(obvs_redshift, bad_indices)
obvs_f_esc_vars = np.delete(obvs_f_esc_vars, bad_indices, axis=1)
obvs_n_esc_vars = np.delete(obvs_n_esc_vars, bad_indices, axis=1)
obvs_log_f_esc_vars = np.delete(obvs_log_f_esc_vars, bad_indices, axis=1)
obvs_log_n_esc_vars = np.delete(obvs_log_n_esc_vars, bad_indices, axis=1)
obvs_star_mass_high_error = np.delete(np.array(obvs_data['log(Mstar)_ehi']), bad_indices)
obvs_star_mass_low_error = np.delete(np.array(obvs_data['log(Mstar)_elo']), bad_indices)
obvs_uv_mag_high_error = np.delete(np.array(obvs_data['M1500int_ehi']), bad_indices)
obvs_uv_mag_low_error = np.delete(np.array(obvs_data['M1500int_elo']), bad_indices)
print(f'rows remaining: {len(obvs_f_esc_vars[i])}')

vars = [obvs_f_esc_vars, obvs_n_esc_vars]
log_vars = [obvs_log_f_esc_vars, obvs_log_n_esc_vars]
keys = [obvs_f_esc_keys, obvs_n_esc_keys]
print(keys)

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
mpl.rcParams.update({'font.size': 20})
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for i_1 in range(len(axes)):

    x = [obvs_log_f_esc_vars[4], obvs_log_f_esc_vars[3]][i_1]
    x_str = ['$M_\mathrm{UV}$', '$\mathrm{Log}_{10}(M_{*} \; [\mathrm{M}_\odot])$'][i_1]
    x_str_no_unit = ['$M_\mathrm{UV}$', '$\mathrm{Log}_{10}(M_{*})$'][i_1]
    x_range = ([-31, -13], [5, 13])[i_1]

    for i_2 in range(len(axes)):
        print((i_1, i_2))

        # each row contains the data for a single galaxy
        X = np.transpose(log_vars[i_2])

        # run target prediction using loaded model
        f_or_n_str = ['$\mathrm{Log}_{10}(f_\mathrm{esc})$',
                      '$\mathrm{Log}_{10}(\dot{n}_\mathrm{ion,esc} \; [\mathrm{s^{-1}}])$'][i_2]
        f_or_n_str_no_unit = ['$\mathrm{Log}_{10}(f_\mathrm{esc})$',
                              '$\mathrm{Log}_{10}(\dot{n}_\mathrm{ion,esc})$'][i_2]
        loaded_model = joblib.load(folder + ['f_esc', 'n_esc'][i_2] + '_rf_observational_charlotte_model.pkl')
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

        error_bars = axes[i_2][i_1].errorbar(x, y,
                                             xerr=(x_err_low, x_err_high),
                                             yerr=y_err,
                                             fmt='none', ecolor=(0.8, 0.8, 0.8, 0.8),
                                             elinewidth=0.4, zorder=2)
        z = obvs_redshift
        sorted_indices = np.argsort(z)
        scatter = axes[i_2][i_1].scatter(x[sorted_indices], y[sorted_indices], alpha=0.9,
                                         c=z[sorted_indices], cmap='inferno', s=5, zorder=3,
                                         #vmin=3, vmax=10
                                         )

        y_range = ([-3.5, -0.5], [48, 54])[i_2]
        axes[i_2][i_1].set_ylabel(f_or_n_str)
        axes[i_2][i_1].set_xlabel(x_str)
        axes[i_2][i_1].set_xlim(x_range)
        axes[i_2][i_1].set_ylim(y_range)

        if least_squares:
            # Fit using Weighted Least Squares Regression, accounting only for y errors
            def linear_1var(X, a, b):
                return a * X + b
            
            popt, pcov = curve_fit(linear_1var, x, y, sigma=y_err, absolute_sigma=True)
            a, b = popt
            a_err, b_err = np.sqrt(np.diag(pcov))

            x_fit = np.linspace(x_range[0], x_range[1], 100)
            y_fit = linear_1var(x_fit, a, b)

        else:
            # Fit using Orthogonal Distance Regression (ODR) to account for errors in both x and y
            def linear_1var(p, X):
                a, b = p
                return a * X + b
            
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

        # plot the best fit line
        fit = axes[i_2][i_1].plot(x_fit, y_fit, c='teal', alpha=0.8, zorder=4)
        # Add confidence bands (regions of possible fit based on parameter errors)
        n_std = 5
        y_upper = np.zeros_like(x_fit)
        y_lower = np.zeros_like(x_fit)
        for i, xi in enumerate(x_fit):
            x_vec = np.array([xi, 1])
            var_y = x_vec @ pcov @ x_vec
            std_error = np.sqrt(var_y)
            y_upper[i] = y_fit[i] + n_std * std_error
            y_lower[i] = y_fit[i] - n_std * std_error
        axes[i_2][i_1].fill_between(x_fit, y_lower, y_upper, color='teal', alpha=0.4, zorder=1,
                    label='95% Confidence Band')

        a_str, a_err_str = round_to_error(a, a_err)
        b_str, b_err_str = round_to_error(b, b_err)
        fit_label = f'{f_or_n_str_no_unit} = ({a_str}$\pm${a_err_str}) {x_str_no_unit} + ({b_str}$\pm${b_err_str})'
        axes[i_2][i_1].text(0.1, 0.03, fit_label, ha='left', va='bottom',
                            transform=axes[i_2][i_1].transAxes, fontsize=12)

        axes[i_2][i_1].set_box_aspect(1)
        # add grid lines in background of graph
        axes[i_2][i_1].grid(True, alpha=0.6, linestyle='--')
        axes[i_2][i_1].set_axisbelow(True)

fig.tight_layout(w_pad=5)
cbar = fig.colorbar(scatter, ax=axes, orientation='vertical', aspect=30)
cbar.set_label("$z$")

mpl.rcParams['figure.dpi'] = 500
folder = "final_graph_generation/"
fig.savefig(folder + "report_graphs/report_graph.png", bbox_inches='tight', dpi=500)
plt.show()