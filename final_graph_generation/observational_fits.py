import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import joblib
import json
from sklearn.metrics import mean_absolute_error
from functions import *
from scipy.optimize import curve_fit
import math
from astropy.cosmology import Planck15 as cosmo

folder = "final_rf_model/"
file3 = folder + 'f_esc_rf_observational_test_train.json'
file4 = folder + 'n_esc_rf_observational_test_train.json' 

obvs = True

if obvs:
    with open('lola_cat.txt', 'r') as f:
        # extract fixed UV flux from data loaded to a dictionary
        # with open('UV_fluxes_prospector.txt', 'r') as f_fix_1:
        #     lines_fix_1 = f_fix_1.readlines()
        #     obvs_keys_fix_1 = lines_fix_1[0].strip().split()
        #     print(obvs_keys_fix_1)
        #     values_fix_1 = []
        #     for line_fix_1 in lines_fix_1[1:]:
        #         values_fix_1.append(line_fix_1.strip().split())
        # with open('UV_fluxes_prospector_fresco.txt', 'r') as f_fix_2:
        #     lines_fix_2 = f_fix_2.readlines()
        #     obvs_keys_fix_2 = lines_fix_2[0].strip().split()
        #     print(obvs_keys_fix_2)
        #     values_fix_2 = []
        #     for line_fix_2 in lines_fix_2[1:]:
        #         values_fix_2.append(line_fix_2.strip().split())
        # obvs_data_fix = dict(zip(obvs_keys_fix_1, np.array(values_fix_1 + values_fix_2).T.astype(float)))
        # obvs_uv_flux = obvs_uv_flux*3631 /1e23 /(1 + obvs_redshift)

        # extract fixed UV flux from data loaded to a dictionary
        with open('fixed_fluxes.txt', 'r') as f_fix:
            lines_fix = f_fix.readlines()
            obvs_keys_fix = lines_fix[0].strip().split()
            print(obvs_keys_fix)
            values_fix = []
            for line_fix in lines_fix[1:]:
                values_fix.append(line_fix.strip().split())
            obvs_data_fix = dict(zip(obvs_keys_fix, np.array(values_fix).T.astype(float)))

            id_fix = np.array(obvs_data_fix['ID'])
            sorted_fix = np.argsort(id_fix)
            obvs_uv_flux = np.array(obvs_data_fix['fUV_flux'])[sorted_fix]
            sorted_non_zero = np.where(obvs_uv_flux > 0)
            obvs_uv_flux = obvs_uv_flux[sorted_non_zero]

        # loads the observational data from the text file and converts it to a dictionary
        lines = f.readlines()
        obvs_keys = lines[0].strip().split()
        print(obvs_keys)
        values = []
        for line in lines[1:]:
            values.append(line.strip().split())
        obvs_data = dict(zip(obvs_keys, np.array(values).T.astype(float)))

        id = np.array(obvs_data['ID'])
        sorted = np.argsort(id)

        obvs_redshift = np.array(obvs_data['z_spec'])[sorted][sorted_non_zero]
        print(len(obvs_redshift))
        log_obvs_star_mass = np.array(obvs_data['logmstar'])[sorted][sorted_non_zero]
        obvs_ha_flux = 10**np.array(obvs_data['log_Ha_flux'])[sorted][sorted_non_zero]
        # obvs_uv_flux = np.array(obvs_data['UV_flux'])[sorted][sorted_non_zero]
        obvs_sfr10 = 10**np.array(obvs_data['logSFR10'])[sorted][sorted_non_zero]
        obvs_sfr100 = 10**np.array(obvs_data['logSFR100'])[sorted][sorted_non_zero]
        obvs_ha_size = np.array(obvs_data['r_eff_Ha_kpc'])[sorted][sorted_non_zero]
        obvs_uv_size = np.array(obvs_data['r_eff_UV_kpc'])[sorted][sorted_non_zero]
        obvs_star_mass_high_error = np.array(obvs_data['logmstar_ehigh'])[sorted][sorted_non_zero]
        obvs_star_mass_low_error = np.array(obvs_data['logmstar_elow'])[sorted][sorted_non_zero]

        obvs_star_mass = 10**log_obvs_star_mass
        obvs_ssfr10 = ssfr_func(obvs_sfr10, obvs_star_mass)
        obvs_ssfr100 = ssfr_func(obvs_sfr100, obvs_star_mass)
        log10_offset10 = np.log10(obvs_ssfr10) - sfms_func(np.array([obvs_redshift, log_obvs_star_mass]), s[0], b[0], u[0])

        # converts the fluxes to magnitudes
        obvs_muv_app = -2.5 * np.log10(obvs_uv_flux) - 48.6
        obvs_mha_app = -2.5 * np.log10(obvs_ha_flux) - 48.6
        dist_mod = cosmo.distmod(obvs_redshift)
        obvs_uv_mag = np.array(obvs_muv_app - dist_mod.value)
        obvs_ha_mag = np.array(obvs_mha_app - dist_mod.value)
        print(np.mean(obvs_uv_mag))

        random_variable = np.random.uniform(low=0, high=1, size=(len(obvs_redshift)))

        obvs_f_esc_vars = [obvs_ssfr10, obvs_ssfr10/obvs_ssfr100, obvs_star_mass, obvs_uv_flux,
                        obvs_uv_flux/obvs_ha_flux, obvs_uv_size, obvs_ha_size, 1+obvs_redshift, random_variable]
        obvs_f_esc_keys = ['offset10', 'ssfr10/ssfr100', 'star_mass', 'uv_mag',
                        'uv_flux/ha_flux', 'uv_size', 'ha_size', '1 + redshift', 'random_variable']
        obvs_n_esc_vars = [obvs_sfr10, obvs_sfr100, obvs_star_mass, obvs_uv_flux, obvs_ha_flux,
                        1+obvs_redshift, random_variable]
        obvs_n_esc_keys = ['sfr10', 'sfr100', 'star_mass', 'uv_mag', 'ha_mag',
                        '1 + redshift', 'random_variable']
        
        obvs_log_f_esc_vars = np.log10(obvs_f_esc_vars).astype('float32')
        obvs_log_n_esc_vars = np.log10(obvs_n_esc_vars).astype('float32')
        
        # replaces ssfr10 with offset from the star forming main sequence over 10Myrs
        obvs_log_f_esc_vars[0] = log10_offset10.astype('float32')
        # replaces the luminosities with magnitudes
        obvs_log_f_esc_vars[3] = obvs_uv_mag.astype('float32')
        obvs_log_n_esc_vars[2] = obvs_uv_mag.astype('float32')
        obvs_log_n_esc_vars[3] = obvs_ha_mag.astype('float32')

        vars = [obvs_f_esc_vars, obvs_n_esc_vars]
        log_vars = [obvs_log_f_esc_vars, obvs_log_n_esc_vars]
        keys = [obvs_f_esc_keys, obvs_n_esc_keys]

else:
    # loads both the catalogue of galaxies and their variables
    with h5py.File('cat.hdf5', 'r') as hdf:

        f_esc = np.array(hdf['f_esc_vir_full']).astype('float32')
        n_esc = np.array(hdf['Ndot_LyC_vir_full'])

        redshift = np.array(hdf['redshift_full'])
        star_mass = np.array(hdf['stellar_mass_full'])
        sfr10 = np.array(hdf['sfr_full_10'])
        sfr50 = np.array(hdf['sfr_full_50'])
        sfr100 = np.array(hdf['sfr_full_100'])
        ssfr10 = ssfr_func(sfr10, star_mass)
        ssfr50 = ssfr_func(sfr50, star_mass)
        ssfr100 = ssfr_func(sfr100, star_mass)
        vir_mass = np.array(hdf['M_vir_full'])
        gas_mass = np.array(hdf['gas_mass_full'])
        ha_lum = np.array(hdf['ha_lum_int_full']) * 5
        uv_lum = np.array(hdf['uv_lum_int_full'])
        # gas_met = np.array(hdf['gas_met_full'])
        star_met = np.array(hdf['star_met_full'])
        star_size = np.array(hdf['stellar_size_full'])
        sfr_size = np.array(hdf['sfr_size_full'])
        ha_size = np.array(hdf['ha_size_int_full'])
        uv_size = np.array(hdf['uv_size_int_full'])
        uv_projection = np.array(hdf['uv_size_int_2d_full'])
        ha_projection = np.array(hdf['ha_size_int_2d_full'])
        sfr10 = np.array(hdf['sfr_full_10'])
        sfr10_density = sfr10 / (np.pi * sfr_size**2)
        # fixing gas mass units
        gas_mass = gas_mass / (0.76 / 1.6735575e-24)
        gas_mass = gas_mass / 1.989e33

        random_variable = np.random.uniform(low=0, high=1, size=(len(f_esc)))

        f_esc_vars = [ssfr10, sfr10/ssfr100, star_mass, uv_lum,
                        uv_lum/ha_lum, uv_projection, ha_projection, 1+redshift, random_variable]
        f_esc_keys = ['offset10', 'ssfr10/ssfr100', 'star_mass', 'uv_mag',
                        'uv_flux/ha_flux', 'uv_size', 'ha_size', '1 + redshift', 'random_variable']
        n_esc_vars = [sfr10, sfr100, star_mass, uv_lum, ha_lum,
                        1+redshift, random_variable]
        n_esc_keys = ['sfr10', 'sfr100', 'star_mass', 'uv_mag', 'ha_mag',
                        '1 + redshift', 'random_variable']
        
        log_f_esc_vars = np.log10(f_esc_vars).astype('float32')
        log_n_esc_vars = np.log10(n_esc_vars).astype('float32')

        # replaces ssfr10 with offset from the star forming main sequence over 10Myrs
        log_osfms10 =  log_f_esc_vars[0] - sfms_func(np.array([redshift, np.log10(star_mass)]), s[0], b[0], u[0])
        log_f_esc_vars[0] = log_osfms10.astype('float32')

        # replaces the luminosities with magnitudes
        lum_to_tenpc = 4 * np.pi * (10 * 3.086e18)**2
        uv_mag = -2.5 * np.log10((n_esc_vars[3]) / lum_to_tenpc) - 48.6
        ha_mag = -2.5 * np.log10((n_esc_vars[4]) / lum_to_tenpc) - 48.6
        log_f_esc_vars[3] = uv_mag.astype('float32')
        log_n_esc_vars[3] = uv_mag.astype('float32')
        log_n_esc_vars[4] = ha_mag.astype('float32')
        
        # removes any rows that have zero ,unity or infinity for the vars and f_esc
        # ssfr50 is included to remove galaxies that have had no recent star formation
        f_esc[np.isnan(f_esc)] = 0
        for i in range(len(np.concatenate((log_f_esc_vars, log_n_esc_vars, [f_esc], [ssfr50])))):
            nan_indices = [index for index, val in 
                           enumerate(list(np.concatenate((log_f_esc_vars, log_n_esc_vars, [f_esc], [ssfr50]))[i]))
                            if (val == 0 or val == 1 or val == np.inf or val== -np.inf or np.isnan(val))][::-1]
            print(f"rows deleted: {len(nan_indices)}")
            f_esc, n_esc = (np.delete(f_esc, nan_indices), np.delete(n_esc, nan_indices))
            log_f_esc_vars, log_n_esc_vars = (np.delete(log_f_esc_vars, nan_indices, axis=1),
                                              np.delete(log_n_esc_vars, nan_indices, axis=1))
            ssfr10, ssfr50, ssfr100 = (np.delete(ssfr10, nan_indices), 
                                    np.delete(ssfr50, nan_indices),
                                    np.delete(ssfr100, nan_indices))
            redshift = np.delete(redshift, nan_indices)
        print(f'rows remaining: {len(f_esc)}')

        vars = [f_esc_vars, n_esc_vars]
        log_vars = [log_f_esc_vars, log_n_esc_vars]
        keys = [f_esc_keys, n_esc_keys]
        log_f_esc = np.log10(f_esc).astype('float32')
        log_n_esc = np.log10(n_esc).astype('float32')


def linear_1var(X, a, b):
    y = X
    return a * y + b

def round_to_error(value, error):
    if error == 0:
        return f"{value:.3f}", f"{error:.3f}"
    # Determine number of decimal places based on error's significant digit
    digits = -int(math.floor(math.log10(abs(error))))
    rounded_value = round(value, digits)
    rounded_error = round(error, digits)
    format_str = f"{{:.{digits}f}}"
    return format_str.format(rounded_value), format_str.format(rounded_error)

plt.style.use('./MNRAS_Style.mplstyle')
mpl.rcParams.update({'font.size': 20})
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for i_1 in range(len(axes)):

    if obvs:
        x_var = [obvs_uv_mag, log_obvs_star_mass][i_1]
    else:
        x_var = [log_vars[0][3], log_vars[0][2]][i_1]
    x_str = ['$M_\mathrm{UV}$', '$\mathrm{Log}_{10}(M_{*} \; [\mathrm{M}_\odot])$'][i_1]
    x_str_no_unit = ['$M_\mathrm{UV}$', '$\mathrm{Log}_{10}(M_{*})$'][i_1]
    x_range = ([-23, -17], [7, 11])[i_1]

    for i_2 in range(len(axes)):

        # each row contains the data for a single galaxy
        X = np.transpose(log_vars[i_2])

        # run target prediction using loaded model
        f_or_n_str = ['$\mathrm{Log}_{10}(f_\mathrm{esc})$',
                      '$\mathrm{Log}_{10}(\dot{n}_\mathrm{ion,esc} \; [\mathrm{s^{-1}}])$'][i_2]
        f_or_n_str_no_unit = ['$\mathrm{Log}_{10}(f_\mathrm{esc})$',
                              '$\mathrm{Log}_{10}(\dot{n}_\mathrm{ion,esc})$'][i_2]
        loaded_model = joblib.load(folder + ['f_esc', 'n_esc'][i_2] + '_rf_observational_model.pkl')
        predictions = loaded_model.predict(X)

        with open((file3, file4)[i_2], 'r') as json_data:
            data = json.load(json_data)
            test = np.array(data['f_esc_test'])
            test_pred = np.array(data['f_esc_test_pred'])

        # seperates the test sample galaxies into bins of predicted test target with each containing equal numbers of galaxies, 
        # then calculates the median of predicted test target and the mean_absolute_error of test target in each bin
        nbins = 30
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
        
        if obvs:
            error_bars = axes[i_2][i_1].errorbar(x_var, predictions,
                                            xerr=[None, (obvs_star_mass_low_error, obvs_star_mass_high_error)][i_1],
                                            yerr=pred_mae, fmt='none', ecolor=(0.7, 0.7, 0.7, 0.7),
                                            elinewidth=0.7, zorder=2)
            z = obvs_redshift
        else:
            error_bars = axes[i_2][i_1].errorbar(x_var, predictions,
                                            yerr=pred_mae, fmt='none', ecolor=(0.7, 0.7, 0.7, 0.7),
                                            elinewidth=0.7, zorder=2)
            z = redshift
        sorted_indices = np.argsort(z)
        scatter = axes[i_2][i_1].scatter(x_var[sorted_indices], predictions[sorted_indices], alpha=0.9,
                                         c=z[sorted_indices], cmap='inferno', s=30, vmin=4, vmax=[15, 6][obvs], zorder=3)

        y_range = ([-2.5, -0.5], [51, 53])[i_2]
        axes[i_2][i_1].set_ylabel(f_or_n_str)
        axes[i_2][i_1].set_xlabel(x_str)
        if obvs:
            axes[i_2][i_1].set_xlim(x_range)
            axes[i_2][i_1].set_ylim(y_range)

        # calculates the best fit parameters
        subset_indices = range(len(predictions))
        #subset_indices = np.where(log_obvs_star_mass > 9)[0]
        #subset_indices = np.where(obvs_redshift > 5)[0]
        #subset_indices = np.where(obvs_uv_mag < -19.5)[0]
        print(len(subset_indices))
        popt, pcov = curve_fit(linear_1var, x_var[subset_indices], predictions[subset_indices],
                            sigma=pred_mae[subset_indices], absolute_sigma=True)
        a, b = popt
        a_err, b_err = np.sqrt(np.diag(pcov))
        x_fit = np.linspace(*x_range, 100)
        y_fit = linear_1var(x_fit, a, b)
        fit = axes[i_2][i_1].plot(x_fit, y_fit, c='teal', alpha=0.6, zorder=4)

        # Add confidence bands (regions of possible fit based on parameter errors)
        n_std = 1.96  # 95% confidence interval (from normal distribution)
        # Generate points within the confidence band
        y_upper = np.zeros(len(x_fit))
        y_lower = np.zeros(len(x_fit))
        for i, x in enumerate(x_fit):
            # Calculate standard error at this x value
            # For a linear model: y = a*x + b
            # Step 1: Create the design matrix row for this x value
            # For linear model, this is [x, 1] where x is predictor and 1 is for intercept
            x_vec = np.array([x, 1])
            # Step 2: Calculate the variance of the prediction at this x value
            # This uses the formula: Var(y_pred) = X * Cov(params) * X^T
            # where X is the design matrix row and Cov(params) is the covariance matrix
            var_y_pred = x_vec @ pcov @ x_vec
            # Step 3: Calculate standard error (square root of variance)
            std_error = np.sqrt(var_y_pred)
            # Step 4: Calculate confidence interval bounds
            # CI = prediction ± (critical value * standard error)
            y_upper[i] = y_fit[i] + n_std * std_error
            y_lower[i] = y_fit[i] - n_std * std_error
        
        # Plot the confidence band as a shaded region
        axes[i_2][i_1].fill_between(x_fit, y_lower, y_upper, color='teal', alpha=0.2, zorder=1,
                            label='95% Confidence Band')
        
        a_str, a_err_str = round_to_error(a, a_err)
        b_str, b_err_str = round_to_error(b, b_err)
        fit_label = f'{f_or_n_str_no_unit} = ({a_str}$\pm${a_err_str}) {x_str_no_unit} + ({b_str}$\pm${b_err_str})'  
        axes[i_2][i_1].text(0.05, 0.05, fit_label, ha='left', va='bottom',
                            transform=axes[i_2][i_1].transAxes, fontsize=15)

        axes[i_2][i_1].set_box_aspect(1)
        axes[i_2][i_1].grid(False)

fig.tight_layout(w_pad=5)
cbar = fig.colorbar(scatter, ax=axes, orientation='vertical', aspect=30)
cbar.set_label("$z$")

mpl.rcParams['figure.dpi'] = 500
folder = "final_graph_generation/"
fig.savefig(folder + "report_graphs/report_graph.png", bbox_inches='tight', dpi=500)
plt.show()