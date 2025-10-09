import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import joblib
import json
from sklearn.metrics import mean_absolute_error
from functions import *
from scipy.optimize import curve_fit
from scipy.integrate import quad
import numpy as np
from astropy.cosmology import Planck15 as cosmo

# 0 for f_esc, 1 for n_esc
f_or_n = 1

# False for plotting total N_esc against redshift and True for plotting mass and magnitude bands
split_contribution = False

folder = "final_rf_model/"
file3 = folder + 'f_esc_rf_observational_test_train.json'
file4 = folder + 'n_esc_rf_observational_test_train.json' 

with open('lola_cat.txt', 'r') as f:
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

    obvs_vars = [obvs_f_esc_vars, obvs_n_esc_vars][f_or_n]
    obvs_log_vars = [obvs_log_f_esc_vars, obvs_log_n_esc_vars][f_or_n]
    obvs_keys = [obvs_f_esc_keys, obvs_n_esc_keys][f_or_n]


X = np.transpose(obvs_log_vars)
# run target prediction using loaded model
f_or_n_str = ['$\mathrm{Log}_{10}(f_\mathrm{esc})$', '$\mathrm{Log}_{10}(\dot{n}_\mathrm{ion,esc})$'][f_or_n]
loaded_model = joblib.load(folder + ['f_esc', 'n_esc'][f_or_n] + '_rf_observational_model.pkl')
predictions = loaded_model.predict(X)
with open((file3, file4)[f_or_n], 'r') as json_data:
    data = json.load(json_data)
    test = np.array(data['f_esc_test'])
    test_pred = np.array(data['f_esc_test_pred'])

    # seperates the test sample into bins of predicted test target with each containing equal numbers of galaxies, 
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

    # finds which mae bin a prediction of the target falls into so its error can be classified as that mae
    pred_mae = np.zeros(len(predictions))
    for i in range(len(predictions)):
        index = np.argmin(np.abs(test_pred_medians - predictions[i]))
        pred_mae[i] = (test_pred_mae[index])


def linear_2var(X, a, b, c):
    y, z = X
    return a * y + b * z + c

def linear_1var(X, a, b):
    y = X
    return a * y + b


# form for the number density of galaxies as a function of stellar mass
m_alpha_fit = {4: -1.79, 5: -1.86, 6: -1.95, 7: -1.93, 8: -2.16, 9: -2.0}
m_alpha_fit_err = {4: 0.01, 5: 0.03, 6: 0.07, 7: 0.04, 8: 0.19, 9: 0}
log_m_phi_fit = {4: -4.52,5 : -4.07, 6: -4.26, 7: -4.36, 8: -4.86, 9: -4.93}
log_m_phi_fit_err = {4: 0.13, 5: 0.13, 6: 0.36, 7: 0.05, 8: 0.20, 9: 0.07}
log_m_mass_fit = {4: 11.01, 5: 10.26, 6: 10.01, 7: 10.0, 8: 10.0, 9: 10.0}
log_m_mass_fit_err = {4: 0.14, 5: 0.12, 6: 0.32, 7: 0, 8: 0, 9: 0}
def schechter_mass(mass, m_alpha_fit, log_m_phi_fit, log_m_mass_fit):
    m_phi_fit = 10**log_m_phi_fit
    m_mass_fit = 10**log_m_mass_fit
    return m_phi_fit * (mass / m_mass_fit)**m_alpha_fit * np.exp(-mass / m_mass_fit) / m_mass_fit

# integrand to integrate stellar mass over to get N_esc
def integrand_mass(mass, z, a, b, c, m_alpha_fit, log_m_phi_fit, log_m_mass_fit):
    number_density_func = schechter_mass(mass, m_alpha_fit, log_m_phi_fit, log_m_mass_fit)
    linear_func = 10 ** linear_2var((np.log10(float(mass)), z), a, b, c)
    return number_density_func * linear_func

# calculates the best fit parameters for stellar mass
subset_indices = range(len(predictions))
popt, pcov = curve_fit(linear_2var, (log_obvs_star_mass[subset_indices], obvs_redshift[subset_indices]),
                       predictions[subset_indices], sigma=pred_mae[subset_indices], absolute_sigma=True)
a, b, c = popt
a_err, b_err, c_err = np.sqrt(np.diag(pcov))
print([a, b, c])
print([a_err, b_err, c_err])

# integrates over the number density of galaxies multiplied by n_esc as a function of stellar mass to calculate N_esc
m_redshifts = np.array([4, 5, 6, 7, 8, 9])
if not split_contribution:
    m_N_escs = np.zeros(len(m_redshifts))
    m_N_escs_high = np.zeros(len(m_redshifts))
    m_N_escs_low = np.zeros(len(m_redshifts))
    for z_i in range(len(m_redshifts)):
        z = m_redshifts[z_i]
        args = (z, a, b, c, m_alpha_fit[z], log_m_phi_fit[z], log_m_mass_fit[z])
        result, _ = quad(integrand_mass, 10**7.5, 10**12.5, args=args)
        # integrations for plus/minus parameter error to find integration errors
        args_plus = (z, a + a_err, b, c, m_alpha_fit[z], log_m_phi_fit[z], log_m_mass_fit[z])
        result_plus, _ = quad(integrand_mass, 10**7.5, 10**12.5, args=args_plus)
        args_minus = (z, a - a_err, b, c, m_alpha_fit[z], log_m_phi_fit[z], log_m_mass_fit[z])
        result_minus, _ = quad(integrand_mass, 10**7.5, 10**12.5, args=args_minus)
        m_N_escs[z_i] = result
        m_N_escs_high[z_i] = result_plus
        m_N_escs_low[z_i] = result_minus
        print(f"z = {m_redshifts[z_i]}: N_esc = {result:.3e}, Upper = {result_plus:.3e}, Lower = {result_minus:.3e}")
    log_m_N_escs = np.log10(np.array(m_N_escs))
    log_m_err_low = log_m_N_escs - np.log10(np.array(m_N_escs_low))
    log_m_err_high = np.log10(np.array(m_N_escs_high)) - log_m_N_escs

    # # Monte Carlo error propagation for N_esc
    # std_m_N_escs = np.zeros(len(m_redshifts))
    # for z_i in range(len(m_redshifts)):
    #     z = m_redshifts[z_i]
    #     N = 1000
    #     error_results = np.zeros(N)
    #     a_samples = np.random.normal(a, a_err, N)
    #     b_samples = np.random.normal(b, b_err, N)
    #     c_samples = np.random.normal(c, c_err, N)
    #     alpha_samples = np.random.normal(m_alpha_fit[z], m_alpha_fit_err[z], N)
    #     log_phi_samples = np.random.normal(log_m_phi_fit[z], log_m_phi_fit_err[z], N)
    #     log_mass_samples = np.random.normal(log_m_mass_fit[z], log_m_mass_fit_err[z], N)
    #     for i in range(N):
    #         # only include errors on log_star_mass gradient fit parameter
    #         args = (z, a_samples[i], b, c, m_alpha_fit[z], log_m_phi_fit[z], log_m_mass_fit[z])
    #         result, _ = quad(integrand_mass, 10**7.5, 10**12.5, args=args)
    #         error_results[i] = result
    #     std_m_N_escs[z_i] = np.std(error_results)
    #     print(f"z = {m_redshifts[z_i]}: N_esc = {m_N_escs[z_i]:.3e} ± {std_m_N_escs[z_i]:.3e}")

# integrates over the number density of galaxies multiplied by n_esc for different mass bands
else:
    bands = [(10**7.5, 10**8.5), (10**8.5, 10**9.5), (10**9.5, 10**12.5)]
    split_m_N_escs = []
    for band in bands:
        m_N_escs = []
        for z in m_redshifts:
            args = (z, a, b, c, m_alpha_fit[z], log_m_phi_fit[z], log_m_mass_fit[z])
            result, _ = quad(integrand_mass, band[0], band[1], args=args)
            m_N_escs.append(result)
        m_N_escs = np.array(m_N_escs)
        split_m_N_escs.append(m_N_escs)


# form for the number density of galaxies as a function of UV magnitude
uv_alpha_fit = {2.1: -1.52, 2.9: -1.61, 3.8: -1.69, 4.9: -1.74, 5.9: -1.93, 6.8: -2.06,
                7.9: -2.23, 8.9: -2.33, 10.2: -2.38}
uv_alpha_fit_err = {2.1: 0.03, 2.9: 0.03, 3.8: 0.03, 4.9: 0.06, 5.9: 0.08, 6.8: 0.11,
                    7.9: 0.20, 8.9: 0.19, 10.2: 0.28}
uv_phi_fit = {2.1: 4.0, 2.9: 2.1, 3.8: 1.69, 4.9: 0.79, 5.9: 0.51, 6.8: 0.19,
              7.9: 0.09, 8.9: 0.021, 10.2: 0.0042}
uv_phi_fit_err = {2.1: 0.4, 2.9: 0.3, 3.8: 0.21, 4.9: 0.14, 5.9: 0.11, 6.8: 0.07,
                  7.9: 0.07, 8.9: 0.011, 10.2: 0.0033}
uv_mag_fit = {2.1: -20.28, 2.9: -20.87, 3.8: -20.93, 4.9: -21.10, 5.9: -20.93, 6.8: -21.15,
              7.9: -20.93, 8.9: -21.15, 10.2: -21.19}
uv_mag_fit_err = {2.1: 0.09, 2.9: 0.09, 3.8: 0.08, 4.9: 0.11, 5.9: 0.09, 6.8: 0.13,
                  7.9: 0.28, 8.9: 0, 10.2: 0}
def schechter_uv(mag, uv_alpha_fit, uv_phi_fit, uv_mag_fit):
    uv_phi_fit = uv_phi_fit * 1e-3
    k = 10 ** (0.4 * (uv_mag_fit - mag))
    return 0.4 * np.log(10) * uv_phi_fit * k**(uv_alpha_fit+1) * np.exp(-k)

# integrand to integrate UV magnitude over to get N_esc
def integrand_uv(mag, z, a, b, c, uv_alpha_fit, uv_phi_fit, uv_mag_fit):
    number_density_func = schechter_uv(mag, uv_alpha_fit, uv_phi_fit, uv_mag_fit)
    linear_func = 10 ** linear_2var((mag, z), a, b, c)
    return number_density_func * linear_func

# calculates the best fit parameters for uv magnitude
subset_indices = range(len(predictions))
popt, pcov = curve_fit(linear_2var, (obvs_uv_mag[subset_indices], obvs_redshift[subset_indices]),
                       predictions[subset_indices], sigma=pred_mae[subset_indices], absolute_sigma=True)
a, b, c = popt
a_err, b_err, c_err = np.sqrt(np.diag(pcov))
print([a, b, c])
print([a_err, b_err, c_err])

# integrates over the number density of galaxies multiplied by n_esc as a function of stellar mass to calculate N_esc
uv_redshifts = np.array([2.1, 2.9, 3.8, 4.9, 5.9, 6.8, 7.9, 8.9, 10.2])
if not split_contribution:
    uv_N_escs = np.zeros(len(uv_redshifts))
    uv_N_escs_low = np.zeros(len(uv_redshifts))
    uv_N_escs_high = np.zeros(len(uv_redshifts))
    for z_i in range(len(uv_redshifts)):
        z = uv_redshifts[z_i]
        args = (z, a, b, c, uv_alpha_fit[z], uv_phi_fit[z], uv_mag_fit[z])
        result, _ = quad(integrand_uv, -28, -16, args=args)
        # integrations for plus/minus parameter error to find integration errors
        result_plus_args = (z, a + a_err, b, c, uv_alpha_fit[z], uv_phi_fit[z], uv_mag_fit[z])
        result_plus, _ = quad(integrand_uv, -28, -16, args=result_plus_args)
        result_minus_args = (z, a - a_err, b, c, uv_alpha_fit[z], uv_phi_fit[z], uv_mag_fit[z])
        result_minus, _ = quad(integrand_uv, -28, -16, args=result_minus_args)
        uv_N_escs[z_i] = result
        uv_N_escs_low[z_i] = result_plus
        uv_N_escs_high[z_i] = result_minus
        print(f"z = {uv_redshifts[z_i]}: N_esc = {result:.3e}, Upper = {result_minus:.3e}, Lower = {result_plus:.3e}")
    log_uv_N_escs = np.log10(np.array(uv_N_escs))
    log_uv_err_low = log_uv_N_escs - np.log10(np.array(uv_N_escs_low))
    log_uv_err_high = np.log10(np.array(uv_N_escs_high)) - log_uv_N_escs

    # # Monte Carlo error propagation for N_esc
    # std_uv_N_escs = np.zeros(len(uv_redshifts))
    # for z_i in range(len(uv_redshifts)):
    #     z = uv_redshifts[z_i]
    #     N = 10000
    #     error_results = np.zeros(N)
    #     a_samples = np.random.normal(a, a_err, N)
    #     b_samples = np.random.normal(b, b_err, N)
    #     c_samples = np.random.normal(c, c_err, N)
    #     alpha_samples = np.random.normal(uv_alpha_fit[z], uv_alpha_fit_err[z], N)
    #     phi_samples = np.random.normal(uv_phi_fit[z], uv_phi_fit_err[z], N)
    #     mag_samples = np.random.normal(uv_mag_fit[z], uv_mag_fit_err[z], N)
    #     for i in range(N):
    #         # only include errors on magnitude gradient fit parameter
    #         args = (z, a_samples[i], b, c, uv_alpha_fit[z], uv_phi_fit[z], uv_mag_fit[z])
    #         result, _ = quad(integrand_uv, -24, -16, args=args)
    #         error_results[i] = result
    #     std_uv_N_escs[z_i] = np.std(error_results)
    #     print(f"z = {uv_redshifts[z_i]}: N_esc = {uv_N_escs[z_i]:.3e} ± {std_uv_N_escs[z_i]:.3e}")

# integrates over the number density of galaxies multiplied by n_esc for different magnitude bands
else:
    bands = [(-18, -16), (-20, -18), (-28, -20)]
    split_uv_N_escs = []
    for band in bands:
        uv_N_escs = []
        for z in uv_redshifts:
            args = (z, a, b, c, uv_alpha_fit[z], uv_phi_fit[z], uv_mag_fit[z])
            result, _ = quad(integrand_uv, band[0], band[1], args=args)
            uv_N_escs.append(result)
        uv_N_escs = np.array(uv_N_escs)
        split_uv_N_escs.append(uv_N_escs)


if not split_contribution:
    def critical(z, C):
        omega_b = 0.0486
        h_50 = 100 * 0.6774 / 50
        C_30 = C / 30
        return 10**(51.2) * C_30 * ((1 + z) / 6)**3 * ((omega_b * h_50**2) / 0.08)**2

    def log_errors(errors, results):
        return errors / (results * np.log(10))

    plt.style.use('./MNRAS_Style.mplstyle')
    mpl.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(figsize=(16, 8))

    z_space = np.linspace(1, 11, 100)
    for C in [1, 3, 10]:
        ax.plot(z_space, np.log10(critical(z_space, C)), c='grey', zorder=1)
        text_z = 8.25
        ax.text(text_z, np.log10(critical(text_z, C))-0.08, f'$C = {C}$',
                rotation=15, color='grey')
    ax.fill_between(z_space, np.log10(critical(z_space, 1)), np.log10(critical(z_space, 10)),
                    color='grey', alpha=0.2, zorder=1, label='$\dot{N}_\mathrm{ion}$ Critical')
    
    #log_uv_errors  = log_errors(std_uv_N_escs, uv_N_escs)
    ax.errorbar(uv_redshifts, log_uv_N_escs, yerr=(log_uv_err_low, log_uv_err_high),
                fmt='none', c='royalblue', elinewidth=2, capsize=5, zorder=3)
    ax.plot(uv_redshifts, log_uv_N_escs, linestyle='--', c='royalblue', linewidth=3, zorder=2)
    ax.scatter(uv_redshifts, log_uv_N_escs, s=100, c='royalblue', edgecolors='black', zorder=4,
            label='$\dot{N}_\mathrm{ion}$ UV Magnitude Integration')

    #log_m_errors = log_errors(std_m_N_escs, m_N_escs)
    ax.errorbar(m_redshifts, log_m_N_escs, yerr=(log_m_err_low, log_m_err_high),
                fmt='none', c='darkorange', elinewidth=2, capsize=5, zorder=3)
    ax.plot(m_redshifts, log_m_N_escs, linestyle='--', c='darkorange', linewidth=3, zorder=2)
    ax.scatter(m_redshifts, log_m_N_escs, s=100, c='darkorange',  edgecolors='black', zorder=4,
            label='$\dot{N}_\mathrm{ion}$ Stellar Mass Integration')

    ax.set_xlabel("$z$")
    ax.set_ylabel("$\mathrm{log}_{10}(\dot{N}_\mathrm{ion} \; [\mathrm{s^{-1} \; cMpc^{-3}}])$")
    ax.yaxis.set_label_coords(-0.075, 0.5)
    ax.set_xlim((1, 11))
    ax.set_ylim((49, 51.5))
    ax.grid(True, alpha=0.8, linestyle='--')
    ax.set_axisbelow(True)
    for line in ax.get_xgridlines() + ax.get_ygridlines():
        line.set_zorder(0)
    legend = ax.legend(fontsize=18, loc='lower center', bbox_to_anchor=(0.5, 0.025))
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_boxstyle('Square')
    legend.get_frame().set_alpha(1.0)

else:

    plt.style.use('./MNRAS_Style.mplstyle')
    mpl.rcParams.update({'font.size': 20})
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    bar_width = 0.25
    offsets = [-bar_width, 0, bar_width]
    uv_labels = ('$-18 < M_\mathrm{UV} \leq -16$',
                 '$-20 < M_\mathrm{UV} \leq -18$',
                 '$M_\mathrm{UV} \leq -20$')
    m_labels = ('$7.5 \leq \mathrm{log}_{10}(M_*) < 8.5$',
                '$8.5 \leq \mathrm{log}_{10}(M_*) < 9.5$',
                '$\mathrm{log}_{10}(M_*) \geq 9.5$')
    uv_colors = ['#66c2a5', '#3288bd', '#5e4fa2']  # Teal, Medium Blue, Deep Blue
    m_colors = ['#fdb863', '#e66101', '#b2182b']  # Gold, Orange, Crimson

    for ax_i in range(len(axes)):
        split_N_escs = [[arr[2:8] for arr in split_uv_N_escs], split_m_N_escs][ax_i]
        redshifts = np.array([4, 5, 6, 7, 8, 9])
        labels = [uv_labels, m_labels][ax_i]
        colors = [uv_colors, m_colors][ax_i]
        for i in range(3):
            axes[ax_i].bar(redshifts + offsets[i], split_N_escs[i]/10**50,
                           width=bar_width, label=labels[i], color=colors[i], edgecolor='black')

        axes[ax_i].set_xticks(redshifts)
        axes[ax_i].set_xlabel('$z$')
        axes[ax_i].tick_params(axis='x', which='both', bottom=False, top=False)
        axes[ax_i].set_ylabel('$\dot{N}_{\mathrm{ion}} \; [10^{50} \; \mathrm{s^{-1} \; cpc^{-3}}]$')
        axes[ax_i].yaxis.set_label_coords(-0.10, 0.5)
        axes[ax_i].set_xlim(3.25, 9.75)
        axes[ax_i].set_ylim(0, 3.5)
        axes[ax_i].grid(False)
        axes[ax_i].grid(True,  alpha=0.8, axis='y')
        axes[ax_i].set_axisbelow(True)
        for line in axes[ax_i].get_xgridlines() + axes[ax_i].get_ygridlines():
            line.set_zorder(0)
        legend = axes[ax_i].legend(loc='upper right', bbox_to_anchor=(0.975, 0.975), fontsize=18)
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_boxstyle('Square')
        legend.get_frame().set_alpha(1.0)


mpl.rcParams['figure.dpi'] = 500
folder = "final_graph_generation/"
fig.savefig(folder + "report_graphs/report_graph.png", bbox_inches='tight', dpi=500)
plt.show()