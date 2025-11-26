import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import joblib
import json
from sklearn.metrics import mean_absolute_error
from functions import *
from scipy.optimize import curve_fit
from scipy.integrate import quad, fixed_quad
import numpy as np
from astropy.io import fits
from scipy import odr
import csv
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.signal import fftconvolve
import pickle

# 0 for f_esc, 1 for n_esc
f_or_n = 1

# False for plotting total N_esc against redshift and True for plotting mass and magnitude bands
split_contribution = False

# Whether to use the random forest models trained with the dusty or dust-free Thesan-Zoom data
dusty = True

folder = "final_rf_model/"
if not dusty:
    file5 = folder + 'f_esc_rf_observational_charlotte_test_train.json'
    file6 = folder + 'n_esc_rf_observational_charlotte_test_train.json'
else:
    file5 = folder + 'f_esc_rf_observational_charlotte_dusty_test_train.json'
    file6 = folder + 'n_esc_rf_observational_charlotte_dusty_test_train.json'
# file5 = folder + 'f_esc_rf_observational_charlotte_test_train_test.json'
# file6 = folder + 'n_esc_rf_observational_charlotte_test_train_test.json'

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
b_i += [index for index, val in enumerate(list(obvs_data['red_chi2(JWST)'])) if val > 1]
print(f"red_chi2(JWST) > 1 rows: {len(b_i)}")
bad_indices += b_i
for i in range(len(obvs_f_esc_vars)):
    b_i = [index for index, val in enumerate(list((obvs_f_esc_vars)[i]))
                    if (val == 0 or val == np.inf or val== -np.inf or np.isnan(val))]
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
obvs_uv_mag_high_error = np.delete(np.array(obvs_data['M1500obs_ehi']), bad_indices)
obvs_uv_mag_low_error = np.delete(np.array(obvs_data['M1500obs_elo']), bad_indices)
obvs_redshift_high_error = np.delete(np.array(obvs_data['z_ehi']), bad_indices)
obvs_redshift_low_error = np.delete(np.array(obvs_data['z_elo']), bad_indices)
print(f'rows remaining: {len(obvs_f_esc_vars[i])}')

vars = [obvs_f_esc_vars, obvs_n_esc_vars][f_or_n]
log_vars = [obvs_log_f_esc_vars, obvs_log_n_esc_vars][f_or_n]
keys = [obvs_f_esc_keys, obvs_n_esc_keys][f_or_n]
log_star_mass = log_vars[(3, 2)[f_or_n]]
log_uv_mag = log_vars[(4, 3)[f_or_n]]
print(keys)

X = np.transpose(log_vars)
# run target prediction using loaded model
f_or_n_str = ['$\mathrm{Log}_{10}(f_\mathrm{esc})$', '$\mathrm{Log}_{10}(\dot{n}_\mathrm{ion,esc})$'][f_or_n]
if not dusty:
    loaded_model = joblib.load(folder + ['f_esc', 'n_esc'][f_or_n] + '_rf_observational_charlotte_model.pkl')
else:
    loaded_model = joblib.load(folder + ['f_esc', 'n_esc'][f_or_n] + '_rf_observational_charlotte_dusty_model.pkl')
# loaded_model = joblib.load(folder + ['f_esc', 'n_esc'][f_or_n] + '_rf_observational_charlotte_model_test.pkl')
predictions = loaded_model.predict(X)
with open((file5, file6)[f_or_n], 'r') as json_data:
    data = json.load(json_data)
    test = np.array(data['f_esc_test'])
    test_pred = np.array(data['f_esc_test_pred'])

    # seperates the test sample into bins of predicted test target with each containing equal numbers of galaxies, 
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

    # finds which mae bin a prediction of the target falls into so its error can be classified as that mae
    pred_mae = np.zeros(len(predictions))
    for i in range(len(predictions)):
        index = np.argmin(np.abs(test_pred_medians - predictions[i]))
        pred_mae[i] = (test_pred_mae[index])


def linear_2var(p, X):
    a, b, c = p
    x, z = X
    return a * x + b * z + c

def linear_1var(p, X):
    a, b = p
    return a * X + b

def n_ion_esc_linear_2var(z, a, b, c):
    def linear_func(mag):
        return 10 ** linear_2var((a, b, c), (mag, z))
    return linear_func

def n_ion_esc_linear_1var(a, b):
    def linear_func(mag):
        return 10 ** linear_1var((a, b), mag)
    return linear_func


# Schechter parameters for the number density of galaxies as a function of UV magnitude from Bouwens et al. (2021)
sch_uv_redshifts = np.array([2.1, 2.9, 3.8, 4.9, 5.9, 6.8, 7.9, 8.9, 10.2])
sch_uv_alpha_fit = {2.1: -1.52, 2.9: -1.61, 3.8: -1.69, 4.9: -1.74, 5.9: -1.93, 6.8: -2.06,
                7.9: -2.23, 8.9: -2.33, 10.2: -2.38}
sch_uv_alpha_fit_err = {2.1: 0.03, 2.9: 0.03, 3.8: 0.03, 4.9: 0.06, 5.9: 0.08, 6.8: 0.11,
                    7.9: 0.20, 8.9: 0.19, 10.2: 0.28}
sch_uv_phi_fit = {2.1: 4.0, 2.9: 2.1, 3.8: 1.69, 4.9: 0.79, 5.9: 0.51, 6.8: 0.19,
              7.9: 0.09, 8.9: 0.021, 10.2: 0.0042}
sch_uv_phi_fit_err = {2.1: 0.4, 2.9: 0.3, 3.8: 0.21, 4.9: 0.14, 5.9: 0.11, 6.8: 0.07,
                  7.9: 0.07, 8.9: 0.011, 10.2: 0.0033}
sch_uv_mag_fit = {2.1: -20.28, 2.9: -20.87, 3.8: -20.93, 4.9: -21.10, 5.9: -20.93, 6.8: -21.15,
              7.9: -20.93, 8.9: -21.15, 10.2: -21.19}
sch_uv_mag_fit_err = {2.1: 0.09, 2.9: 0.09, 3.8: 0.08, 4.9: 0.11, 5.9: 0.09, 6.8: 0.13,
                  7.9: 0.28, 8.9: 0.28, 10.2: 0.28}

# extra Schechter parameters at z=9.8 and z=12.8 from Whitler et al. (2025)
jwst_sch_funcs = True
if jwst_sch_funcs:
    sch_uv_redshifts = sch_uv_redshifts[[0, 1, 2, 3, 4, 5, 6]]
    sch_uv_redshifts = np.append(sch_uv_redshifts, [9.8, 12.8])
    sch_uv_alpha_fit |= {9.8: -2.36, 12.8: -2.23}
    sch_uv_alpha_fit_err |= {9.8: 0.19, 12.8: 0.31}
    sch_uv_phi_fit |= {9.8: 0.10070, 12.8: 0.04083}
    sch_uv_phi_fit_err |= {9.8: 0.08210, 12.8: 0.02563}
    sch_uv_mag_fit |= {9.8: -20.32, 12.8: -20.54}
    sch_uv_mag_fit_err |= {9.8: 0.48, 12.8: 0.48}

# Double Power Law parameters for the number density of galaxies as a function of UV magnitude from Bowler et al. (2020)
dpl_uv_redshifts = np.array([4, 5, 6, 7, 8, 9])
dpl_uv_phi_fit = {4: 0.55, 5: 0.323, 6: 0.193, 7: 0.38, 8: 0.48, 9: 0.29}
dpl_uv_phi_fit_err = {4: 0.09, 5: 0.046, 6: 0.053, 7: 0.17, 8: 0.23, 9: 0.14}
dpl_uv_mag_fit = {4: -21.34, 5: -21.48, 6: -21.24, 7: -20.27, 8: -19.80, 9: -19.67}
dpl_uv_mag_fit_err = {4: 0.10, 5: 0.06, 6: 0.10, 7: 0.19, 8: 0.26, 9: 0.33}
dpl_uv_alpha_fit = {4: -1.89, 5: -1.88, 6: -2.07, 7: -2.12, 8: -1.96, 9: -2.10}
dpl_uv_alpha_fit_err = {4: 0.06, 5: 0.04, 6: 0.07, 7: 0.08, 8: 0.15, 9: 0.15}
dpl_uv_beta_fit = {4: -4.77, 5: -5.29, 6: -5.23, 7: -4.23, 8: -3.98, 9: -3.75}
dpl_uv_beta_fit_err = {4: 0.20, 5: 0.19, 6: 0.19, 7: 0.17, 8: 0.14, 9: 0.22}

# extra DPL parameters from Donnan et al. (2024) or Whitler et al. (2025)
jwst_dpl_funcs = True
if jwst_dpl_funcs:
    # dpl_uv_redshifts = dpl_uv_redshifts[[0, 1, 2, 3, 4]]
    # dpl_uv_redshifts = np.append(dpl_uv_redshifts, [9, 10, 11, 12.5, 14.5])
    # dpl_uv_phi_fit |= {9: 0.23, 10: 0.14, 11: 0.033, 12.5: 0.0099, 14.5: 0.0018}
    # dpl_uv_phi_fit_err |= {9: 0.39, 10: 0.16, 11: 0.099, 12.5: 0.0099, 14.5: 0.0028}
    # dpl_uv_mag_fit |= {9: -19.70, 10: -19.98, 11: -20.73, 12.5: -20.82, 14.5: -20.82}
    # dpl_uv_mag_fit_err |= {9: 0.96, 10: 0.61, 11: 1.61, 12.5: 0.71, 14.5: 0.71}
    # dpl_uv_alpha_fit |= {9: -2.00, 10: -1.98, 11: -2.19, 12.5: -2.19 , 14.5: -2.19}
    # dpl_uv_alpha_fit_err |= {9: 0.47, 10: 0.40, 11: 0.69, 12.5: 0.69, 14.5: 0.69}
    # dpl_uv_beta_fit |= {9: -3.81, 10: -4.05, 11: -4.29, 12.5: -4.29, 14.5: -4.29}
    # dpl_uv_beta_fit_err |= {9: 0.49, 10: 1.30, 11: 1.30, 12.5: 1.30, 14.5: 1.30}
    dpl_uv_redshifts = dpl_uv_redshifts[[0, 1, 2, 3]]
    dpl_uv_redshifts = np.append(dpl_uv_redshifts, [9.8, 12.8])
    dpl_uv_phi_fit |= {9.8: 0.04849, 12.8: 0.02263}
    dpl_uv_phi_fit_err |= {9.8: 0.02147, 12.8: 0.01485}
    dpl_uv_mag_fit |= {9.8: -20.54, 12.8: -20.54}
    dpl_uv_mag_fit_err |= {9.8: 0.35, 12.8: 0.35}
    dpl_uv_alpha_fit |= {9.8: -2.60, 12.8: -2.42}
    dpl_uv_alpha_fit_err |= {9.8: 0.18, 12.8: 0.32}
    dpl_uv_beta_fit |= {9.8: -3.49, 12.8: -3.50}
    dpl_uv_beta_fit_err |= {9.8: 1.07, 12.8: 1.28}


def schechter_uv(uv_alpha_fit, uv_phi_fit, uv_mag_fit):
    uv_phi_fit = uv_phi_fit * 1e-3
    def uv_lum_func(mag):
        k = 10 ** (0.4 * (uv_mag_fit - mag))
        return 0.4 * np.log(10) * uv_phi_fit * k**(uv_alpha_fit + 1) * np.exp(-k)
    return uv_lum_func

def schechter_uv_convolved(mag_min, mag_max, sigma_mag, uv_alpha_fit, uv_phi_fit, uv_mag_fit, n_bins=1001):
    # integral over a uniform grid in magnitude space
    mags = np.linspace(mag_min, mag_max, n_bins)
    dM = mags[1] - mags[0]
    # original UV luminosity function
    phi = schechter_uv(uv_alpha_fit, uv_phi_fit, uv_mag_fit)(mags)
    # Gaussian kernel
    kernel_extent = 6 * sigma_mag
    n_kernel = int(np.ceil((2 * kernel_extent) / dM)) + 1
    if n_kernel % 2 == 0:
        n_kernel += 1
    delta_mags = np.linspace(-kernel_extent, kernel_extent, n_kernel)
    gaussian = np.exp(-0.5 * (delta_mags / sigma_mag)**2)
    gaussian /= np.sum(gaussian) * dM
    # convolution
    phi_conv = fftconvolve(phi, gaussian, mode='same') * dM
    # return interpolated convoluted UV luminosity function
    return interp1d(mags, phi_conv, bounds_error=False, fill_value=0, assume_sorted=True)

def dpl_uv(uv_alpha_fit, uv_beta_fit, uv_phi_fit, uv_mag_fit):
    uv_phi_fit = uv_phi_fit * 1e-3
    def uv_lum_func(mag):
        k = 10 ** (-0.4 * (uv_mag_fit - mag))
        return 0.4 * np.log(10) * uv_phi_fit / (k ** (uv_alpha_fit + 1) + k ** (uv_beta_fit + 1))
    return uv_lum_func

def dpl_uv_convolved(mag_min, mag_max, sigma_mag, uv_alpha_fit, uv_beta_fit, uv_phi_fit, uv_mag_fit, n_bins=1000):
    # integral over a uniform grid in magnitude space
    mags = np.linspace(mag_min, mag_max, n_bins)
    dM = mags[1] - mags[0]
    # original UV luminosity function
    phi = dpl_uv(uv_alpha_fit, uv_beta_fit, uv_phi_fit, uv_mag_fit)(mags)
    # Gaussian kernel
    kernel_extent = 6 * sigma_mag
    n_kernel = int(np.ceil((2 * kernel_extent) / dM)) + 1
    if n_kernel % 2 == 0:
        n_kernel += 1
    delta_mags = np.linspace(-kernel_extent, kernel_extent, n_kernel)
    gaussian = np.exp(-0.5 * (delta_mags / sigma_mag)**2)
    gaussian /= np.sum(gaussian) * dM
    # convolution
    phi_conv = fftconvolve(phi, gaussian, mode='same') * dM
    # return interpolated convoluted UV luminosity function
    return interp1d(mags, phi_conv, bounds_error=False, fill_value=0, assume_sorted=True)

# calculates the best fit parameters for UV magnitude using ODR to account for x and y errors
subset_indices = range(len(predictions))
subset_indices = np.where(log_uv_mag < -0)[0]
x_err_sym = (obvs_uv_mag_low_error + obvs_uv_mag_high_error) / 2
x_err_median = np.nanmedian(x_err_sym[x_err_sym > 0]) * 1e-6
x_err_sym[x_err_sym <= 0] = x_err_median
z_err_sym = (obvs_redshift_low_error + obvs_redshift_high_error) / 2
z_err_median = np.nanmedian(z_err_sym[z_err_sym > 0]) * 1e-6
z_err_sym[z_err_sym <= 0] = z_err_median
model = odr.Model(linear_2var)
data = odr.RealData((log_uv_mag[subset_indices], obvs_redshift[subset_indices]), predictions[subset_indices],
                    sx=(x_err_sym[subset_indices], z_err_sym[subset_indices]), sy=pred_mae[subset_indices])
odr_inst = odr.ODR(data, model, beta0=[0, 0, 0])
out = odr_inst.run()
a, b, c = out.beta
a_err, b_err, c_err = out.sd_beta
popt, pcov = out.beta, out.cov_beta

# # calculates the best fit parameters for UV magnitude using WLSR with only y errors
# def curve_fit_func(X, a, b, c):
#     return linear_2var((a, b, c), X)
# popt, pcov = curve_fit(curve_fit_func, (log_uv_mag[subset_indices], obvs_redshift[subset_indices]), predictions[subset_indices], 
#                        sigma=pred_mae[subset_indices], absolute_sigma=True)
# a, b, c = popt
# a_err, b_err, c_err = np.sqrt(np.diag(pcov))

a_str, a_err_str = round_to_error(a, a_err)
b_str, b_err_str = round_to_error(b, b_err)
c_str, c_err_str = round_to_error(c, c_err)
print('')
print(
    f'Log10({["f_esc", "n_ion,esc"][f_or_n]}) = '
    f'({a_str}±{a_err_str})Muv + '
    f'({b_str}±{b_err_str})z + '
    f'({c_str}±{c_err_str})'
)

sigma_mag = [0.671, 0.709][dusty]
n_bins = 2000
mag_min, mag_max = -30, -10

# integrates over the number density of galaxies multiplied by n_esc as a function of UV luminosity to calculate N_esc
if not split_contribution:
    sch_uv_N_escs, sch_uv_N_escs_low, sch_uv_N_escs_high = [np.zeros(len(sch_uv_redshifts)) for _ in range(3)]
    dpl_uv_N_escs, dpl_uv_N_escs_low, dpl_uv_N_escs_high = [np.zeros(len(dpl_uv_redshifts)) for _ in range(3)]

    # Monte-Carlo error propagation for UV N_esc integrations including a b c covariance and UV parameter errors
    n_mc = 1000
    rng = np.random.default_rng()
    # defensive cov handling for a,b,c (should be positive definite and symmetric)
    beta_mean = np.asarray(popt, dtype=float)
    # beta_mean = np.array([-0.631, 0.061, 40.07]) # direct thesan fit values for uv magnitude
    beta_cov = np.asarray(pcov, dtype=float)
    beta_cov = (beta_cov + beta_cov.T) / 2
    eig = np.linalg.eigvalsh(beta_cov)
    if np.any(eig < 0):
        beta_cov += np.eye(3) * (abs(np.min(eig)) + 1e-12)
    # draw samples of a,b,c using the covariance matrix
    betas = rng.multivariate_normal(beta_mean, beta_cov, size=n_mc)

    def draw(mean, err): 
        return rng.normal(mean, err, n_mc) if err > 0 else np.full(n_mc, mean)


    # MC analysis and integrations for Schechter UV functions
    for j, z in enumerate(sch_uv_redshifts):
        # sample Schechter parameters (fixed if err == 0)
        alpha_samples = draw(sch_uv_alpha_fit[z], sch_uv_alpha_fit_err[z])
        phi_samples = draw(sch_uv_phi_fit[z], sch_uv_phi_fit_err[z])
        mag_samples = draw(sch_uv_mag_fit[z], sch_uv_mag_fit_err[z])

        # integrate for each MC draw
        N_samples = np.empty(n_mc)
        for i in range(n_mc):
            a_s, b_s, c_s = betas[i]
            n_ion_esc_func = n_ion_esc_linear_2var(z, a_s, b_s, c_s)
            # uv_lum_function = schechter_uv(alpha_samples[i], phi_samples[i], mag_samples[i])
            uv_lum_function = schechter_uv_convolved(mag_min, mag_max, sigma_mag,
                                                     alpha_samples[i], phi_samples[i], mag_samples[i], n_bins=n_bins)
            integrand = lambda mag: n_ion_esc_func(mag) * uv_lum_function(mag)
            # N_samples[i], _ = quad(integrand, mag_min, mag_max)
            # N_samples[i], _ = fixed_quad(integrand, mag_min, mag_max, n=1000)
            mags = np.linspace(mag_min, mag_max, n_bins)
            N_samples[i] = np.maximum(np.trapz(integrand(mags), mags), 1e45)

        # compute median and 16/84 percentiles
        sch_uv_N_escs[j] = np.median(N_samples)
        sch_uv_N_escs_low[j] = np.percentile(N_samples, 16)
        sch_uv_N_escs_high[j] = np.percentile(N_samples, 84)
        print(f"z = {z}: N_esc = {sch_uv_N_escs[j]:.3e}, Upper = {sch_uv_N_escs_high[j]:.3e}, Lower = {sch_uv_N_escs_low[j]:.3e}")

    # convert to log and asymmetric dex errors (same form you used earlier)
    log_sch_uv_N_escs = np.log10(sch_uv_N_escs)
    log_sch_uv_err_low = log_sch_uv_N_escs - np.log10(sch_uv_N_escs_low)
    log_sch_uv_err_high = np.log10(sch_uv_N_escs_high) - log_sch_uv_N_escs
    sch_weights = 2 / (log_sch_uv_err_low + log_sch_uv_err_high)

    # interpolates N_ion as a function of redshift using a polynomial fit and a spline fit
    sch_poly_coeffs, sch_poly_cov = np.polyfit(1 + sch_uv_redshifts, log_sch_uv_N_escs, deg=3, w=sch_weights, cov=True)
    sch_poly_coeffs_errors = np.sqrt(np.diag(sch_poly_cov))
    sch_polynomial = np.poly1d(sch_poly_coeffs)
    sch_smooth = (0.2, 0.05)[jwst_sch_funcs]
    sch_spline = UnivariateSpline(sch_uv_redshifts, log_sch_uv_N_escs,
                              w=sch_weights, k=3, s=sch_smooth)
    sch_spline_high = UnivariateSpline(sch_uv_redshifts, log_sch_uv_N_escs + log_sch_uv_err_high,
                                   w=sch_weights, k=3, s=sch_smooth)
    sch_spline_low = UnivariateSpline(sch_uv_redshifts, log_sch_uv_N_escs - log_sch_uv_err_low,
                                  w=sch_weights, k=3, s=sch_smooth)
    with open("final_graph_generation/uv_N_ion_spline_sch.pkl", "wb") as f:
        pickle.dump(sch_spline, f)
    with open("final_graph_generation/uv_N_ion_spline_high_sch.pkl", "wb") as f:
        pickle.dump(sch_spline_high, f)
    with open("final_graph_generation/uv_N_ion_spline_low_sch.pkl", "wb") as f:
        pickle.dump(sch_spline_low, f)
    print("")

    
    # MC analysis and integrations for DPL UV functions
    for j, z in enumerate(dpl_uv_redshifts):
        # sample DPL parameters (fixed if err == 0)
        alpha_samples = draw(dpl_uv_alpha_fit[z], dpl_uv_alpha_fit_err[z])
        beta_samples = draw(dpl_uv_beta_fit[z], dpl_uv_beta_fit_err[z])
        phi_samples = draw(dpl_uv_phi_fit[z], dpl_uv_phi_fit_err[z])
        mag_samples = draw(dpl_uv_mag_fit[z], dpl_uv_mag_fit_err[z])

        # integrate for each MC draw
        N_samples = np.empty(n_mc)
        for i in range(n_mc):
            a_s, b_s, c_s = betas[i]
            n_ion_esc_func = n_ion_esc_linear_2var(z, a_s, b_s, c_s)
            # uv_lum_function = dpl_uv(alpha_samples[i], beta_samples[i], phi_samples[i], mag_samples[i])
            uv_lum_function = dpl_uv_convolved(mag_min, mag_max, sigma_mag, 
                                               alpha_samples[i], beta_samples[i], phi_samples[i], mag_samples[i], n_bins=n_bins)
            integrand = lambda mag: n_ion_esc_func(mag) * uv_lum_function(mag)
            # N_samples[i], _ = quad(integrand, mag_min, mag_max)
            # N_samples[i], _ = fixed_quad(integrand, mag_min, mag_max, n=1000)
            mags = np.linspace(mag_min, mag_max, n_bins)
            N_samples[i] = np.maximum(np.trapz(integrand(mags), mags), 1e45)

        # compute median and 16/84 percentiles
        dpl_uv_N_escs[j] = np.median(N_samples)
        dpl_uv_N_escs_low[j] = np.percentile(N_samples, 16)
        dpl_uv_N_escs_high[j] = np.percentile(N_samples, 84)
        print(f"z = {z}: N_esc = {dpl_uv_N_escs[j]:.3e}, Upper = {dpl_uv_N_escs_high[j]:.3e}, Lower = {dpl_uv_N_escs_low[j]:.3e}")

    # convert to log and asymmetric dex errors (same form you used earlier)
    log_dpl_uv_N_escs = np.log10(dpl_uv_N_escs)
    log_dpl_uv_err_low = log_dpl_uv_N_escs - np.log10(dpl_uv_N_escs_low)
    log_dpl_uv_err_high = np.log10(dpl_uv_N_escs_high) - log_dpl_uv_N_escs
    dpl_weights = 2 / (log_dpl_uv_err_low + log_dpl_uv_err_high)

    # interpolates N_ion as a function of redshift using a polynomial fit and a spline fit
    dpl_poly_coeffs, dpl_poly_cov = np.polyfit(1 + dpl_uv_redshifts, log_dpl_uv_N_escs, deg=3, w=dpl_weights, cov=True)
    dpl_poly_coeffs_errors = np.sqrt(np.diag(dpl_poly_cov))
    dpl_polynomial = np.poly1d(dpl_poly_coeffs)
    dpl_smooth = (0.2, 0.05)[jwst_sch_funcs]
    dpl_spline = UnivariateSpline(dpl_uv_redshifts, log_dpl_uv_N_escs,
                              w=dpl_weights, k=3, s=dpl_smooth)
    dpl_spline_high = UnivariateSpline(dpl_uv_redshifts, log_dpl_uv_N_escs + log_dpl_uv_err_high,
                                   w=dpl_weights, k=3, s=dpl_smooth)
    dpl_spline_low = UnivariateSpline(dpl_uv_redshifts, log_dpl_uv_N_escs - log_dpl_uv_err_low,
                                  w=dpl_weights, k=3, s=dpl_smooth)
    with open("final_graph_generation/uv_N_ion_spline_dpl.pkl", "wb") as f:
        pickle.dump(dpl_spline, f)
    with open("final_graph_generation/uv_N_ion_spline_high_dpl.pkl", "wb") as f:
        pickle.dump(dpl_spline_high, f)
    with open("final_graph_generation/uv_N_ion_spline_low_dpl.pkl", "wb") as f:
        pickle.dump(dpl_spline_low, f)


    # interpolates a combination of the Schechter and DPL results
    combined_sch_dpl = False
    if combined_sch_dpl:
        combined_uv_redshifts = np.concatenate((sch_uv_redshifts[0:6], dpl_uv_redshifts[-2:]))
        combined_log_uv_N_escs = np.concatenate((log_sch_uv_N_escs[0:6], log_dpl_uv_N_escs[-2:]))
        combined_log_uv_err_low = np.concatenate((log_sch_uv_err_low[0:6], log_dpl_uv_err_low[-2:]))
        combined_log_uv_err_high = np.concatenate((log_sch_uv_err_high[0:6], log_dpl_uv_err_high[-2:]))
        combined_weights = np.concatenate((sch_weights[0:6], dpl_weights[-2:]))
        sch_smooth = (0.2, 0.05)[jwst_sch_funcs]
        sch_spline = UnivariateSpline(combined_uv_redshifts, combined_log_uv_N_escs,
                                w=combined_weights, k=3, s=sch_smooth)
        sch_spline_high = UnivariateSpline(combined_uv_redshifts, combined_log_uv_N_escs + combined_log_uv_err_high,
                                    w=combined_weights, k=3, s=sch_smooth)
        sch_spline_low = UnivariateSpline(combined_uv_redshifts, combined_log_uv_N_escs - combined_log_uv_err_low,
                                    w=combined_weights, k=3, s=sch_smooth)
        with open("final_graph_generation/uv_N_ion_spline_sch.pkl", "wb") as f:
            pickle.dump(sch_spline, f)
        with open("final_graph_generation/uv_N_ion_spline_high_sch.pkl", "wb") as f:
            pickle.dump(sch_spline_high, f)
        with open("final_graph_generation/uv_N_ion_spline_low_sch.pkl", "wb") as f:
            pickle.dump(sch_spline_low, f)

    
# integrates over the number density of galaxies multiplied by n_esc for different magnitude bands
else:
    uv_bands = [(-17, -10), (-20, -17), (-30, -20)]
    split_sch_uv_N_escs = []
    a, b, c = ([-0.631, 0.061, 40.07]) # direct thesan fit values for uv magnitude
    for band in uv_bands:
        sch_uv_N_escs = []
        for z in sch_uv_redshifts:
            n_ion_esc_func = n_ion_esc_linear_2var(z, a, b, c)
            # uv_lum_function = schechter_uv(sch_uv_alpha_fit[z], sch_uv_phi_fit[z], sch_uv_mag_fit[z])
            uv_lum_function = schechter_uv_convolved(mag_min, mag_max, sigma_mag, sch_uv_alpha_fit[z], sch_uv_phi_fit[z], sch_uv_mag_fit[z])
            integrand = lambda mag: n_ion_esc_func(mag) * uv_lum_function(mag)
            mags = np.linspace(band[0], band[1], n_bins)
            result = np.trapz(integrand(mags), mags)
            sch_uv_N_escs.append(result)
        sch_uv_N_escs = np.array(sch_uv_N_escs)
        split_sch_uv_N_escs.append(sch_uv_N_escs)

# Schechter parameters for the number density of galaxies as a function of stellar mass from Weibel et al. (2024)
m_redshifts = np.array([4, 5, 6, 7, 8, 9])
m_alpha_fit = {4: -1.79, 5: -1.86, 6: -1.95, 7: -1.93, 8: -2.08, 9: -2.0}
m_alpha_fit_err = {4: 0.01, 5: 0.03, 6: 0.07, 7: 0.04, 8: 0.19, 9: 0.19}
log_m_phi_fit = {4: -4.52,5 : -4.07, 6: -4.26, 7: -4.36, 8: -4.86, 9: -4.93}
log_m_phi_fit_err = {4: 0.13, 5: 0.13, 6: 0.36, 7: 0.05, 8: 0.20, 9: 0.07}
log_m_mass_fit = {4: 11.01, 5: 10.26, 6: 10.01, 7: 10.0, 8: 10.0, 9: 10.0}
log_m_mass_fit_err = {4: 0.14, 5: 0.12, 6: 0.32, 7: 0.32, 8: 0.32, 9: 0.32}

def schechter_mass(m_alpha_fit, log_m_phi_fit, log_m_mass_fit):
    m_phi_fit = 10**log_m_phi_fit
    m_mass_fit = 10**log_m_mass_fit
    def mass_func(mass):
        return (m_phi_fit / m_mass_fit) * (mass / m_mass_fit)**m_alpha_fit * np.exp(-mass / m_mass_fit)
    return mass_func
def schechter_log_mass( m_alpha_fit, log_m_phi_fit, log_m_mass_fit):
    m_phi_fit = 10**log_m_phi_fit
    def mass_func(log_mass):
        k = 10 ** (log_mass - log_m_mass_fit)
        return np.log(10) * m_phi_fit * (k ** (m_alpha_fit+1)) * np.exp(-k)
    return mass_func

def schechter_mass_convolved(mass_min, mass_max, sigma_mass, m_alpha_fit, log_m_phi_fit, log_m_mass_fit, log_space=True, n_bins=1000):
    # integral over a uniform grid in log mass space
    masses = np.linspace(mass_min, mass_max, n_bins)
    dM = masses[1] - masses[0]
    # original stellar mass function
    if log_space:
        phi = schechter_log_mass(m_alpha_fit, log_m_phi_fit, log_m_mass_fit)(masses)
    else:
        phi = schechter_mass(m_alpha_fit, log_m_phi_fit, log_m_mass_fit)(masses)
    # Gaussian kernel
    kernel_extent = 6 * sigma_mass
    n_kernel = int(np.ceil((2 * kernel_extent) / dM)) + 1
    if n_kernel % 2 == 0:
        n_kernel += 1
    kernel_x = np.linspace(-kernel_extent, kernel_extent, n_kernel)
    gaussian = np.exp(-0.5 * (kernel_x / sigma_mass)**2)
    gaussian /= np.sum(gaussian) * dM
    # convolution
    phi_conv = fftconvolve(phi, gaussian, mode='same') * dM
    # return interpolated convoluted stellar mass function
    return interp1d(masses, phi_conv, bounds_error=False, fill_value=0, assume_sorted=True)

# integrand to integrate over stellar mass to get N_esc
def integrand_mass(mass, z, a, b, c, m_alpha_fit, log_m_phi_fit, log_m_mass_fit):
    number_density_func = schechter_mass(m_alpha_fit, log_m_phi_fit, log_m_mass_fit)
    linear_func = 10 ** linear_2var((a, b, c), (np.log10(float(mass)), z))
    return number_density_func * linear_func
def integrand_log_mass(log_mass, z, a, b, c, m_alpha_fit, log_m_phi_fit, log_m_mass_fit):
    number_density_func = schechter_log_mass(log_mass, m_alpha_fit, log_m_phi_fit, log_m_mass_fit)
    linear_func = 10 ** linear_2var((a, b, c), (log_mass, z))
    return number_density_func * linear_func

# calculates the best fit parameters for stellar mass using ODR to account for x and y errors
subset_indices = range(len(predictions))
x_err_sym = (obvs_star_mass_low_error + obvs_star_mass_high_error) / 2
x_err_median = np.nanmedian(x_err_sym[x_err_sym > 0]) * 1e-6
x_err_sym[x_err_sym <= 0] = x_err_median
z_err_sym = (obvs_redshift_low_error + obvs_redshift_high_error) / 2
z_err_median = np.nanmedian(z_err_sym[z_err_sym > 0]) * 1e-6
z_err_sym[z_err_sym <= 0] = z_err_median
model = odr.Model(linear_2var)
data = odr.RealData((log_star_mass[subset_indices], obvs_redshift[subset_indices]), predictions[subset_indices],
                    sx=(x_err_sym[subset_indices], z_err_sym[subset_indices]), sy=pred_mae[subset_indices])
odr_inst = odr.ODR(data, model, beta0=[0, 0, 0])
out = odr_inst.run()
a, b, c = out.beta
a_err, b_err, c_err = out.sd_beta
popt, pcov = out.beta, out.cov_beta

# calculates the best fit parameters for UV magnitude using WLSR with only y errors
# def curve_fit_func(X, a, b, c):
#     return linear_2var((a, b, c), X)
# popt, pcov = curve_fit(curve_fit_func, (log_star_mass[subset_indices], obvs_redshift[subset_indices]), predictions[subset_indices], 
#                        sigma=pred_mae[subset_indices], absolute_sigma=True)
# a, b, c = popt
# a_err, b_err, c_err = np.sqrt(np.diag(pcov))

a_str, a_err_str = round_to_error(a, a_err)
b_str, b_err_str = round_to_error(b, b_err)
c_str, c_err_str = round_to_error(c, c_err)
print(
    f'\nLog10({["f_esc", "n_ion,esc"][f_or_n]}) = '
    f'({a_str}±{a_err_str})M* + '
    f'({b_str}±{b_err_str})z + '
    f'({c_str}±{c_err_str})'
)

# sigma_mass = [0.876, 0.973][dusty]
sigma_mass = [0.671, 0.709][dusty]
n_bins = 2000
log_mass_min, log_mass_max = 6, 12
mass_min, mass_max = 10**log_mass_min, 10**log_mass_max


# integrates over the number density of galaxies multiplied by n_esc as a function of stellar mass to calculate N_esc
if not split_contribution:
    m_N_escs, m_N_escs_low, m_N_escs_high = [np.zeros(len(m_redshifts)) for _ in range(3)]

    # Monte-Carlo error propagation for N_esc including a b c covariance and Schechter param errors
    n_mc = 1000
    rng = np.random.default_rng()
    # defensive cov handling for a,b,c (should be positive definite and symmetric)
    beta_mean = np.asarray(popt, dtype=float)
    # beta_mean = np.array([1.434, 0.274, 38.25]) # direct thesan fit values for uv magnitude
    beta_cov = np.asarray(pcov, dtype=float)
    beta_cov = (beta_cov + beta_cov.T) / 2
    eig = np.linalg.eigvalsh(beta_cov)
    if np.any(eig < 0):
        beta_cov += np.eye(3) * (abs(np.min(eig)) + 1e-12)
    # draw samples of a,b,c using the covariance matrix
    betas = rng.multivariate_normal(beta_mean, beta_cov, size=n_mc)

    # loop over redshifts (small loop; inner loop does integration)
    for j, z in enumerate(m_redshifts):
        # sample Schechter parameters (fixed if err == 0)
        def draw(mean, err): 
            return rng.normal(mean, err, n_mc) if err > 0 else np.full(n_mc, mean)
        alpha_samples = draw(m_alpha_fit[z],   m_alpha_fit_err[z])
        philog_samples = draw(log_m_phi_fit[z], log_m_phi_fit_err[z])
        masslog_samples = draw(log_m_mass_fit[z], log_m_mass_fit_err[z])

        # integrate for each MC draw
        N_samples = np.empty(n_mc)
        for i in range(n_mc):
            a_s, b_s, c_s = betas[i]
            n_ion_esc_func = n_ion_esc_linear_2var(z, a_s, b_s, c_s)
            # mass_function = schechter_mass(alpha_samples[i], philog_samples[i], masslog_samples[i])
            log_space = True
            int_min = [mass_min, log_mass_min][log_space]
            int_max = [mass_max, log_mass_max][log_space]
            mass_function = schechter_mass_convolved(int_min, int_max, sigma_mass,
                                                        alpha_samples[i], philog_samples[i], masslog_samples[i],
                                                        log_space=log_space, n_bins=n_bins)
            integrand = lambda mass: n_ion_esc_func(mass) * mass_function(mass)
            # N_samples[i], _ = quad(integrand_mass, mass_min, mass_max, args=args, epsrel=1e-3, limit=500)
            # N_samples[i], _ = quad(integrand_log_mass, log_mass_min, log_mass_max, args=args, epsrel=1e-3, limit=500)
            masses = np.linspace(int_min, int_max, n_bins)
            N_samples[i] = np.trapz(integrand(masses), masses)

        # compute median and 16/84 percentiles
        m_N_escs[j] = np.median(N_samples)
        m_N_escs_low[j] = np.percentile(N_samples, 16)
        m_N_escs_high[j] = np.percentile(N_samples, 84)
        print(f"z = {z}: N_esc = {m_N_escs[j]:.3e}, Upper = {m_N_escs_high[j]:.3e}, Lower = {m_N_escs_low[j]:.3e}")

    log_m_N_escs = np.log10(m_N_escs)
    log_m_err_low = log_m_N_escs - np.log10(m_N_escs_low)
    log_m_err_high = np.log10(m_N_escs_high) - log_m_N_escs

# integrates over the number density of galaxies multiplied by n_esc for different mass bands
else:
    m_bands = [(6, 8), (8, 10), (10, 12), (6, 12)]
    split_m_N_escs = []
    for band in m_bands:
        m_N_escs = []
        for z in m_redshifts:
            n_ion_esc_func = n_ion_esc_linear_2var(z, a, b, c)
            # mass_function = schechter_mass(m_alpha_fit[z], log_m_phi_fit[z], log_m_mass_fit[z])
            mass_function = schechter_mass_convolved(log_mass_min, log_mass_max, sigma_mass,
                                                        m_alpha_fit[z], log_m_phi_fit[z], log_m_mass_fit[z],
                                                        log_space=True, n_bins=n_bins)
            integrand = lambda mass: n_ion_esc_func(mass) * mass_function(mass)
            masses = np.linspace(band[0], band[1], n_bins)
            result = np.trapz(integrand(masses), masses)
            m_N_escs.append(result)
        m_N_escs = np.array(m_N_escs)
        split_m_N_escs.append(m_N_escs)


if not split_contribution:
    z_range = (1.5, 15.5)
    z_space = np.linspace(z_range[0], z_range[1], 1000)

    def critical(z, C):
        # omega_b = 0.0486
        # h_50 = 100 * 0.6774 / 50
        # C_30 = C / 30
        # return 10**(51.2) * C_30 * ((1 + z) / 6)**3 * ((omega_b * h_50**2) / 0.08)**2
        Y = 0.24668
        X = 1 - Y
        chi = Y / (4 * X)
        alpha_B = 2.5775e-13 # cm^3 s^-1 at T=1e4 K
        n_H_0 = 1.8786e-7 # cm^-3
        t_rec = 1 / ((1 + chi) * alpha_B * n_H_0 * (1 + z)**3 * C)
        return n_H_0 / t_rec * (3.086e24)**3

    def log_errors(errors, results):
        return errors / (results * np.log(10))

    plt.style.use('./MNRAS_Style.mplstyle')
    mpl.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(figsize=(20, 8))

    # plot critical N_esc curves
    for C in [1, 3, 10]:
        ax.plot(z_space, np.log10(critical(z_space, C)), c='grey', zorder=1)
        text_z = 2.5
        ax.text(text_z, np.log10(critical(text_z, C))-0.1, f'$C = {C}$',
                rotation=30, color='grey')
    ax.fill_between(z_space, np.log10(critical(z_space, 1)), np.log10(critical(z_space, 10)),
                    color='grey', alpha=0.2, zorder=1, label='$\dot{N}_\mathrm{ion}$ Critical')
    
    constraint_colours = {
    # "kuhlen":  "#9EC9E2",
    "becker": "#1F4E79",
    "gaikwad": "#4F81BD",
    # "davies": "#4BACC6",
    "rinaldi": "#9EC9E2",
    }

    # plot observational constraints on N_ion from Kuhlen (2012)
    # constraints_folder = 'other_paper_graphs/'
    # kuhlen_2012 = np.zeros((12, 4))
    # kuhlen_2012_files = ['kuhlen_2012_Nion_values.csv',
    #                       'kuhlen_2012_Nion_errors_low.csv',
    #                       'kuhlen_2012_Nion_errors_high.csv']
    # for i in range(len(kuhlen_2012_files)):
    #     with open(constraints_folder + kuhlen_2012_files[i], newline='', encoding='utf-8') as f:
    #         reader = csv.reader(f, delimiter=',', skipinitialspace=True)
    #         rows = [row for row in reader]
    #         if i == 0:
    #             kuhlen_2012[:,0] = np.array([row[0] for row in rows]).astype('float64')
    #             kuhlen_2012[:,1] = np.array([row[1] for row in rows]).astype('float64') + 51
    #         else:
    #             kuhlen_2012[:,i+1] = np.array([row[1] for row in rows]).astype('float64') + 51
    # ax.errorbar(kuhlen_2012[:,0], kuhlen_2012[:,1],
    #             yerr=(kuhlen_2012[:,1] - kuhlen_2012[:,2], kuhlen_2012[:,3] - kuhlen_2012[:,1]),
    #             fmt='none', c='black', elinewidth=2, capsize=5, zorder=3')
    # ax.scatter(kuhlen_2012[:,0], kuhlen_2012[:,1], s=50, marker='^', c='black', edgecolors='black',
    #            zorder=4, label='kuhlen et al. (2012)')
    # ax.fill_between(kuhlen_2012[:,0], kuhlen_2012[:,2], kuhlen_2012[:,3],
    #                 color=constraint_colours["kuhlen"], alpha=0.7, zorder=2, label='Kuhlen (2012)')

    # plot observational constraints on N_ion from Becker (2013)
    constraints_folder = 'other_paper_graphs/'
    becker_2013 = np.zeros((4, 4))
    becker_2013_files = ['becker_2013_Nion_values.csv',
                          'becker_2013_Nion_errors_low.csv',
                          'becker_2013_Nion_errors_high.csv']
    for i in range(len(becker_2013_files)):
        with open(constraints_folder + becker_2013_files[i], newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',', skipinitialspace=True)
            rows = [row for row in reader]
            if i == 0:
                becker_2013[:,0] = np.array([row[0] for row in rows]).astype('float64')
                becker_2013[:,1] = np.array([row[1] for row in rows]).astype('float64') + 51
            else:
                becker_2013[:,i+1] = np.array([row[1] for row in rows]).astype('float64') + 51
    # ax.errorbar(becker_2013[:,0], becker_2013[:,1],
    #             yerr=(becker_2013[:,1] - becker_2013[:,2], becker_2013[:,3] - becker_2013[:,1]),
    #             fmt='none', c='black', elinewidth=2, capsize=5, zorder=3)
    # ax.scatter(becker_2013[:,0], becker_2013[:,1], s=50, marker='o', c='black', edgecolors='black',
    #            zorder=4, label='Becker et al. (2013)')
    ax.fill_between(becker_2013[:,0], becker_2013[:,2], becker_2013[:,3],
                    color=constraint_colours["becker"], alpha=0.6, zorder=2, label='Becker (2013)')

    # plot observational constraints on N_ion from Gaikwad (2023)
    constraints_folder = 'other_paper_graphs/'
    gaikwad_2023 = np.zeros((12, 4))
    gaikwad_2023_files = ['gaikwad_2023_Nion_values.csv',
                          'gaikwad_2023_Nion_errors_low.csv',
                          'gaikwad_2023_Nion_errors_high.csv']
    for i in range(len(gaikwad_2023_files)):
        with open(constraints_folder + gaikwad_2023_files[i], newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',', skipinitialspace=True)
            rows = [row for row in reader]
            if i == 0:
                gaikwad_2023[:,0] = np.array([row[0] for row in rows]).astype('float64')
                gaikwad_2023[:,1] = np.array([row[1] for row in rows]).astype('float64')
            else:
                gaikwad_2023[:,i+1] = np.array([row[1] for row in rows]).astype('float64')
    # ax.errorbar(gaikwad_2023[:,0], gaikwad_2023[:,1],
    #             yerr=(gaikwad_2023[:,1] - gaikwad_2023[:,2], gaikwad_2023[:,3] - gaikwad_2023[:,1]),
    #             fmt='o', c='black', elinewidth=2, capsize=5, zorder=3)
    # ax.scatter(gaikwad_2023[:,0], gaikwad_2023[:,1], s=50, marker='*', c='black', edgecolors='black',
    #            zorder=4, label='Gaikwad et al. (2023)')
    ax.fill_between(gaikwad_2023[:,0], gaikwad_2023[:,2], gaikwad_2023[:,3],
                    color=constraint_colours["gaikwad"], alpha=0.5, zorder=2, label='Gaikwad (2023)')
    
    # plot observational constraints on N_ion from Davies (2024)
    # constraints_folder = 'other_paper_graphs/'
    # davies_2024 = np.zeros((5, 4))
    # davies_2024_files = ['davies_2024_Nion_values.csv',
    #                      'davies_2024_Nion_errors_low.csv',
    #                      'davies_2024_Nion_errors_high.csv']
    # for i in range(len(davies_2024_files)):
    #     with open(constraints_folder + davies_2024_files[i], newline='', encoding='utf-8') as f:
    #         reader = csv.reader(f, delimiter=',', skipinitialspace=True)
    #         rows = [row for row in reader]
    #         if i == 0:
    #             davies_2024[:,0] = np.array([row[0] for row in rows]).astype('float64')
    #             davies_2024[:,1] = np.log10(np.array([row[1] for row in rows]).astype('float64'))
    #         else:
    #             davies_2024[:,i+1] = np.log10(np.array([row[1] for row in rows]).astype('float64'))
    # ax.errorbar(davies_2024[:,0], davies_2024[:,1],
    #             yerr=(davies_2024[:,1] - davies_2024[:,2], davies_2024[:,3] - davies_2024[:,1]),
    #             fmt='o', c='black', elinewidth=2, capsize=5, zorder=3)
    # ax.scatter(davies_2024[:,0], davies_2024[:,1], s=50, marker='*', c='black', edgecolors='black',
    #            zorder=4, label='Davies et al. (2024)')
    # ax.fill_between(davies_2024[:,0], davies_2024[:,2], davies_2024[:,3],
    #                 color=constraint_colours["davies"], alpha=0.5, zorder=2, label='Davies (2024)')
    
    # plot obsrevational constraints on N_ion from Rinaldi (2024)
    rinaldi_2024_z = [7, 8]
    rinaldi_2024_N_ion = np.array([50.53, 50.53])
    rinaldi_2024_N_ion_err = np.array([0.45, 0.45])
    ax.fill_between(rinaldi_2024_z,
                    rinaldi_2024_N_ion - rinaldi_2024_N_ion_err, rinaldi_2024_N_ion + rinaldi_2024_N_ion_err,
                    color=constraint_colours['rinaldi'], alpha=0.7, zorder=2, label='Rinaldi (2024)')
    
    # plot Charlotte's N_ion integrations for her f_esc = 10% and f_esc Chisholm (2022) prescriptions
    charlotte_z = [3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
    N_ion_f_esc_10 = [51.23478634, 50.85474357, 50.60201125, 50.57420019, 50.34034352, 49.99205213]
    N_ion_f_esc_chisholm = [50.76232333, 50.52949668, 50.52709877, 50.61663425, 50.36362092, 49.44427493]
    ax.errorbar(charlotte_z, N_ion_f_esc_10, yerr=0.3,
                fmt='none', c='gold', elinewidth=2, alpha=0.7, capsize=0, zorder=2)
    ax.plot(charlotte_z, N_ion_f_esc_10, linestyle=':', c='gold', linewidth=3, zorder=2)
    ax.scatter(charlotte_z, N_ion_f_esc_10, s=150, marker='^', c='gold', edgecolors='black', zorder=3,
            label='Simmonds (2024b), $f_\mathrm{esc} = 10\%$')
    ax.errorbar(charlotte_z, N_ion_f_esc_chisholm, yerr=0.3,
                fmt='none', c='limegreen', elinewidth=2, alpha=0.7, capsize=0, zorder=2)
    ax.plot(charlotte_z, N_ion_f_esc_chisholm, linestyle=':', c='limegreen', linewidth=3, zorder=2)
    ax.scatter(charlotte_z, N_ion_f_esc_chisholm, s=150, marker='v', c='limegreen', edgecolors='black', zorder=3,
            label='Simmonds (2024b), $f_\mathrm{esc}$ Chisholm (2022)')
    
    ax.errorbar(m_redshifts, log_m_N_escs, yerr=(log_m_err_low, log_m_err_high),
                fmt='none', c='darkorange', elinewidth=2, capsize=5, zorder=3)
    ax.plot(m_redshifts, log_m_N_escs, linestyle='--', c='darkorange', linewidth=3, zorder=4)
    ax.scatter(m_redshifts, log_m_N_escs, s=150, c='darkorange',  edgecolors='black', zorder=5,
            label='$\dot{N}_\mathrm{ion}$ Stellar Mass Integration')
    
    ax.errorbar(sch_uv_redshifts, log_sch_uv_N_escs, yerr=(log_sch_uv_err_low, log_sch_uv_err_high),
                fmt='none', c='darkorchid', elinewidth=2, capsize=5, zorder=4)
    ax.plot(sch_uv_redshifts, log_sch_uv_N_escs, linestyle='--', c='darkorchid', linewidth=3, zorder=2)
    ax.scatter(sch_uv_redshifts, log_sch_uv_N_escs, s=150, c='darkorchid', edgecolors='black', zorder=5,
            label='$\dot{N}_\mathrm{ion}$ UV Magnitude Integration')

    ax.errorbar(dpl_uv_redshifts, log_dpl_uv_N_escs, yerr=(log_dpl_uv_err_low, log_dpl_uv_err_high),
                fmt='none', c='crimson', elinewidth=2, capsize=5, zorder=4)
    ax.plot(dpl_uv_redshifts, log_dpl_uv_N_escs, linestyle='--', c='crimson', linewidth=3, zorder=2)
    ax.scatter(dpl_uv_redshifts, log_dpl_uv_N_escs, s=150, c='crimson', edgecolors='black', zorder=5,
            label='$\dot{N}_\mathrm{ion}$ UV Magnitude Integration')
    
    # ax.plot(z_space, sch_polynomial(z_space), alpha = 0.8, linestyle='-', c='black', linewidth=2,
    #         zorder=3, label='UV $\dot{N}_\mathrm{ion}$ Polynomial Fit')
    # ax.plot(z_space, sch_spline_low(z_space), alpha = 0.8, linestyle='-', c='black', linewidth=2,
    #         zorder=3, label='UV $\dot{N}_\mathrm{ion}$ Spline Fit')
    # ax.plot(z_space, dpl_polynomial(z_space), alpha = 0.8, linestyle='-', c='black', linewidth=2,
    #         zorder=3, label='UV $\dot{N}_\mathrm{ion}$ Polynomial Fit')
    # ax.plot(z_space, dpl_spline_low(z_space), alpha = 0.8, linestyle='-', c='black', linewidth=2,
    #         zorder=3, label='UV $\dot{N}_\mathrm{ion}$ Spline Fit')

    ax.set_xlabel("$z$")
    ax.set_ylabel("$\mathrm{log}_{10}(\dot{N}_\mathrm{ion} \; [\mathrm{s^{-1} \; cMpc^{-3}}])$")
    ax.yaxis.set_label_coords(-0.075, 0.5)
    ax.set_xlim(z_range)
    ax.set_ylim((48.9, 51.9))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=12, integer=True))
    ax.grid(False)
    # ax.grid(True, alpha=0.8, linestyle='--')
    # ax.set_axisbelow(True)
    # for line in ax.get_xgridlines() + ax.get_ygridlines():
    #     line.set_zorder(0)
    legend = ax.legend(fontsize=18, loc='center left', bbox_to_anchor=(1.05, 0.5), borderaxespad=0)
    frame = legend.get_frame()
    frame.set_edgecolor('black')
    frame.set_boxstyle('Square')
    frame.set_alpha(0.8)
    plt.subplots_adjust(right=0.7)

else:

    plt.style.use('./MNRAS_Style.mplstyle')
    mpl.rcParams.update({'font.size': 20})
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    bar_width = 0.25
    offsets = [-bar_width, 0, bar_width]
    uv_labels = ('$M_\mathrm{UV} < -17$',
                 '$-20 < M_\mathrm{UV} \leq -17$',
                 '$M_\mathrm{UV} \geq -20$')
    m_labels = ('$\mathrm{log}_{10}(M_*) < 8$',
                '$8 \leq \mathrm{log}_{10}(M_*) < 10$',
                '$\mathrm{log}_{10}(M_*) \geq 10$')
    uv_colors = ['#66c2a5', '#3288bd', '#5e4fa2']  # Teal, Medium Blue, Deep Blue
    m_colors = ['#fdb863', '#e66101', '#b2182b']  # Gold, Orange, Crimson

    for ax_i in range(len(axes)):
        split_N_escs = [[arr[2:8] for arr in split_sch_uv_N_escs], split_m_N_escs][ax_i]
        redshifts = [np.array([3.8, 4.9, 5.9 , 6.8, 7.9, 9.8]), np.array([4, 5, 6, 7, 8, 9])][ax_i]
        redshift_positions = np.array([4, 5, 6, 7, 8, 9])
        labels = [uv_labels, m_labels][ax_i]
        colors = [uv_colors, m_colors][ax_i]
        for i in range(3):
            axes[ax_i].bar(redshift_positions + offsets[i], split_N_escs[i]/10**50,
                           width=bar_width, label=labels[i], color=colors[i], edgecolor='black')

        axes[ax_i].set_xticks(redshift_positions)
        axes[ax_i].set_xticklabels([str(z) for z in redshifts])
        axes[ax_i].set_xlabel('$z$')
        axes[ax_i].tick_params(axis='x', which='both', bottom=False, top=False)
        axes[ax_i].set_ylabel('$\dot{N}_{\mathrm{ion}} \; [10^{50} \; \mathrm{s^{-1} \; cpc^{-3}}]$')
        axes[ax_i].yaxis.set_label_coords(-0.10, 0.5)
        axes[ax_i].set_xlim(3.25, 9.75)
        axes[ax_i].set_ylim(0, 8)
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