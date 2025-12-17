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

# False for plotting total N_esc against redshift and True for plotting mass and magnitude bands
split_contribution = False

# Do integrations with double power law UV luminosity functions
include_dpl = False

# Do integrations with stellar mass functions
include_stellar_mass = False

# Whether to use the random forest models trained with the dusty or dust-free Thesan-Zoom data
dusty = True

folder = "final_rf_model/"
if dusty:
    file = folder + 'n_esc_rf_observational_charlotte_dusty_test_train.json'
else:
    file = folder + 'n_esc_rf_observational_charlotte_test_train.json'
# file = folder + 'n_esc_rf_observational_charlotte_dusty_test_train_test.json'

# Thesan data
thesan_file = ['cat.hdf5', 'cat_dusttestszeb.hdf5'][dusty]
# thesan_file = 'cat_dusttestszeb_mcrtesc.hdf5'
thesan_keys, thesan_log_vars, thesan_log_f_esc, thesan_log_n_esc = prepare_data(thesan_file, f_or_n=1, obvs=True,
                                                                                dusty=dusty, eps=True, add_vars=['redshift_full'])
thesan_log_uv_mag = thesan_log_vars[3]
thesan_log_star_mass = thesan_log_vars[2]
thesan_redshift = 10**thesan_log_vars[-1]

# Prepare the observational data
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
random_variable = np.random.uniform(low=0, high=1, size=(len(obvs_redshift)))

obvs_n_esc_vars = [obvs_sfr10, obvs_sfr100, obvs_star_mass, random_variable, 1+obvs_redshift, random_variable]
obvs_n_esc_keys = ['sfr10', 'sfr100', 'star_mass', 'uv_mag', '1 + redshift', 'random_variable']
obvs_log_n_esc_vars = np.log10(obvs_n_esc_vars).astype('float32')

# add the uv magnitude to the variables
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
for i in range(len(obvs_n_esc_vars)):
    b_i = [index for index, val in enumerate(list((obvs_n_esc_vars)[i]))
                    if (val == 0 or val == np.inf or val== -np.inf or np.isnan(val))]
    print(f"feature {i+1} bad rows: {len(b_i)}")
    bad_indices += b_i
bad_indices = list(set(bad_indices))[::-1]
obvs_redshift = np.delete(obvs_redshift, bad_indices)
obvs_n_esc_vars = np.delete(obvs_n_esc_vars, bad_indices, axis=1)
obvs_log_n_esc_vars = np.delete(obvs_log_n_esc_vars, bad_indices, axis=1)
obvs_star_mass_high_error = np.delete(np.array(obvs_data['log(Mstar)_ehi']), bad_indices)
obvs_star_mass_low_error = np.delete(np.array(obvs_data['log(Mstar)_elo']), bad_indices)
obvs_uv_mag_high_error = np.delete(np.array(obvs_data['M1500obs_ehi']), bad_indices)
obvs_uv_mag_low_error = np.delete(np.array(obvs_data['M1500obs_elo']), bad_indices)
obvs_redshift_high_error = np.delete(np.array(obvs_data['z_ehi']), bad_indices)
obvs_redshift_low_error = np.delete(np.array(obvs_data['z_elo']), bad_indices)
print(f'rows remaining: {len(obvs_redshift)}')

obvs_vars = obvs_n_esc_vars
obvs_log_vars = obvs_log_n_esc_vars
obvs_keys = obvs_n_esc_keys
obvs_log_uv_mag = obvs_log_vars[3]
obvs_log_star_mass = obvs_log_vars[2]
print(obvs_keys)

X = np.transpose(obvs_log_vars)
# run observational target prediction using loaded model
if not dusty:
    loaded_model = joblib.load(folder + 'n_esc_rf_observational_charlotte_model.pkl')
else:
    loaded_model = joblib.load(folder + 'n_esc_rf_observational_charlotte_dusty_model.pkl')
# loaded_model = joblib.load(folder + 'n_esc_rf_observational_charlotte_dusty_model_test.pkl')
pred_log_n_esc = loaded_model.predict(X)
with open(file, 'r') as json_data:
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
    pred_mae = np.zeros(len(pred_log_n_esc))
    for i in range(len(pred_log_n_esc)):
        index = np.argmin(np.abs(test_pred_medians - pred_log_n_esc[i]))
        pred_mae[i] = (test_pred_mae[index])


all_log_n_esc = [thesan_log_n_esc, pred_log_n_esc]
all_log_uv_mag = [thesan_log_uv_mag, obvs_log_uv_mag]
all_log_star_mass = [thesan_log_star_mass, obvs_log_star_mass]
all_redshift = [thesan_redshift, obvs_redshift]
all_log_n_esc_err = [np.zeros_like(thesan_log_n_esc), pred_mae]
all_log_uv_mag_err = [np.zeros_like(thesan_log_uv_mag), (obvs_uv_mag_high_error + obvs_uv_mag_low_error) / 2]
all_log_star_mass_err = [np.zeros_like(thesan_log_star_mass), (obvs_star_mass_high_error + obvs_star_mass_low_error) / 2]
all_redshift_err = [np.zeros_like(thesan_redshift), (obvs_redshift_high_error + obvs_redshift_low_error) / 2]


def linear_1var(p, X):
    a, b = p
    return a * X + b

def linear_2var(p, X):
    a, b, c = p
    x, z = X
    return a * x + b * z + c

def curve_fit_func(X, a, b, c):
    return linear_2var((a, b, c), X)

def n_ion_esc_linear_1var(a, b):
    def linear_func(mag):
        return 10 ** linear_1var((a, b), mag)
    return linear_func

def n_ion_esc_linear_2var(z, a, b, c):
    def linear_func(mag):
        return 10 ** linear_2var((a, b, c), (mag, np.log10(1+z)))
    return linear_func

def draw(mean, err): 
    return rng.normal(mean, err, n_mc) if err > 0 else np.full(n_mc, mean)



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
    dpl_uv_redshifts = dpl_uv_redshifts[[0, 1, 2, 3, 4]]
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


sigma_mag = [0.693, 0.703][dusty] # standard deviation of UV magnitude and n_esc residuals
# sigma_mag = 1.4
n_bins = 2000 # number of bins for UV convolution
n_mc = 20000 # number of Monte-Carlo samples for N_esc integrations
mag_min, mag_max = -33, -13 # UV magnitude limits for integrations
uv_bands = [(-17, -13), (-20, -17), (-33, -20)] # UV magnitude bins for split contributions

all_log_sch_uv_N_escs, all_log_sch_uv_err_low, all_log_sch_uv_err_high = ([], [], [])
all_log_dpl_uv_N_escs, all_log_dpl_uv_err_low, all_log_dpl_uv_err_high = ([], [], [])
all_sch_polynomial, all_sch_spline, all_sch_spline_high, all_sch_spline_low = ([], [], [], [])
all_dpl_polynomial, all_dpl_spline, all_dpl_spline_high, all_dpl_spline_low = ([], [], [], [])
all_combined_polynomial, all_combined_spline, all_combined_spline_high, all_combined_spline_low = ([], [], [], [])
all_split_sch_uv_N_escs = []
for cat in range(len(all_log_n_esc)):
    log_n_esc = all_log_n_esc[cat]
    log_uv_mag = all_log_uv_mag[cat]
    redshift = all_redshift[cat]
    log_n_esc_err = all_log_n_esc_err[cat]
    log_uv_mag_err = all_log_uv_mag_err[cat]
    redshift_err = all_redshift_err[cat]
    log_redshfit = np.log10(1 + redshift)
    sigma_mag = (sigma_mag, 1.0)[cat]  # Assume a different scatter for observations than what we find in Thesan-Zoom

    subset_indices = range(len(log_n_esc))
    # subset_indices = np.where(log_uv_mag < -0)[0]
    if sum(log_n_esc_err) > 0:
        # calculates the best fit parameters for UV magnitude using ODR to account for x and y errors
        x_err_sym = log_uv_mag_err
        x_err_median = np.nanmedian(x_err_sym[x_err_sym > 0]) * 1e-6
        x_err_sym[x_err_sym <= 0] = x_err_median
        # z_err_sym = redshift_err
        z_err_sym = redshift_err / (np.log(10) * (1+redshift)) 
        z_err_median = np.nanmedian(z_err_sym[z_err_sym > 0]) * 1e-6
        z_err_sym[z_err_sym <= 0] = z_err_median
        model = odr.Model(linear_2var)
        data = odr.RealData((log_uv_mag[subset_indices], log_redshfit[subset_indices]), log_n_esc[subset_indices],
                            sx=(x_err_sym[subset_indices], z_err_sym[subset_indices]), sy=pred_mae[subset_indices])
        odr_inst = odr.ODR(data, model, beta0=[0, 0, 0])
        out = odr_inst.run()
        a, b, c = out.beta
        a_err, b_err, c_err = out.sd_beta
        popt, pcov = out.beta, out.cov_beta
    
    else:
        # calculates the best fit parameters for UV magnitude using WLSR
        popt, pcov = curve_fit(curve_fit_func, (log_uv_mag[subset_indices], log_redshfit[subset_indices]), log_n_esc[subset_indices])
        a, b, c = popt
        a_err, b_err, c_err = np.sqrt(np.diag(pcov))

    a_str, a_err_str = round_to_error(a, a_err)
    b_str, b_err_str = round_to_error(b, b_err)
    c_str, c_err_str = round_to_error(c, c_err)
    print('')
    print(
        f'log10(n_esc) = '
        f'({a_str}±{a_err_str})Muv + '
        f'({b_str}±{b_err_str})log10(1+z) + '
        f'({c_str}±{c_err_str})'
    )

    # integrates over the number density of galaxies multiplied by n_esc as a function of UV luminosity to calculate N_esc
    if not split_contribution:
        sch_uv_N_escs, sch_uv_N_escs_low, sch_uv_N_escs_high = [np.zeros(len(sch_uv_redshifts)) for _ in range(3)]
        dpl_uv_N_escs, dpl_uv_N_escs_low, dpl_uv_N_escs_high = [np.zeros(len(dpl_uv_redshifts)) for _ in range(3)]

        # Monte-Carlo error propagation for UV N_esc integrations including a b c covariance and UV parameter errors
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
        sch_smooth = (0.2, 0.04)[jwst_sch_funcs]
        sch_spline = UnivariateSpline(sch_uv_redshifts, log_sch_uv_N_escs,
                                w=sch_weights, k=3, s=sch_smooth)
        sch_spline_high = UnivariateSpline(sch_uv_redshifts, log_sch_uv_N_escs + log_sch_uv_err_high,
                                    w=sch_weights, k=3, s=sch_smooth)
        sch_spline_low = UnivariateSpline(sch_uv_redshifts, log_sch_uv_N_escs - log_sch_uv_err_low,
                                    w=sch_weights, k=3, s=sch_smooth)
        
        all_log_sch_uv_N_escs.append(log_sch_uv_N_escs)
        all_log_sch_uv_err_low.append(log_sch_uv_err_low)
        all_log_sch_uv_err_high.append(log_sch_uv_err_high)
        all_sch_polynomial.append(sch_polynomial)
        all_sch_spline.append(sch_spline)
        all_sch_spline_high.append(sch_spline_high)
        all_sch_spline_low.append(sch_spline_low)


        # MC analysis and integrations for DPL UV functions
        if include_dpl:
            print('')
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
            dpl_smooth = (0.2, 0.04)[jwst_sch_funcs]
            dpl_spline = UnivariateSpline(dpl_uv_redshifts, log_dpl_uv_N_escs,
                                    w=dpl_weights, k=3, s=dpl_smooth)
            dpl_spline_high = UnivariateSpline(dpl_uv_redshifts, log_dpl_uv_N_escs + log_dpl_uv_err_high,
                                        w=dpl_weights, k=3, s=dpl_smooth)
            dpl_spline_low = UnivariateSpline(dpl_uv_redshifts, log_dpl_uv_N_escs - log_dpl_uv_err_low,
                                        w=dpl_weights, k=3, s=dpl_smooth)
            # with open("final_graph_generation/uv_N_ion_spline_dpl.pkl", "wb") as f:
            #     pickle.dump(dpl_spline, f)
            # with open("final_graph_generation/uv_N_ion_spline_high_dpl.pkl", "wb") as f:
            #     pickle.dump(dpl_spline_high, f)
            # with open("final_graph_generation/uv_N_ion_spline_low_dpl.pkl", "wb") as f:
            #     pickle.dump(dpl_spline_low, f)

            all_log_dpl_uv_N_escs.append(log_dpl_uv_N_escs)
            all_log_dpl_uv_err_low.append(log_dpl_uv_err_low)
            all_log_dpl_uv_err_high.append(log_dpl_uv_err_high)
            all_dpl_polynomial.append(dpl_polynomial)
            all_dpl_spline.append(dpl_spline)
            all_dpl_spline_high.append(dpl_spline_high)
            all_dpl_spline_low.append(dpl_spline_low)


            # interpolates a combination of the Schechter and DPL results
            combined_sch_dpl = False
            if combined_sch_dpl:
                combined_uv_redshifts = np.concatenate((sch_uv_redshifts[0:6], dpl_uv_redshifts[-2:]))
                combined_log_uv_N_escs = np.concatenate((log_sch_uv_N_escs[0:6], log_dpl_uv_N_escs[-2:]))
                combined_log_uv_err_low = np.concatenate((log_sch_uv_err_low[0:6], log_dpl_uv_err_low[-2:]))
                combined_log_uv_err_high = np.concatenate((log_sch_uv_err_high[0:6], log_dpl_uv_err_high[-2:]))
                combined_weights = np.concatenate((sch_weights[0:6], dpl_weights[-2:]))
                combined_poly_coeffs, combined_poly_cov = np.polyfit(1 + combined_uv_redshifts, combined_log_uv_N_escs, deg=3, w=combined_weights, cov=True)
                combined_poly_coeffs_errors = np.sqrt(np.diag(combined_poly_cov))
                combined_polynomial = np.poly1d(combined_poly_coeffs)
                combined_smooth = (0.2, 0.04)[jwst_sch_funcs]
                combined_spline = UnivariateSpline(combined_uv_redshifts, combined_log_uv_N_escs,
                                        w=combined_weights, k=3, s=combined_smooth)
                combined_spline_high = UnivariateSpline(combined_uv_redshifts, combined_log_uv_N_escs + combined_log_uv_err_high,
                                            w=combined_weights, k=3, s=combined_smooth)
                combined_spline_low = UnivariateSpline(combined_uv_redshifts, combined_log_uv_N_escs - combined_log_uv_err_low,
                                            w=combined_weights, k=3, s=combined_smooth)

                all_combined_polynomial.append(combined_polynomial)
                all_combined_spline.append(combined_spline)
                all_combined_spline_high.append(combined_spline_high)
                all_combined_spline_low.append(combined_spline_low)

    
    # integrates over the number density of galaxies multiplied by n_esc for different magnitude bands
    else:
        split_sch_uv_N_escs = []
        # a, b, c = ([-0.631, 0.061, 40.07]) # direct thesan fit values for uv magnitude
        for band in uv_bands:
            sch_uv_N_escs = []
            for z in sch_uv_redshifts:
                n_ion_esc_func = n_ion_esc_linear_2var(z, a, b, c)
                # uv_lum_function = schechter_uv(sch_uv_alpha_fit[z], sch_uv_phi_fit[z], sch_uv_mag_fit[z])
                uv_lum_function = schechter_uv_convolved(mag_min, mag_max, sigma_mag,
                                                         sch_uv_alpha_fit[z], sch_uv_phi_fit[z], sch_uv_mag_fit[z])
                integrand = lambda mag: n_ion_esc_func(mag) * uv_lum_function(mag)
                mags = np.linspace(band[0], band[1], n_bins)
                result = np.trapz(integrand(mags), mags)
                sch_uv_N_escs.append(result)
            sch_uv_N_escs = np.array(sch_uv_N_escs)
            split_sch_uv_N_escs.append(sch_uv_N_escs)
        all_split_sch_uv_N_escs.append(split_sch_uv_N_escs)



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

if include_stellar_mass:

    sigma_mass = [1.020, 0.973][dusty] # standard deviation of stellar mass and n_esc residuals
    sigma_mass = [0.455, 0.471][dusty]  # standard deviation of stellar mass and n_esc residuals reversed
    n_bins = 2000 # number of bins for stellar mass convolution
    n_mc = 2000 # number of Monte-Carlo samples for N_esc integrations
    log_mass_min, log_mass_max = 6, 12 # logarithmic stellar mass limits for integrations
    mass_min, mass_max = 10**log_mass_min, 10**log_mass_max # stellar mass limits for integrations
    m_bands = [(6, 8), (8, 10), (10, 12), (6, 12)] # logarithmic stellar mass bins for split contributions

    all_log_m_N_escs, all_log_m_err_low, all_log_m_err_high = ([], [], [])
    all_split_m_N_escs = []
    for cat in range(len(all_log_n_esc)):
        log_n_esc = all_log_n_esc[cat]
        log_star_mass = all_log_star_mass[cat]
        redshift = all_redshift[cat]
        log_n_esc_err = all_log_n_esc_err[cat]
        log_star_mass_err = all_log_star_mass_err[cat]
        redshift_err = all_redshift_err[cat]
        log_redshift = np.log10(1 + redshift)

        subset_indices = range(len(log_n_esc))
        if sum(log_n_esc_err) > 0:
            # calculates the best fit parameters for stellar mass using ODR to account for x and y errors
            x_err_sym = log_star_mass_err
            x_err_median = np.nanmedian(x_err_sym[x_err_sym > 0]) * 1e-6
            x_err_sym[x_err_sym <= 0] = x_err_median
            z_err_sym = z_err_sym = redshift_err / (np.log(10) * (1+redshift)) 
            z_err_median = np.nanmedian(z_err_sym[z_err_sym > 0]) * 1e-6
            z_err_sym[z_err_sym <= 0] = z_err_median
            model = odr.Model(linear_2var)
            data = odr.RealData((log_star_mass[subset_indices], log_redshift[subset_indices]), log_n_esc[subset_indices],
                                sx=(x_err_sym[subset_indices], z_err_sym[subset_indices]), sy=pred_mae[subset_indices])
            odr_inst = odr.ODR(data, model, beta0=[0, 0, 0])
            out = odr_inst.run()
            a, b, c = out.beta
            a_err, b_err, c_err = out.sd_beta
            popt, pcov = out.beta, out.cov_beta
        
        else:
            # calculates the best fit parameters for UV magnitude using WLSR
            popt, pcov = curve_fit(curve_fit_func, (log_star_mass[subset_indices], log_redshift[subset_indices]), log_n_esc[subset_indices])
            a, b, c = popt
            a_err, b_err, c_err = np.sqrt(np.diag(pcov))

        a_str, a_err_str = round_to_error(a, a_err)
        b_str, b_err_str = round_to_error(b, b_err)
        c_str, c_err_str = round_to_error(c, c_err)
        print(
            f'\nLog10(n_esc) = '
            f'({a_str}±{a_err_str})M* + '
            f'({b_str}±{b_err_str})z + '
            f'({c_str}±{c_err_str})'
        )


        # integrates over the number density of galaxies multiplied by n_esc as a function of stellar mass to calculate N_esc
        if not split_contribution:
            m_N_escs, m_N_escs_low, m_N_escs_high = [np.zeros(len(m_redshifts)) for _ in range(3)]

            # Monte-Carlo error propagation for N_esc including a b c covariance and Schechter param errors
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

            all_log_m_N_escs.append(log_m_N_escs)
            all_log_m_err_low.append(log_m_err_low)
            all_log_m_err_high.append(log_m_err_high)

        # integrates over the number density of galaxies multiplied by n_esc for different mass bands
        # else:
        #     split_m_N_escs = []
        #     for band in m_bands:
        #         m_N_escs = []
        #         for z in m_redshifts:
        #             n_ion_esc_func = n_ion_esc_linear_2var(z, a, b, c)
        #             # mass_function = schechter_mass(m_alpha_fit[z], log_m_phi_fit[z], log_m_mass_fit[z])
        #             mass_function = schechter_mass_convolved(log_mass_min, log_mass_max, sigma_mass,
        #                                                         m_alpha_fit[z], log_m_phi_fit[z], log_m_mass_fit[z],
        #                                                         log_space=True, n_bins=n_bins)
        #             integrand = lambda mass: n_ion_esc_func(mass) * mass_function(mass)
        #             masses = np.linspace(band[0], band[1], n_bins)
        #             result = np.trapz(integrand(masses), masses)
        #             m_N_escs.append(result)
        #         m_N_escs = np.array(m_N_escs)
        #         split_m_N_escs.append(m_N_escs)
        #     all_split_m_N_escs.append(split_m_N_escs)



if not split_contribution:
    z_range = (1.5, 10.5)
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
    mpl.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(figsize=(20, 8))

    # plot critical N_esc curves
    for C in [1, 3, 10]:
        ax.plot(z_space, np.log10(critical(z_space, C)), c='grey', zorder=1)
        text_z = 2.5
        ax.text(text_z, np.log10(critical(text_z, C)) - (0.10, 0.13)[include_stellar_mass], f'$C = {C}$',
                rotation=(33, 28)[include_stellar_mass], color='grey')
    ax.fill_between(z_space, np.log10(critical(z_space, 1)), np.log10(critical(z_space, 10)),
                    color='grey', alpha=0.2, zorder=1, label='Madau+1999')
    
    constraint_colours = {
    "becker":  "#FFA54C",
    "kuhlen": "#FF7A30",
    "davies": "#FF4C1A",
    "gaikwad": "#E22A10",
    "rinaldi": "#B21807",
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
    #                 color=constraint_colours["kuhlen"], alpha=0.5, zorder=2, label='Kuhlen+2012')

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
                    color=constraint_colours["becker"], alpha=0.5, zorder=2, label='Becker+2013')

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
                    color=constraint_colours["gaikwad"], alpha=0.5, zorder=2, label='Gaikwad+2023')
    
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
    #                 color=constraint_colours["davies"], alpha=0.5, zorder=2, label='Davies+2024')
    
    # plot obsrevational constraints on N_ion from Rinaldi (2024)
    rinaldi_2024_z = [7, 8]
    rinaldi_2024_N_ion = np.array([50.53, 50.53])
    rinaldi_2024_N_ion_err = np.array([0.45, 0.45])
    ax.fill_between(rinaldi_2024_z,
                    rinaldi_2024_N_ion - rinaldi_2024_N_ion_err, rinaldi_2024_N_ion + rinaldi_2024_N_ion_err,
                    color=constraint_colours['rinaldi'], alpha=0.5, zorder=2, label='Rinaldi+2024')
    
    if not include_stellar_mass and not include_dpl:
        # plot Charlotte's N_ion integrations for her f_esc = 10% and f_esc Chisholm (2022) prescriptions
        charlotte_z = [3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
        N_ion_f_esc_10 = [51.23478634, 50.85474357, 50.60201125, 50.57420019, 50.34034352, 49.99205213]
        N_ion_f_esc_chisholm = [50.76232333, 50.52949668, 50.52709877, 50.61663425, 50.36362092, 49.44427493]
        ax.errorbar(charlotte_z, N_ion_f_esc_10, yerr=0.3,
                    fmt='none', c='crimson', elinewidth=2, alpha=0.8, capsize=0, zorder=2)
        ax.plot(charlotte_z, N_ion_f_esc_10, linestyle=':', c='crimson', linewidth=2.5, alpha=0.8, zorder=2)
        ax.scatter(charlotte_z, N_ion_f_esc_10, s=150, marker='^', c='crimson', edgecolors='black', zorder=3,
                label='Simmonds+2024, $f_\mathrm{esc}$ = 10\%')
        ax.errorbar(charlotte_z, N_ion_f_esc_chisholm, yerr=0.3,
                    fmt='none', c='yellowgreen', elinewidth=2, alpha=0.8, capsize=0, zorder=2)
        ax.plot(charlotte_z, N_ion_f_esc_chisholm, linestyle=':', c='yellowgreen', linewidth=2.5, alpha=0.8, zorder=2)
        ax.scatter(charlotte_z, N_ion_f_esc_chisholm, s=150, marker='v', c='yellowgreen', edgecolors='black', zorder=3,
                label='Simmonds+2024, $f_\mathrm{esc}$ = Chisholm+2022')
    
    # plot this work's Thesan-Zoom derived Schechter UV magnitude N_esc integrations
    ax.errorbar(sch_uv_redshifts, all_log_sch_uv_N_escs[0], yerr=(all_log_sch_uv_err_low[0], all_log_sch_uv_err_high[0]),
                fmt='none', c='darkviolet', elinewidth=2, capsize=5, alpha=0.8, zorder=4)
    ax.plot(sch_uv_redshifts, all_log_sch_uv_N_escs[0], linestyle='--', c='darkviolet', linewidth=2.5, alpha=0.8, zorder=2)
    ax.scatter(sch_uv_redshifts, all_log_sch_uv_N_escs[0], s=150, c='darkviolet', edgecolors='black', zorder=5,
            label=(r'$\textsc{thesan-zoom}$-based (This Work)', 'Schechter Luminosity Functions')[include_dpl])
    with open("final_graph_generation/thesan_uv_N_ion_spline_sch.pkl", "wb") as f:
        pickle.dump(all_sch_spline[0], f)
    with open("final_graph_generation/thesan_uv_N_ion_spline_high_sch.pkl", "wb") as f:
        pickle.dump(all_sch_spline_high[0], f)
    with open("final_graph_generation/thesan_uv_N_ion_spline_low_sch.pkl", "wb") as f:
        pickle.dump(all_sch_spline_low[0], f)
    # ax.plot(z_space, all_sch_spline[0](z_space), alpha = 0.8, linestyle='-', c='black', linewidth=2,
    #         zorder=3, label='UV $\dot{N}_\mathrm{ion}$ Spline Fit')

    if not include_stellar_mass and not include_dpl:
        # plot this work's observational derived Schechter UV magnitude N_esc integrations
        ax.errorbar(sch_uv_redshifts, all_log_sch_uv_N_escs[1], yerr=(all_log_sch_uv_err_low[1], all_log_sch_uv_err_high[1]),
                    fmt='none', c='darkcyan', elinewidth=2, capsize=5, alpha=0.8, zorder=4)
        ax.plot(sch_uv_redshifts, all_log_sch_uv_N_escs[1], linestyle='--', c='darkcyan', linewidth=2.5, alpha=0.8, zorder=2)
        ax.scatter(sch_uv_redshifts, all_log_sch_uv_N_escs[1], s=150, c='darkcyan', edgecolors='black', zorder=5,
                label='JWST-based (This Work)')
        with open("final_graph_generation/observational_uv_N_ion_spline_sch.pkl", "wb") as f:
            pickle.dump(all_sch_spline[1], f)
        with open("final_graph_generation/observational_uv_N_ion_spline_high_sch.pkl", "wb") as f:
            pickle.dump(all_sch_spline_high[1], f)
        with open("final_graph_generation/observational_uv_N_ion_spline_low_sch.pkl", "wb") as f:
            pickle.dump(all_sch_spline_low[1], f)
        # ax.plot(z_space, all_sch_spline[1](z_space), alpha = 0.8, linestyle='-', c='black', linewidth=2,
        #         zorder=3, label='UV $\dot{N}_\mathrm{ion}$ Spline Fit')
        # with open(f"final_graph_generation/observational_uv_N_ion_spline_sch_{str(sigma_mag)}.pkl", "wb") as f:
        #     pickle.dump(all_sch_spline[1], f)
    
    if include_dpl:
        # plot this work's Thesan-Zoom derived DPL UV magnitude N_esc integrations
        ax.errorbar(dpl_uv_redshifts, all_log_dpl_uv_N_escs[0], yerr=(all_log_dpl_uv_err_low[0], all_log_dpl_uv_err_high[0]),
                    fmt='none', c='mediumblue', elinewidth=2, capsize=5, alpha=0.8, zorder=4)
        ax.plot(dpl_uv_redshifts, all_log_dpl_uv_N_escs[0], linestyle='--', c='mediumblue', linewidth=2.5, alpha=0.8, zorder=2)
        ax.scatter(dpl_uv_redshifts, all_log_dpl_uv_N_escs[0], s=150, c='mediumblue', edgecolors='black', zorder=5,
                label='DPL Luminoisty Functions')

        # # plot this work's observational derived DPL UV magnitude N_esc integrations
        # ax.errorbar(dpl_uv_redshifts, all_log_dpl_uv_N_escs[1], yerr=(all_log_dpl_uv_err_low[1], all_log_dpl_uv_err_high[1]),
        #             fmt='none', c='crimson', elinewidth=2, capsize=5, alpha=0.8, zorder=4)
        # ax.plot(dpl_uv_redshifts, all_log_dpl_uv_N_escs[1], linestyle='--', c='crimson', linewidth=2.5, alpha=0.8, zorder=2)
        # ax.scatter(dpl_uv_redshifts, all_log_dpl_uv_N_escs[1], s=150, c='crimson', edgecolors='black', zorder=5,
        #         label='$\dot{N}_\mathrm{ion}$, $M_\mathrm{UV}$ DPL LFs')

    if include_stellar_mass:
        # plot this work's Thesan-Zoom derived stellar mass N_esc integrations
        ax.errorbar(m_redshifts, all_log_m_N_escs[0], yerr=(all_log_m_err_low[0], all_log_m_err_high[0]),
                    fmt='none', c='darkorange', elinewidth=2, capsize=5, alpha=0.8, zorder=3)
        ax.plot(m_redshifts, all_log_m_N_escs[0], linestyle='--', c='darkorange', linewidth=2.5, alpha=0.8, zorder=4)
        ax.scatter(m_redshifts, all_log_m_N_escs[0], s=150, c='darkorange',  edgecolors='black', zorder=5,
                label='Schechter Stellar Mass Functions')

        # # plot this work's observational derived stellar mass N_esc integrations
        # ax.errorbar(m_redshifts, all_log_m_N_escs[1], yerr=(all_log_m_err_low[1], all_log_m_err_high[1]),
        #             fmt='none', c='darkorange', elinewidth=2, capsize=5, alpha=0.8, zorder=3)
        # ax.plot(m_redshifts, all_log_m_N_escs[1], linestyle='--', c='darkorange', linewidth=2.5, alpha=0.8, zorder=4)
        # ax.scatter(m_redshifts, all_log_m_N_escs[1], s=150, c='darkorange',  edgecolors='black', zorder=5,
        #         label='$\dot{N}_\mathrm{ion}$, $M_*$ Schechter Functions')

    ax.set_xlabel("$z$")
    ax.set_ylabel("$\mathrm{log}_{10}(\dot{N}_\mathrm{ion} \; [\mathrm{s^{-1} \; cMpc^{-3}}])$")
    ax.yaxis.set_label_coords(-0.075, 0.5)
    ax.set_xlim(z_range)
    ax.set_ylim(((48.9, 51.6), (48.9, 52.1))[include_stellar_mass])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=12, integer=True))
    ax.grid(False)
    # ax.grid(True, alpha=0.8, linestyle='--')
    # ax.set_axisbelow(True)
    # for line in ax.get_xgridlines() + ax.get_ygridlines():
    #     line.set_zorder(0)
    legend = ax.legend(fontsize=20, loc='center left', bbox_to_anchor=(1.025, 0.5), borderaxespad=0)
    frame = legend.get_frame()
    frame.set_edgecolor('black')
    frame.set_boxstyle('Square')
    frame.set_alpha(0.8)
    plt.subplots_adjust(right=0.7)

else:

    plt.style.use('./MNRAS_Style.mplstyle')
    mpl.rcParams.update({'font.size': 20})
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))

    bar_width = 0.25
    offsets = [-bar_width, 0, bar_width]
    uv_labels = ('$' + str(uv_bands[0][1]) + ' \geq M_\mathrm{UV} > ' + str(uv_bands[0][0]) + '$',
                 '$' + str(uv_bands[1][1]) + ' \geq M_\mathrm{UV} > ' + str(uv_bands[1][0]) + '$',
                 '$M_\mathrm{UV} \leq ' + str(uv_bands[2][1]) + '$')
    m_labels = ('$\mathrm{log}_{10}(M_*) < 8$',
                '$8 \leq \mathrm{log}_{10}(M_*) < 10$',
                '$\mathrm{log}_{10}(M_*) \geq 10$')
    thesan_uv_colours = ['#D04ECF', '#7B1FA2', '#4A0D4A']
    obvs_uv_colours = ['#00B7C6', '#00729D', '#004A63']
    m_colours = ['#fdb863', '#e66101', '#b2182b']

    for ax_i in range(len(axes)):
        ax = axes[ax_i]
        split_N_escs = [arr[3:8] for arr in all_split_sch_uv_N_escs[ax_i]]
        redshifts = np.array([5, 6, 7, 8, 10])
        labels = [uv_labels, uv_labels][ax_i]
        colors = [thesan_uv_colours, obvs_uv_colours][ax_i]
        for i in range(3):
            ax.bar(redshifts + offsets[i], split_N_escs[i]/10**50,
                           width=bar_width, label=labels[i], color=colors[i], edgecolor='black')
        ax.set_xlabel('$z$')
        ax.tick_params(axis='x', which='both', bottom=False, top=False)
        ax.set_xticks(redshifts)
        ax.set_ylabel('$\dot{N}_{\mathrm{ion}} \; [10^{50} \; \mathrm{s^{-1} \; cMpc^{-3}}]$')
        ax.yaxis.set_label_coords(-0.05, 0.5)
        ax.set_xlim(4.25, 10.75)
        ax.set_ylim(0, 10)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        ax.grid(False)
        ax.grid(True,  alpha=0.8, axis='y')
        ax.set_axisbelow(True)
        for line in ax.get_xgridlines() + ax.get_ygridlines():
            line.set_zorder(0)
        legend = ax.legend(loc='upper right', bbox_to_anchor=(0.975, 0.975), fontsize=16)
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_boxstyle('Square')
        legend.get_frame().set_alpha(1.0)
        ax.text(0.025, 0.975, [r'$\textsc{thesan-zoom}$-based', 'JWST-based'][ax_i],
                ha='left', va='top', transform=ax.transAxes, fontsize=20)

mpl.rcParams['figure.dpi'] = 500
folder = "final_graph_generation/"
fig.savefig(folder + "report_graphs/report_graph.png", bbox_inches='tight', dpi=500)
plt.show()