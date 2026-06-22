import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
import joblib
from functions_old import *

# 0 for f_esc, 1 for n_esc
f_or_n = 1

# True if model is generated to predict for an observational catalogue 
obvs = True
obvs_cat = 'charlotte'  # 'charlotte' or 'lola'

# True for dusty Thesan-Zoom catalogue, False for dust-free catalogue
dusty = True

# True to use galactic radius instead of virial radius
gal = False

folder = "final_rf_model/"
file = ['cat.hdf5', 'cat_dusttestszeb_fdust.hdf5'][dusty]

# loads both the catalogue of galaxies and their variables
with h5py.File(file, 'r') as hdf:

    gal_str = ['', '_gal'][gal]
    dust_str = ['', '_fdust_40'][dusty]

    f_esc = np.array(hdf[f'f_esc{dust_str}_vir_full'])
    n_esc = np.array(hdf[f'Ndot_LyC{dust_str}_vir_full'])
    resolution = np.array([zoom.decode('utf-8') for zoom in hdf['zoomlevel_full']])
    redshift = np.array(hdf['redshift_full'])
    star_mass = np.array(hdf[f'stellar_mass{gal_str}_full'])
    sfr10 = np.array(hdf[f'sfr{gal_str}_full_10'])
    sfr50 = np.array(hdf[f'sfr{gal_str}_full_50'])
    sfr100 = np.array(hdf[f'sfr{gal_str}_full_100'])
    ssfr10 = ssfr_func(sfr10, star_mass)
    ssfr50 = ssfr_func(sfr50, star_mass)
    ssfr100 = ssfr_func(sfr100, star_mass)
    vir_mass = np.array(hdf['M_vir_full'])
    gas_mass = np.array(hdf['gas_mass_full'])
    ha_lum = np.array(hdf[f'ha_lum_obs{gal_str}_full']) * 5
    # gas_met = np.array(hdf['gas_met_full'])
    star_met = np.array(hdf['star_met_full'])
    star_size = np.array(hdf['stellar_size_full'])
    sfr_size = np.array(hdf['sfr_size_full'])
    # ha_size = np.array(hdf['ha_size_obs_full'])
    # uv_size = np.array(hdf['uv_size_obs_full'])
    uv_size = np.array(hdf['uv_size_obs_full'])
    ha_size = np.array(hdf['ha_size_obs_full'])
    sfr10_density = sfr10 / (np.pi * sfr_size**2)
    uv_int_lum = np.array(hdf[f'uv_lum_int{gal_str}{dust_str}_full'])
    uv_obs_lum = np.array(hdf[f'uv_lum_obs{gal_str}{dust_str}_full'])

    # # RHD variables
    # ha_lum = np.array(hdf['ha_lum_obs_full'])
    # ha_size = np.array(hdf['ha_size_obs_full'])
    # uv_int_lum = np.array(hdf['uv_lum_int_fdust_40_full'])
    # uv_obs_lum = np.array(hdf['uv_lum_obs_fdust_40_full'])
    # uv_size = np.array(hdf['uv_size_obs_2d_full'])
    # f_esc = np.array(hdf['f_esc_fdust_40_vir_full'])
    # n_esc = np.array(hdf['Ndot_LyC_fdust_40_vir_full'])

    # # MCRT variables 
    # ha_lum = np.array(hdf['ha_mcrt_lum_obs_full'])
    # ha_size = np.array(hdf['ha_mcrt_size_obs_full'])
    # uv_int_lum = np.array(hdf['uv_lum_int_fdust_40_full'])
    # uv_obs_lum = np.array(hdf['uv_lum_obs_fdust_40_full'])
    # uv_size = np.array(hdf['uv_size_obs_2d_full'])
    # f_esc = np.array(hdf['f_esc_mcrt_fdust_40_vir_full'])
    # n_esc = np.array(hdf['Ndot_LyC_mcrt_fdust_40_vir_full'])

    # fixing gas mass units
    gas_mass = gas_mass / (0.76 / 1.6735575e-24)
    gas_mass = gas_mass / 1.989e33

    random_variable = np.random.uniform(low=0, high=1, size=(len(f_esc)))

    f_esc_vars = np.array([ssfr10, ssfr100, ssfr100, star_mass, gas_mass,
                           vir_mass, star_met, uv_obs_lum, ha_lum, uv_size,
                           ha_size, sfr_size, star_size, sfr10_density, uv_int_lum,
                           (1+redshift), random_variable])
    f_esc_keys = np.array(['offset10', 'ssfr100', 'ssfr10/ssfr100', 'star_mass', 'gas_mass/star_mass',
                           'star_mass/vir_mass', 'star_met', 'uv_mag', 'uv_lum/ha_lum', 'uv_size',
                           'ha_size', 'sfr_size', 'sfr_size/star_size', 'sfr10_density', 'attenuation',
                           '1 + redshift', 'random_variable'])
    n_esc_vars = np.array([sfr10, sfr100, star_mass, gas_mass, vir_mass,
                           star_met, uv_obs_lum, ha_lum, uv_int_lum, (1+redshift),
                           random_variable])
    n_esc_keys = np.array(['sfr10', 'sfr100', 'star_mass','gas_mass/star_mass', 'star_mass/vir_mass',
                           'star_met', 'uv_mag', 'ha_lum', 'attenuation', '1 + redshift',
                           'random_variable'])
    
    # adds a small epsilon to the variables to avoid log(0) errors
    eps_frac = 0.01
    f_epsilons = np.array([min([v for v in var if v !=0])*eps_frac for var in f_esc_vars])
    eps_array = np.zeros(f_esc_vars.shape)
    for i in range(len(eps_array)):
        eps_array[i].fill(f_epsilons[i])
    f_esc_vars = f_esc_vars + eps_array
    n_epsilons = np.array([min([v for v in var if v !=0])*eps_frac for var in n_esc_vars])
    eps_array = np.zeros(n_esc_vars.shape)
    for i in range(len(eps_array)):
        eps_array[i].fill(n_epsilons[i])
    n_esc_vars = n_esc_vars + eps_array

    # post adding epsilon, the variables are changed to match the desired form given by their key
    f_esc_vars[2] = f_esc_vars[0] / f_esc_vars[1]
    f_esc_vars[4] = f_esc_vars[4] / f_esc_vars[3]
    f_esc_vars[5] = f_esc_vars[3] / f_esc_vars[5]
    f_esc_vars[8] = f_esc_vars[7] / f_esc_vars[8]
    f_esc_vars[12] = f_esc_vars[11] / f_esc_vars[12]
    n_esc_vars[3] = n_esc_vars[3] / n_esc_vars[2]
    n_esc_vars[4] = n_esc_vars[2] / n_esc_vars[4]
    
    log_f_esc_vars = np.log10(f_esc_vars).astype('float32')
    log_n_esc_vars = np.log10(n_esc_vars).astype('float32')

    # replaces ssfr10 with offset from the star forming main sequence over 10Myrs
    log_osfms10 =  log_f_esc_vars[0] - sfms_func(np.array([redshift, np.log10(star_mass)]), s[0], b[0], u[0])
    log_f_esc_vars[0] = log_osfms10.astype('float32')

    # replaces the UV luminosities with magnitudes
    lum_to_tenpc = 4 * np.pi * (10 * 3.086e18)**2
    uv_obs_mag = -2.5 * np.log10((n_esc_vars[6]) / lum_to_tenpc) - 48.6
    uv_int_mag = -2.5 * np.log10((n_esc_vars[8]) / lum_to_tenpc) - 48.6
    attenuation = uv_obs_mag - uv_int_mag
    log_f_esc_vars[7] = uv_obs_mag.astype('float32')
    log_f_esc_vars[14] = attenuation.astype('float32')
    log_n_esc_vars[6] = uv_obs_mag.astype('float32')
    log_n_esc_vars[8] = attenuation.astype('float32')

    vars = [f_esc_vars, n_esc_vars][f_or_n]
    log_vars = [log_f_esc_vars, log_n_esc_vars][f_or_n]
    keys = [f_esc_keys, n_esc_keys][f_or_n]

    # selects only the variables that are present in the observational catalogue
    if obvs and obvs_cat == 'lola':
        f_esc_selected_vars = [0, 2, 3, 7, 8, 9, 10, 15, 16]
        n_esc_selected_vars = [0, 1, 2, 6, 7, 9, 10]
    elif obvs and obvs_cat == 'charlotte':
        f_esc_selected_vars = [0, 2, 3, 7, 15, 16]        
        n_esc_selected_vars = [0, 1, 2, 6, 9, 10]
    else:
        f_esc_selected_vars = [0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
        n_esc_selected_vars = [0,1,2,3,4,5,6,7,8,9,10]
    selected_vars = [f_esc_selected_vars, n_esc_selected_vars][f_or_n]
    keys = keys[selected_vars]
    vars = vars[selected_vars]
    log_vars = log_vars[selected_vars]
    print(keys)
    
    # removes any rows that have zero ,unity or infinity for the vars and f_esc
    # ssfr50 is included to remove galaxies that have had no recent star formation
    f_esc[np.isnan(f_esc)] = 0
    n_esc[np.isnan(n_esc)] = 0
    bad_indices = []
    for i in range(len(np.concatenate((log_vars, [f_esc, n_esc], [ssfr50])))):
        b_i = [index for index, val in enumerate(list(np.concatenate((log_vars, [f_esc, n_esc], [ssfr50]))[i]))
                        if (val == 0 or val == 1 or val == np.inf or val== -np.inf)]
        print(f"feature {i+1} bad rows: {len(b_i)}")
        bad_indices += b_i
    b_i = [index for index, zoom in enumerate(hdf['zoomlevel_full']) if zoom.decode('utf-8') != 'z4']
    print(f"zoom level bad rows: {len(b_i)}")
    bad_indices += b_i
    bad_indices = list(set(bad_indices))[::-1]
    f_esc, n_esc = (np.delete(f_esc, bad_indices), np.delete(n_esc, bad_indices))
    log_vars = np.delete(log_vars, bad_indices, axis=1)
    ssfr10, ssfr50, ssfr100 = (np.delete(ssfr10, bad_indices),
                                np.delete(ssfr50, bad_indices),
                                np.delete(ssfr100, bad_indices))
    resolution = np.delete(resolution, bad_indices)
        
    print(f'rows remaining: {len(f_esc)}')
    log_f_esc = np.log10(f_esc).astype('float32')
    log_n_esc = np.log10(n_esc).astype('float32')
    print(f'mean f_esc: {np.mean(f_esc)}')
    print(f'mean n_esc: {np.mean(n_esc)}')

    # each row contains the data for a single galaxy
    X = np.transpose(log_vars)
    Y = [log_f_esc, log_n_esc][f_or_n]

# run random forest 1000 times to get an average on importances and errors
n = 1000
test_mae_list = np.zeros(n)
test_mse_list = np.zeros(n)
train_mae_list = np.zeros(n)
train_mse_list = np.zeros(n)
importances_list = np.zeros(shape=(n, len(keys)))
for i in range(n):

    x_train, x_test, y_train, y_test, res_train, res_test = train_test_split(
        X, Y, resolution, test_size=0.25)
        
    # random forest training with the test data
    rf = RandomForestRegressor(# random_state=i,
                               n_estimators=210, n_jobs=-1, min_samples_leaf=50,
                               max_features='sqrt', criterion='squared_error')
    rf.fit(x_train, y_train)
    y_test_pred = rf.predict(x_test)
    y_train_pred =  rf.predict(x_train)

    # calculate errors on the test and train data
    test_mae_list[i] = mean_absolute_error(y_test, y_test_pred)
    test_mse_list[i] = mean_squared_error(y_test, y_test_pred)
    train_mae_list[i] = mean_absolute_error(y_train, y_train_pred)
    train_mse_list[i] = mean_squared_error(y_train, y_train_pred)

    # gives the importance weightings for each variable in the RF
    importances_list[i] = rf.feature_importances_

    if i==0 or test_mae_list[i] <= min([mae for mae in test_mae_list if mae != 0]):
        best_rf = rf
        best_rf_data = {'keys': keys.tolist(),
                   'f_esc_test': y_test.tolist(), 
                   'f_esc_train': y_train.tolist(), 
                   'f_esc_test_pred': y_test_pred.tolist(), 
                   'f_esc_train_pred': y_train_pred.tolist(),
                   'importances': rf.feature_importances_.tolist(),
                    'res_train': res_train.tolist(),
                    'res_test': res_test.tolist()}
        best_rf_index = i
    
    print(f"Run {i+1}, train size: {len(x_train)}, MAE: {test_mae_list[i]}, MSE: {test_mse_list[i]}")

print(f"test size: {len(x_test)}")
print(f"Mean Test Mean Absolute Error: {np.mean(test_mae_list)}")
print(f"Mean Test Mean Squared Error: {np.mean(test_mse_list)}")
print(f"Mean Train Mean Absolute Error: {np.mean(train_mae_list)}")
print(f"Mean Train Mean Squared Error: {np.mean(train_mse_list)}")
print(f"Best Test Mean Absolute Error: {test_mae_list[best_rf_index]}")
print(f"Best Test Mean Squared Error: {test_mse_list[best_rf_index]}")
print(f"Best Train Mean Absolute Error: {train_mae_list[best_rf_index]}")
print(f"Best Train Mean Squared Error: {train_mse_list[best_rf_index]}")
for index, v in enumerate(keys):
    print(f'{v}: {np.mean(importances_list[:,index])}')

f_or_n_str = ['f_esc', 'n_esc'][f_or_n]
obvs_str = ['final', 'observational'][obvs] + ['', f'_{obvs_cat}'][obvs]
dust_str = ['', '_dusty'][dusty]
testing = ''  # '_test' or ''
# saves the best rf data to a json file
with open(folder+f'{f_or_n_str}_rf_{obvs_str}{dust_str}_test_train{testing}.json', 'w') as f:
    best_rf_data['mean_importances'] = np.mean(importances_list, axis=0).tolist()
    best_rf_data['std_importances'] = np.std(importances_list, axis=0).tolist()
    json.dump(best_rf_data, f)
# saves the rf model to a pickle file
if obvs:
    joblib.dump(best_rf, folder+f'{f_or_n_str}_rf_{obvs_str}{dust_str}_model{testing}.pkl')

mpl.rcParams.update({'font.size': 16})
fig, ax = plt.subplots(figsize=(12,6))

min_importances = np.min(importances_list, axis=0)
max_importances = np.max(importances_list, axis=0)
mean_importances = np.mean(importances_list, axis=0)
std_importances = np.std(importances_list, axis=0)
sorted_indices = np.argsort(mean_importances)[::-1]

ax.bar(keys[sorted_indices], mean_importances[sorted_indices], 
       yerr=std_importances[sorted_indices], capsize=5, edgecolor='black')
ax.set_ylabel('Importance')
ax.set_xticklabels(keys[sorted_indices], rotation='vertical')

fig.tight_layout()
plt.show()