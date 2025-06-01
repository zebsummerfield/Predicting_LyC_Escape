import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
from scipy import stats
import joblib
from final_model.functions_old import *

folder = "final_model/"
file = 'cat.hdf5'

# 0 for f_esc, 1 for n_esc
f_or_n = 1

# True if model is generated to predict for an observational catalogue 
obvs = True

# loads both the catalogue of galaxies and their variables
with h5py.File(file, 'r') as hdf:

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
    print(np.mean(uv_lum))

    random_variable = np.random.uniform(low=0, high=1, size=(len(f_esc)))

    f_esc_vars = np.array([ssfr10, ssfr100, star_mass, gas_mass, vir_mass, star_met,
                           uv_lum, ha_lum, uv_size, ha_size,
                           sfr_size, star_size, sfr10_density, (1+redshift), random_variable,])
    f_esc_keys = np.array(['offset10', 'ssfr10/ssfr100', 'star_mass', 'gas_mass/star_mass', 'star_mass/vir_mass',
                           'gas_met', 'uv_mag', 'uv_lum/ha_lum', 'uv_size', 'ha_size',
                           'sfr_size', 'sfr_size/star_size', 'sfr10_density', '1 + redshift', 'random_variable'])
    n_esc_vars = np.array([sfr10, sfr100, star_mass, gas_mass, vir_mass, star_met, 
                           uv_lum, ha_lum, (1+redshift), random_variable])
    n_esc_keys = np.array(['sfr10', 'sfr100', 'star_mass','gas_mass/star_mass', 'star_mass/vir_mass', 'gas_met', 
                           'uv_mag', 'ha_mag', '1 + redshift', 'random_variable'])
    
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
    f_esc_vars[1] = f_esc_vars[0] / f_esc_vars[1]
    f_esc_vars[3] = f_esc_vars[3] / f_esc_vars[2]
    f_esc_vars[4] = f_esc_vars[2] / f_esc_vars[4]
    f_esc_vars[7] = f_esc_vars[6] / f_esc_vars[7]
    f_esc_vars[11] = f_esc_vars[10] / f_esc_vars[11]
    n_esc_vars[3] = n_esc_vars[3] / n_esc_vars[2]
    n_esc_vars[4] = n_esc_vars[2] / n_esc_vars[4]

    # replaces the sizes with 2d projections to match the observational catalogue
    if obvs:
        f_esc_vars[8] = uv_projection
        f_esc_vars[9] = ha_projection
    
    log_f_esc_vars = np.log10(f_esc_vars).astype('float32')
    log_n_esc_vars = np.log10(n_esc_vars).astype('float32')

    # replaces ssfr10 with offset from the star forming main sequence over 10Myrs
    log_osfms10 =  log_f_esc_vars[0] - sfms_func(np.array([redshift, np.log10(star_mass)]), s[0], b[0], u[0])
    log_f_esc_vars[0] = log_osfms10.astype('float32')

    # replaces the luminosities with magnitudes
    lum_to_tenpc = 4 * np.pi * (10 * 3.086e18)**2
    uv_mag = -2.5 * np.log10((n_esc_vars[6]) / lum_to_tenpc) - 48.6
    ha_mag = -2.5 * np.log10((n_esc_vars[7]) / lum_to_tenpc) - 48.6
    log_f_esc_vars[6] = uv_mag.astype('float32')
    log_n_esc_vars[6] = uv_mag.astype('float32')
    log_n_esc_vars[7] = ha_mag.astype('float32')

    vars = [f_esc_vars, n_esc_vars][f_or_n]
    log_vars = [log_f_esc_vars, log_n_esc_vars][f_or_n]
    keys = [f_esc_keys, n_esc_keys][f_or_n]

    # selects only the variables that are present in the observational catalogue
    f_esc_observational_vars = [0, 1, 2, 6, 7, 8, 9, 13, 14]
    n_esc_observational_vars = [0, 1, 2, 6, 7, 8, 9]
    selected_vars = [f_esc_observational_vars, n_esc_observational_vars][f_or_n]
    if obvs:
        keys = keys[selected_vars]
        vars = vars[selected_vars]
        log_vars = log_vars[selected_vars]
    print(keys)
    
    # removes any rows that have zero ,unity or infinity for the vars and f_esc
    # ssfr50 is included to remove galaxies that have had no recent star formation
    f_esc[np.isnan(f_esc)] = 0
    for i in range(len(np.concatenate((log_vars, [f_esc], [ssfr50])))):
        nan_indices = [index for index, val in enumerate(list(np.concatenate((log_vars, [f_esc], [ssfr50]))[i]))
                       if (val == 0 or val == 1 or val == np.inf or val== -np.inf)][::-1]
        print(f"rows deleted: {len(nan_indices)}")
        f_esc, n_esc = (np.delete(f_esc, nan_indices), np.delete(n_esc, nan_indices))
        log_vars = np.delete(log_vars, nan_indices, axis=1)
        ssfr10, ssfr50, ssfr100 = (np.delete(ssfr10, nan_indices), 
                                   np.delete(ssfr50, nan_indices),
                                   np.delete(ssfr100, nan_indices))
        
    print(f'rows remaining: {len(f_esc)}')
    log_f_esc = np.log10(f_esc).astype('float32')
    log_n_esc = np.log10(n_esc).astype('float32')
    print(np.mean(f_esc))
    print(np.mean(n_esc))

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

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.25, random_state=i)

    # carries out rejection sampling to cap the density of the f_esc distribution
    height_fraction = 0.35
    mean = np.mean(y_train)
    std = np.std(y_train)
    density_threshold = height_fraction * stats.norm.pdf(mean, mean, std)
    accepted_samples, rejected_samples = ([], [])
    for index in range(len(y_train)):
        density = stats.norm.pdf(y_train[index], mean, std)
        if np.random.random() < density_threshold/density:
            accepted_samples.append(index)
        else:
            rejected_samples.append(index)
    x_train = x_train[accepted_samples]
    y_train = y_train[accepted_samples]
    print(f"Run {i+1}, train size: {len(x_train)}")
        
    # random forest training with the test data
    rf = RandomForestRegressor(n_estimators=140, random_state=i, n_jobs=-1,
                               min_samples_leaf=50, max_features='sqrt', 
                               criterion='squared_error')
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
        best_rf = {'keys': keys.tolist(),
                   'f_esc_test': y_test.tolist(), 
                   'f_esc_train': y_train.tolist(), 
                   'f_esc_test_pred': y_test_pred.tolist(), 
                   'f_esc_train_pred': y_train_pred.tolist(),
                   'importances': rf.feature_importances_.tolist()}
        best_rf_index = i

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
obvs_str = ['final', 'observational'][obvs]
# saves the best rf data to a json file
with open(folder+f'{f_or_n_str}_rf_{obvs_str}_test_train.json', 'w') as f:
    best_rf['std_importances'] = np.std(importances_list, axis=0).tolist()
    json.dump(best_rf, f)
# saves the rf model to a pickle file
if obvs:
    joblib.dump(rf, folder+f'{f_or_n_str}_rf_{obvs_str}_model.pkl')

mpl.rcParams.update({'font.size': 16})
fig, ax = plt.subplots(figsize=(12,6))

# min_importances = np.min(importances_list, axis=0)
# max_importances = np.max(importances_list, axis=0)
# mean_importances = np.mean(importances_list, axis=0)
std_importances = np.std(importances_list, axis=0)
sorted_indices = np.argsort(importances_list[best_rf_index])[::-1]

ax.bar(keys[sorted_indices], importances_list[best_rf_index][sorted_indices], 
       yerr=std_importances[sorted_indices], capsize=5, edgecolor='black')
ax.set_ylabel('Importance')
ax.set_xticklabels(keys[sorted_indices], rotation='vertical')

fig.tight_layout()
plt.show()