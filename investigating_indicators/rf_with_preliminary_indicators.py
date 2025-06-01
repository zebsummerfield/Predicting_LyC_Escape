import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
from investigating_indicators.functions_old import *

folder = "investigating_indicators/"
file = 'cat.hdf5'

# 0 for f_esc, 1 for n_esc
f_or_n = 0

# loads the catalogue of galaxies and their variables
with h5py.File(file, 'r') as hdf:

    f_esc = np.array(hdf['f_esc_vir_full']).astype('float32')
    n_esc = np.array(hdf['Ndot_LyC_vir_full'])
    print(len(f_esc))
    print(hdf['id_full'][0:10])
    redshift = np.array(hdf['redshift_full'])
    star_mass = np.array(hdf['stellar_mass_full'])
    ssfr10 = ssfr_func(hdf['sfr_full_10'], star_mass)
    ssfr50 = ssfr_func(hdf['sfr_full_50'], star_mass)
    ssfr100 = ssfr_func(hdf['sfr_full_100'], star_mass)
    vir_mass = np.array(hdf['M_vir_full'])
    gas_mass = np.array(hdf['gas_mass_full'])
    ha_lum = np.array(hdf['ha_lum_int_full'])
    uv_lum = np.array(hdf['uv_lum_int_full'])
    gas_met = np.array(hdf['gas_met_full'])
    star_met = np.array(hdf['star_met_full'])
    star_size = np.array(hdf['stellar_size_full'])
    sfr_size = np.array(hdf['sfr_size_full'])
    uv_size = np.array(hdf['uv_size_int_full'])
    ha_size = np.array(hdf['ha_size_int_full'])
    uv_projection = np.array(hdf['uv_size_int_2d_full'])
    ha_projection = np.array(hdf['ha_size_int_2d_full'])
    uv_lum_density = uv_lum / (uv_projection**2)
    uv_obs = np.array(hdf['uv_lum_obs_full'])

    random_variable = np.random.uniform(low=0, high=1, size=(len(f_esc)))

    f_esc_vars = np.array([ssfr10, ssfr100, star_mass, gas_mass, gas_met,
                     uv_lum, ha_lum, uv_size, ha_size, uv_lum_density,
                     sfr_size, star_size, (1+redshift), random_variable])
    f_esc_keys = np.array(['offset10', 'ssfr10/ssfr100', 'star_mass', 'gas_mass/star_mass', 'gas_met',
                     'uv_lum', 'uv_lum/ha_lum', 'uv_size', 'ha_size', 'uv_lum_density', 
                     'sfr_size', 'sfr_size/star_size', '1 + redshift', 'random_variable'])
    n_esc_vars = np.array([ssfr100, star_mass, vir_mass, gas_mass, star_mass, gas_met, 
                           uv_lum, ha_lum, uv_size, uv_lum_density, 
                           star_size, (1+redshift), random_variable])
    n_esc_keys = np.array(['ssfr100', 'star_mass', 'vir_mass', 'gas_mass', 'gas_mass/star_mass', 'gas_met', 
                           'uv_lum', 'ha_lum', 'uv_size', 'uv_lum_density', 
                           'star_size', '1 + redshift', 'random_variable'])
    
    # sets 0 values to nan to avoid log(0) errors
    # f_esc_vars[f_esc_vars == 0] = np.nan
    # n_esc_vars[n_esc_vars == 0] = np.nan
    
    # adds a small epsilon to the variables to avoid log(0) errors
    f_epsilons = np.array([min([v for v in var if v !=0])/10 for var in f_esc_vars])
    eps_array = np.zeros(f_esc_vars.shape)
    for i in range(len(eps_array)):
        eps_array[i].fill(f_epsilons[i])
    f_esc_vars = f_esc_vars + eps_array
    n_epsilons = np.array([min([v for v in var if v !=0])/10 for var in n_esc_vars])
    eps_array = np.zeros(n_esc_vars.shape)
    for i in range(len(eps_array)):
        eps_array[i].fill(n_epsilons[i])
    n_esc_vars = n_esc_vars + eps_array

    f_esc_vars[1] = f_esc_vars[0] / f_esc_vars[1]
    f_esc_vars[3] = f_esc_vars[3] / f_esc_vars[2]
    f_esc_vars[6] = f_esc_vars[5] / f_esc_vars[6]
    f_esc_vars[11] = f_esc_vars[10] / f_esc_vars[11]
    n_esc_vars[4] = n_esc_vars[3] / n_esc_vars[1]

    vars = [f_esc_vars, n_esc_vars][f_or_n]
    keys = [f_esc_keys, n_esc_keys][f_or_n]

    phot_vars = [0,1,2,5,13,14]
    grism_vars = [0,1,2,5,6,7,13,14]
    grism_morpho_vars = [0,1,2,5,6,7,8,9,10,11,12,13,14]
    selected_vars = [0,1]
    # keys = keys[selected_vars]
    # vars = vars[selected_vars]
    print(keys)

    # removes any rows that have zero ,unity or infinity for the vars and f_esc
    # ssfr100 is included to remove galaxies that have had no recent star formation
    f_esc[np.isnan(f_esc)] = 0
    for i in range(len(np.concatenate((vars, [f_esc], [ssfr50])))):
        nan_indices = [index for index, val in enumerate(list(np.concatenate((vars, [f_esc], [ssfr50]))[i]))
                       if (val == 0 or val == 1 or val == np.inf or val== -np.inf)][::-1]
        print(f"rows deleted: {len(nan_indices)}")
        f_esc, n_esc = (np.delete(f_esc, nan_indices), np.delete(n_esc, nan_indices))
        vars = np.delete(vars, nan_indices, axis=1)
        redshift, star_mass = (np.delete(redshift, nan_indices), np.delete(star_mass, nan_indices))
        ssfr10, ssfr50, ssfr100 = (np.delete(ssfr10, nan_indices),
                                   np.delete(ssfr50, nan_indices),
                                   np.delete(ssfr100, nan_indices))
    
    # removes any rows that have f_esc < threshold
    threshold = 0
    small_indices = [index for index, val in enumerate(list(f_esc)) if val < threshold][::-1]
    print(f"small f_esc rows changed: {len(small_indices)}")
    # f_esc[small_indices] = threshold
    f_esc, n_esc = (np.delete(f_esc, small_indices), np.delete(n_esc, small_indices))
    vars = np.delete(vars, small_indices, axis=1)
    redshift, star_mass = (np.delete(redshift, small_indices), np.delete(star_mass, small_indices))

    print(f'rows remaining: {len(f_esc)}')

    log_vars = np.log10(vars).astype('float32')
    # replaces ssfr10 with Offset from the star forming main sequence over 10Myrs
    log_sfms10 =  log_vars[0] - sfms_func(np.array([redshift, np.log10(star_mass)]), s[0], b[0], u[0])
    log_vars[0] = log_sfms10.astype('float32')
    log_f_esc = np.log10(f_esc).astype('float32')
    log_n_esc = np.log10(n_esc).astype('float32')

    # each row contains the data for a single galaxy
    X = np.transpose(log_vars)
    Y = [log_f_esc, log_n_esc][f_or_n]


# run random forest 100 times to get an average on importances and errors
n = 10
test_mae_list = np.zeros(n)
test_mse_list = np.zeros(n)
train_mae_list = np.zeros(n)
train_mse_list = np.zeros(n)
importances_list = np.zeros(shape=(n, len(keys)))
for i in range(n):
    print(f"Run {i+1}")
    # random forest training and testing with both the test and train data
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.5, random_state=i)
    rf = RandomForestRegressor(n_estimators=100, random_state=i, n_jobs=-1,
                               min_samples_leaf=65, max_features='sqrt', 
                               criterion='friedman_mse')
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

print(f"Test Mean Absolute Error: {np.mean(test_mae_list)}")
print(f"Test Mean Squared Error: {np.mean(test_mse_list)}")
print(f"Train Mean Absolute Error: {np.mean(train_mae_list)}")
print(f"Train Mean Squared Error: {np.mean(train_mse_list)}")
for index, v in enumerate(keys):
    print(f'{v}: {np.mean(importances_list[:,index])}')

# saves the data to a json file
with open(folder+'rf_indicators.json', 'w') as f:
    json.dump({'f_esc_test': y_test.tolist(), 
               'f_esc_train': y_train.tolist(), 
               'f_esc_test_pred': y_test_pred.tolist(), 
               'f_esc_train_pred': y_train_pred.tolist()}, f)


mpl.rcParams.update({'font.size': 16})
fig, ax = plt.subplots(figsize=(12,6))

mean_importances = np.mean(importances_list, axis=0)
std_importances = np.std(importances_list, axis=0)
sorted_indices = np.argsort(mean_importances)[::-1]

ax.bar(keys[sorted_indices], mean_importances[sorted_indices], 
       yerr=std_importances[sorted_indices], capsize=5, edgecolor='black')
ax.set_ylabel('Importance')
ax.set_xticklabels(keys[sorted_indices], rotation='vertical')

fig.tight_layout()
plt.show()