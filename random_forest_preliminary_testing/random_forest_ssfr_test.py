import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json

def sfms_func(xdata, s, b, u):
    redshift = xdata[0,:]
    stellar_mass = 10**xdata[1,:]
    return np.log10(s * (stellar_mass / 1e10)**b * (1 + redshift)**u)

def ssfr_func(sfr, mass):
    return np.array(sfr) / np.array(mass) * 1e9

s = [0.033, 0.067]
b = [0.041, 0.042]
u = [2.64, 2.57]

folder = "random_forest_testing/"
file = folder + 'cat.hdf5'


# loads the catalogue of galaxies and their variables
with h5py.File(file, 'r') as hdf:
    print(hdf.keys())
    print(hdf['f_esc_vir_full'][0:10])

    f_esc = np.array(hdf['f_esc_vir_full']).astype('float32')
    redshift = np.array(hdf['redshift_full']).astype('float32')
    mass = np.array(hdf['stellar_mass_full'])
    ssfr10 = ssfr_func(hdf['sfr_full_10'], mass)
    ssfr50 = ssfr_func(hdf['sfr_full_50'], mass)
    ssfr100 = ssfr_func(hdf['sfr_full_100'], mass)
    vars = np.array([ssfr10, ssfr50, ssfr100, redshift, mass])

    # removes any rows that have nan for f_esc
    nans = np.isnan(f_esc)
    nan_indices = [index for index, b in enumerate(list(nans)) if b == True][::-1]
    print(len(nan_indices))
    f_esc = np.delete(f_esc, nan_indices, 0)
    vars = np.delete(vars, nan_indices, axis=1)
    
    # removes any rows that have zero for any vars
    for v in range(len(vars)):
        zero_indices = [index for index, val in enumerate(list(vars[v])) if val == 0][::-1]
        print(len(zero_indices))
        f_esc = np.delete(f_esc, zero_indices, 0)
        vars = np.delete(vars, zero_indices, axis=1)

    # takes the log of all the variables and casts them to float32 as required for the RF
    log_vars = np.log10(vars).astype('float32')
    print(log_vars.shape)

    # calculates the offset from the star-forming main sequence
    sfms10 =  log_vars[0] - sfms_func(np.array([vars[3], log_vars[4]]), s[0], b[0], u[0])
    sfms100 =  log_vars[2] - sfms_func(np.array([vars[3], log_vars[4]]), s[1], b[1], u[1])

    # each row contains the data for a single galaxy
    keys = ['ssfr10', 'ssfr50', 'ssfr100']
    X = np.transpose(log_vars[0:3])
    Y = f_esc


# run random forest 100 times to get an average on importances and errors
n = 2
test_mae_list = np.zeros(n)
test_mse_list = np.zeros(n)
train_mae_list = np.zeros(n)
train_mse_list = np.zeros(n)
importances_list = np.zeros(shape=(n, len(keys)))
for i in range(n):
    # random forest training and testing with both the test and train data
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.5, random_state=i)
    rf = RandomForestRegressor(n_estimators=100, random_state=i)
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
with open(folder+'ssfr_test_train.json', 'w') as f:
    json.dump({'f_esc_test': y_test.tolist(), 
               'f_esc_train': y_train.tolist(), 
               'f_esc_test_pred': y_test_pred.tolist(), 
               'f_esc_train_pred': y_train_pred.tolist()}, f)


fig, axes = plt.subplots(1, 2, figsize=(12,6))

# plots the predicted f_escs against the test data
axes[0].scatter(y_test, y_test_pred, s=2, c='b')
axes[0].set_title('Test Data')
axes[0].set_xlabel("$f_{esc}$ Observed")
axes[0].set_ylabel("$f_{esc}$ Predicted")
axes[0].set_xlim([0, 1])
axes[0].set_ylim([0, 1])

# plots the predicted f_escs for the train data to check for overfitting
axes[1].scatter(y_train, y_train_pred, s=2, c='g')
axes[1].set_title('Train Data')
axes[1].set_xlabel("$f_{esc}$ Observed")
axes[1].set_ylabel("$f_{esc}$ Predicted")
axes[1].set_xlim([0, 1])
axes[1].set_ylim([0, 1])

fig.tight_layout()
plt.show()