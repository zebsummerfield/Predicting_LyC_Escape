import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json

folder = "random_forest_testing/"
file = folder + 'cat.hdf5'

# loads the catalogue of galaxies and their variables
with h5py.File(file, 'r') as hdf:
    # print(hdf.keys())
    # isolates all the numeric variables which are not related to f_esc
    keys = [key for key in hdf.keys() if type(hdf[key][0]) in [np.float64, np.float32] and 
            key not in ['f_esc_vir_full', 'f_esc_gal_full']]
    print(keys)

    # creates a 2d array of the log of all the isolated variables,
    # casts them to float32 as required for the RF,
    # each row contains the data for a single galaxy
    X = np.transpose(np.array(
        [np.log10(np.array(hdf[key])) for key in keys]
        )).astype('float32')
    Y = np.array(hdf['f_esc_vir_full'])

    # removes any rows that have nan for f_esc
    nans = np.isnan(Y)
    nan_indices = [index for index, b in enumerate(list(nans)) if b == True][::-1]
    for index in nan_indices:
        X = np.delete(X, index, 0)
        Y = np.delete(Y, index, 0)
    # set infinities to be a negligibally small number
    X = np.nan_to_num(X, -10, neginf=-10)


# random forest training and testing with both the test and train data
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.5, random_state=2)
rf = RandomForestRegressor(n_estimators=100, random_state=1)
rf.fit(x_train, y_train)
y_test_pred = rf.predict(x_test)
y_train_pred =  rf.predict(x_train)

# calculate errors on the test and train data
test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
print(f"Test Mean Absolute Error: {test_mae}")
print(f"Test Mean Squared Error: {test_mse}")
train_mae = mean_absolute_error(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
print(f"Train Mean Absolute Error: {train_mae}")
print(f"Train Mean Squared Error: {train_mse}")

# gives the importance weightings for each variable in the RF
importances = rf.feature_importances_
for index, k in enumerate(keys):
    print(f'{k}: {importances[index]}')

# saves the data to a json file
with open(folder+'all_test_train.json', 'w') as f:
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