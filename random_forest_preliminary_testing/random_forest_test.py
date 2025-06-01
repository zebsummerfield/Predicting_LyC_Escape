import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

folder = "random_forest_testing/"
file = folder + 'cat.hdf5'

with h5py.File(file, 'r') as hdf:
    print(hdf.keys())
    print(hdf['f_esc_vir_full'][0:10])

    X = np.transpose(np.array([
        np.array(hdf['stellar_mass_full']), 
        np.array(hdf['sfr_full_10']), 
        np.array(hdf['uv_lum_int_full']) / 10e30, 
        np.array(hdf['ha_lum_int_full']) / 10e30
        ]))
    Y = np.array(hdf['f_esc_vir_full'])
    nans = np.isnan(Y)
    nan_indices = [index for index, b in enumerate(list(nans)) if b == True][::-1]
    for index in nan_indices:
        X = np.delete(X, index, 0)
        Y = np.delete(Y, index, 0)



x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.5, random_state=42)
    
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(x_train, y_train)

y_pred = rf.predict(x_test)
print(y_pred[0:10])

mse = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mse}")

importances = rf.feature_importances_
print(importances)

plt.scatter(y_test, y_pred, s=2)
plt.xlabel("$f_{esc}$ Observed")
plt.ylabel("$f_{esc}$ Predicted")

plt.show()