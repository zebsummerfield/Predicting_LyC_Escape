"""
This code runs symbolic regression on a test dataset to find an equation for f_esc.
"""

from sr_functions import *
import numpy as np
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import json
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import sympy as sp

folder = "symbolic_regression/"
file = 'cat.hdf5'

# Opening model and data files
with open(folder+"pysr_model.pkl", "rb") as f:
    model = pickle.load(f)
    model.set_params(extra_sympy_mappings={"pow10": lambda x: 10**x})
with open(folder+'esc_sr_test_train.json', 'r') as json_data:
    f_data = json.load(json_data)
    x_test = np.array(f_data['test_data'])
    x_train = np.array(f_data['train_data'])
    y_test = np.array(f_data['esc_test'])
    y_train = np.array(f_data['esc_train'])
    keys = f_data['keys']
    res_train = f_data['res_train']
    res_test = f_data['res_test']

mse = np.array(model.equations_.loss)
complexity = np.array(model.equations_.complexity)
window_length = 9
polyorder = 2
curvature_approach = False

# Finds the knee point in the model's loss vs complexity graph
if curvature_approach:
    # Normalize and smooth
    x = (complexity - complexity.min()) / (complexity.max() - complexity.min())
    y = (mse - mse.min()) / (mse.max() - mse.min())
    y_smooth = savgol_filter(y, window_length=window_length, polyorder=polyorder)
    dy = np.gradient(y_smooth, x)
    d2y = np.gradient(dy, x)
    # Find where curvature starts increasing
    threshold = 0.2 * np.max(d2y)  # first point where curvature > 20% of max curvature
    knee_index = np.where(d2y > threshold)[0][-1]
    #plt.plot(complexity, dy/np.max(np.abs(dy)), linestyle='dashed', label='Normalized Curvature')
    # plt.plot(complexity, d2y/np.max(np.abs(d2y)), linestyle='dashed', label='Normalized Curvature')

else:
    mse = mse[::-1]
    knee_index = -1
    for index in range(len(mse)):
        try:
            if mse[index+1] - mse[index] > 0.01:
                knee_index = len(mse) - (index+1)
                break
        except:
            break
    mse = mse[::-1]

print(model)
print(f"Most Accurate - {complexity[-1]}: {model.sympy()}, MSE: {mse[-1]}")
print(f"Optimal - {complexity[knee_index]} : {model.sympy(knee_index)}, MSE: {mse[knee_index]}")
expr_no_log = simplify_model(model.sympy(knee_index))
latex_str = sp.latex(expr_no_log)
print(f"Optimal Exponentiated: f_esc = {expr_no_log}")

# if save_knee == True then the knee index is used to make predictions
save_knee = True
if save_knee:
    y_train_pred = model.predict(x_train, knee_index)
    y_test_pred = model.predict(x_test, knee_index)
else:
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

# saving the data to a json file
sr_data = {'keys': keys,
           'esc_test': y_test.tolist(), 
           'esc_train': y_train.tolist(), 
           'esc_test_pred': y_test_pred.tolist(), 
           'esc_train_pred': y_train_pred.tolist(),
           'equation': (str(model.sympy()), str(model.sympy(knee_index)))[save_knee],
           'test_data': x_test.tolist(),
           'train_data': x_train.tolist(),
           'res_train': res_train,
           'res_test': res_test}
with open(folder+'esc_sr_test_train.json', 'w') as json_file:
    json.dump(sr_data, json_file)

plt.figure(figsize=(10,10))
plt.plot(complexity, mse, marker='o', label='MSE')
plt.plot(complexity, savgol_filter(mse, window_length=window_length, polyorder=polyorder), label='Smoothed MSE', alpha=0.5)
plt.axvline(complexity[knee_index], color='red', linestyle='dashdot', label='Knee Point')
plt.ylim(0.3,0.8)

plt.text(
    0.975, 0.75,              # top-right corner in axes coords
    "$ f_{esc} = " + latex_str + "$",
    transform=plt.gca().transAxes,
    verticalalignment='top',
    horizontalalignment='right',
    fontsize=16,
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='black')
)
plt.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='black', framealpha=0.8, fontsize=16)
plt.show()

