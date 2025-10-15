"""
This code runs symbolic regression on a test dataset to find an equation for f_esc.
"""

from sr_functions import prepare_data_sr
import numpy as np
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import json
import matplotlib.pyplot as plt

folder = "symbolic_regression/"
file = 'cat.hdf5'

# 0 for f_esc, 1 for n_esc
f_or_n = 0
# True if model is generated to predict for an observational catalogue 
obvs = False

with open(folder+"pysr_model.pkl", "rb") as f:
    model = pickle.load(f)
    model.set_params(extra_sympy_mappings={"pow10": lambda x: 10**x})

with open(folder+'f_esc_sr_test_train.json', 'r') as json_data:
    f_data = json.load(json_data)
    x_test = np.array(f_data['test_data'])
    x_train = np.array(f_data['train_data'])
    y_test = np.array(f_data['f_esc_test'])
    y_train = np.array(f_data['f_esc_train'])
    keys = f_data['keys']
    res_test = f_data['res_test']
    res_train = f_data['res_train']
    
mse = np.array(model.equations_.loss)[::-1]
complexity = np.array(model.equations_.complexity)
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
           'f_esc_test': y_test.tolist(), 
           'f_esc_train': y_train.tolist(), 
           'f_esc_test_pred': y_test_pred.tolist(), 
           'f_esc_train_pred': y_train_pred.tolist(),
           'equation': (str(model.sympy()), str(model.sympy(knee_index)))[save_knee],
           'test_data': x_test.tolist(),
           'train_data': x_train.tolist(),
           'res_test': res_test,
           'res_train': res_train}
with open(folder+'f_esc_sr_test_train.json', 'w') as json_file:
    json.dump(sr_data, json_file)

plt.plot(complexity, mse, marker='o')
plt.axvline(complexity[knee_index], color='red', linestyle='dashdot')
plt.show()
