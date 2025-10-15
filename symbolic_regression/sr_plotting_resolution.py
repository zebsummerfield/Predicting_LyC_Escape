"""
This code plots 15 2D histograms in a 3x5 grid of f_esc_predicted against f_esc,
where the colour of each bin is dictated by one of the variables in the dataset.
"""

import numpy as np
from matplotlib import pyplot as plt
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import mean_absolute_error, mean_squared_error

folder = "symbolic_regression/"

# if c = 0 then large graphs are plotted, if c = 1 then small graphs are plotted
c = 1

with open(folder+'f_esc_sr_test_train.json', 'r') as json_data:
    f_data = json.load(json_data)
    f_esc_test = np.array(f_data['f_esc_test'])
    f_esc_train = np.array(f_data['f_esc_train'])
    f_esc_test_pred = np.array(f_data['f_esc_test_pred'])
    f_esc_train_pred = np.array(f_data['f_esc_train_pred'])
    equation = f_data['equation']
    test_data = np.array(f_data['test_data'])
    train_data = np.array(f_data['train_data'])
    keys = f_data['keys']
    res_test = f_data['res_test']
    res_train = f_data['res_train']

print(equation)
print(keys)
print(f"Train Mean Absolute Error: {mean_absolute_error(f_esc_train, f_esc_train_pred)}")
print(f"Train Mean Squared Error: {mean_squared_error(f_esc_train, f_esc_train_pred)}")
print(f"Test Mean Absolute Error: {mean_absolute_error(f_esc_test, f_esc_test_pred)}")
print(f"Test Mean Squared Error: {mean_squared_error(f_esc_test, f_esc_test_pred)}")

mpl.rcParams.update({'font.size': (12, 10)[c]})
fig, ax = plt.subplots(1, 1, figsize=((8, 8), (6, 6))[c])

x = f_esc_test
y = f_esc_test_pred
z4_i, z8_i, z16_i = [], [], []
for i in range(len(res_test)):
    if res_test[i] == 'z4':
        z4_i.append(i)
    elif res_test[i] == 'z8':
        z8_i.append(i)
    elif res_test[i] == 'z16':
        z16_i.append(i)
z4_i, z8_i, z16_i = np.array(z4_i), np.array(z8_i), np.array(z16_i)

# plots a scatter of x against y where the colour and size of the points is determined by f_esc
z4_scat = ax.scatter(x[z4_i], y[z4_i], c='r', marker='o', s=1, label='z4')
z8_scat = ax.scatter(x[z8_i], y[z8_i], c='b', marker='^', s=2, label='z8')
z16_scat = ax.scatter(x[z16_i], y[z16_i], c='g', marker='s', s=3, label='z16')

# set the limits of the x and y axes depending on the data to be plotted
if np.any(f_esc_test < 0):
    x_min, x_max = (-5, 0)
else:
    x_min, x_max = (46, 54)
ax.set_box_aspect(1)
ax.set_xlim([x_min, x_max])
ax.set_ylim([x_min, x_max])
ax.set_xlabel("$Log_{10}(f_{esc})$")
ax.set_ylabel("$Log_{10}(f_{esc})$ Predicted")
ax.plot((x_min, x_max), (x_min, x_max), c='black', linewidth=1, alpha=0.5, label='$f_{esc}$ = $f_{esc}$ predicted')

plt.tight_layout()
plt.legend(loc='upper left')
plt.show()
