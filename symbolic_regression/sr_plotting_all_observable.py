"""
This code plots 15 2D histograms in a 3x5 grid of esc_predicted against esc,
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
c = 0

with open(folder+'esc_sr_test_train.json', 'r') as json_data:
    f_data = json.load(json_data)
    esc_test = np.array(f_data['esc_test'])
    esc_train = np.array(f_data['esc_train'])
    esc_test_pred = np.array(f_data['esc_test_pred'])
    esc_train_pred = np.array(f_data['esc_train_pred'])
    equation = f_data['equation']
    test_data = np.array(f_data['test_data'])
    train_data = np.array(f_data['train_data'])
    keys = f_data['keys']

print(equation)
print(keys)
print(f"Train Mean Absolute Error: {mean_absolute_error(esc_train, esc_train_pred)}")
print(f"Train Mean Squared Error: {mean_squared_error(esc_train, esc_train_pred)}")
print(f"Test Mean Absolute Error: {mean_absolute_error(esc_test, esc_test_pred)}")
print(f"Test Mean Squared Error: {mean_squared_error(esc_test, esc_test_pred)}")

n_vars = len(keys)
mpl.rcParams.update({'font.size': (12, 7)[c]})
if n_vars > 10:
    fig, axes = plt.subplots(3, 5, figsize=((18, 10), (10, 6))[c])
    n_graphs = 15
    
else:
    fig, axes = plt.subplots(2, 5, figsize=((20, 8), (12, 5))[c])
    n_graphs = 10

histograms = False
x = esc_test
y = esc_test_pred
for i in range(n_graphs):
    y_axis = i // 5
    x_axis = i % 5

    if histograms:
        # plots a 2d histogram of x against y where the colour of the bin is dictated by it's mean log_esc
        nbins = 30
        hist, xedges, yedges = np.histogram2d(x, y, bins=nbins, weights=test_data[:,i])
        hist_norm, xedges, yedges = np.histogram2d(x, y, bins=nbins)
        hist = hist / hist_norm
        hist[hist_norm < 3] = np.nan
        hist = hist.T # transposes the histogram for proper orientation
        color_hist = axes[y_axis][x_axis].imshow(hist, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                                    origin='lower', aspect='auto', cmap='viridis', interpolation='nearest')
        plt.colorbar(color_hist, label="$Log_{10}$(" + keys[i] +")", ax=axes[y_axis][x_axis], shrink=(0.7, 0.5)[c])


    else:
        # plots a scatter of x against y where the colour and size of the points is determined by esc
        color_scatter = axes[y_axis][x_axis].scatter(x, y, c=test_data[:,i], cmap='viridis', marker='.', s=1)
        plt.colorbar(color_scatter, label="$Log_{10}$(" + keys[i] +")", ax=axes[y_axis][x_axis], shrink=(0.7, 0.5)[c])

# set the limits of the x and y axes depending on the data to be plotted
if np.any(esc_test < 0):
    x_min, x_max = (-5, 0)
else:
    x_min, x_max = (46, 54)
for axs in axes:
    for ax in axs:
        ax.set_box_aspect(1)
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([x_min, x_max])
        ax.set_xlabel("$Log_{10}(f_{esc})$")
        ax.set_ylabel("$Log_{10}(f_{esc})$ Predicted")
        ax.plot((x_min, x_max), (x_min, x_max), c='black', linewidth=1, alpha=0.5, label='$f_{esc}$ = $f_{esc}$ predicted')


plt.tight_layout()
plt.show()
