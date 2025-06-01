import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import matplotlib as mpl


# if c = 0 then small graphs are plotted, if c = 1 then large graphs are plotted
c = 0

# if invert == 1 then f_esc is on the x-axis, if invert == -1 then f_esc_predicted is on the x-axis
invert = -1

folder = "final_model/"
file = folder + 'f_esc_rf_final_test_train.json'
with open(file, 'r') as json_data:
    f_data = json.load(json_data)

test = np.array(f_data['f_esc_test'])
test_pred = np.array(f_data['f_esc_test_pred'])
train = np.array(f_data['f_esc_train'])
train_pred = np.array(f_data['f_esc_train_pred'])

# calculate errors on the test and train data
test_mae = mean_absolute_error(test, test_pred)
test_mse = root_mean_squared_error(test, test_pred)
print(f"Test Mean Absolute Error: {test_mae}")
print(f"Test Mean Squared Error: {test_mse}")
train_mae = mean_absolute_error(train, train_pred)
train_mse = root_mean_squared_error(train, train_pred)
print(f"Train Mean Absolute Error: {train_mae}")
print(f"Train Mean Squared Error: {train_mse}")

print(f"maximum test prediction: {max(test_pred)}")
print(f"minimum test prediction: {min(test_pred)}")


mpl.rcParams.update({'font.size': (16, 10)[c]})
fig, axes = plt.subplots(1, 2, figsize=((16, 8), (12, 6))[c])

# set the limits of the x and y axes depending on the data to be plotted
if np.any(test < 0):
    x_min, x_max = (-5, 0)
elif np.any(test > 1):
    x_min, x_max = (46, 54)
else:
    x_min, x_max = (0, 0.5)
for i in range(len(axes)):
    axes[i].set_xlim([x_min, x_max])
    axes[i].set_ylim([x_min, x_max])
nbins = 100
x_bins = np.linspace(x_min, x_max, nbins+1)

# If invert == 1 then f_esc is on the x-axis, if invert == -1 then f_esc_predicted is on the x-axis
test_X, test_Y = [test, test_pred][::invert]
train_X, train_Y = [train_pred, train][::invert]
test_X_name, test_Y_name = ["$Log_{10}$($f_{esc}$)", "$Log_{10}$($f_{esc}$ Predicted)"][::invert]
train_X_name, train_Y_name = ["$Log_{10}$($f_{esc}$ Predicted)", "$Log_{10}$($f_{esc}$)"][::invert]


# plots a 2d histogram of test X data against test Y data,
# where the number of galaxies in a bin dictates it's colour
hist, xedges, yedges = np.histogram2d(test_X, test_Y, bins=[x_bins, x_bins])
hist = hist.T
hist = np.ma.masked_where(hist == 0, hist)
cmap = plt.cm.Reds.copy()
cmap.set_bad(color='white')
h1 = axes[0].imshow(hist, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                  origin='lower', aspect='auto', cmap=cmap, interpolation='nearest')
cbar = fig.colorbar(h1, label="Number of Galaxies in Bin", ax=axes[0], shrink=(0.6, 0.8)[c])
axes[0].set_xlabel(test_X_name)
axes[0].set_ylabel(test_Y_name)
axes[0].set_title("Test Data")

# seperates the galaxies into bins of test X with each containing equal numbers of galaxies, 
# then plots the median of test X against the median of test Y for each bin
bins = np.quantile(test_X, np.linspace(0, 1, nbins + 1))
bin_indices = np.digitize(test_X, bins)
test_X_medians, test_Y_medians = ([], [])
for i in range(1, len(bins)):
    bin_mask = bin_indices == i
    test_X_medians.append(np.median(test_X[bin_mask]))
    test_Y_medians.append(np.median(test_Y[bin_mask]))
axes[0].plot(test_X_medians, test_Y_medians, c='b', linewidth=2, alpha=0.7, label=f'median Log_{10}($f_esc$)')
axes[0].plot((x_min, x_max), (x_min, x_max),
             c='black', linewidth=2, alpha=0.5, label='$f_{esc}$ = $f_{esc}$ predicted')
axes[0].legend()

# plots a 2d histogram of train X data against train Y data,
# where the number of galaxies in a bin dictates it's colour
hist, xedges, yedges = np.histogram2d(train_X, train_Y, bins=[x_bins, x_bins])
hist = hist.T
hist = np.ma.masked_where(hist == 0, hist)
cmap = plt.cm.Reds.copy()
cmap.set_bad(color='white')
h2 = axes[1].imshow(hist, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                  origin='lower', aspect='auto', cmap=cmap, interpolation='nearest')
cbar = fig.colorbar(h2, label="Number of Galaxies in Bin", ax=axes[1], shrink=(0.6, 0.8)[c])
axes[1].set_xlabel(train_X_name)
axes[1].set_ylabel(train_Y_name)
axes[1].set_title("Train Data")

# seperates the galaxies into bins of train X with each containing equal numbers of galaxies, 
# then plots the median of train X against the median of train Y for each bin
bins = np.quantile(train_X, np.linspace(0, 1, nbins + 1))
bin_indices = np.digitize(train_X, bins)
train_X_medians, train_Y_medians = ([], [])
for i in range(1, len(bins)):
    bin_mask = bin_indices == i
    train_X_medians.append(np.median(train_X[bin_mask]))
    train_Y_medians.append(np.median(train_Y[bin_mask]))
axes[1].plot(train_X_medians, train_Y_medians, c='b', linewidth=2, alpha=0.7, label='median $f_{esc}$')
axes[1].plot((x_min, x_max), (x_min, x_max),
             c='black', linewidth=2, alpha=0.5, label='$f_{esc}$ = $f_{esc}$ predicted')
axes[1].legend()

for ax in axes:
    ax.set_box_aspect(1)

plt.tight_layout()
plt.show()