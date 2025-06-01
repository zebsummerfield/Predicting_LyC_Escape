import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

def truncate_colormap(cmap, minval=0.5, maxval=1.0, n=256):
    new_cmap = LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap

folder = "final_model/"
file1 = folder + 'f_esc_rf_final_test_train.json'
file2 = folder + 'n_esc_rf_final_test_train.json'
file3 = folder + 'f_esc_rf_observational_test_train.json'
file4 = folder + 'n_esc_rf_observational_test_train.json' 

# True if model is generated to predict for an observational catalogue 
obvs = True

# True if histogram bin colour is logarithmic
log = False

with open((file1, file3)[obvs], 'r') as json_data:
    f_data = json.load(json_data)
    f_test = np.array(f_data['f_esc_test'])
    f_test_pred = np.array(f_data['f_esc_test_pred'])
    f_train = np.array(f_data['f_esc_train'])
    f_train_pred = np.array(f_data['f_esc_train_pred'])

    # calculate errors on the test and train data
    f_test_mae = mean_absolute_error(f_test, f_test_pred)
    f_test_mse = root_mean_squared_error(f_test, f_test_pred)
    print(f"f_esc Test Mean Absolute Error: {f_test_mae}")
    print(f"f_esc Test Mean Squared Error: {f_test_mse}")
    f_train_mae = mean_absolute_error(f_train, f_train_pred)
    f_train_mse = root_mean_squared_error(f_train, f_train_pred)
    print(f"f_esc Train Mean Absolute Error: {f_train_mae}")
    print(f"f_esc Train Mean Squared Error: {f_train_mse}")

    print(f"Maximum f_esc Test Prediction: {max(f_test_pred)}")
    print(f"Minimum f_esc Test Prediction: {min(f_test_pred)}")

with open((file2, file4)[obvs], 'r') as json_data:
    n_data = json.load(json_data)
    n_test = np.array(n_data['f_esc_test'])
    n_test_pred = np.array(n_data['f_esc_test_pred'])
    n_train = np.array(n_data['f_esc_train'])
    n_train_pred = np.array(n_data['f_esc_train_pred'])

    # calculate errors on the test and train data
    n_test_mae = mean_absolute_error(n_test, n_test_pred)
    n_test_mse = root_mean_squared_error(n_test, n_test_pred)
    print(f"n_esc Test Mean Absolute Error: {n_test_mae}")
    print(f"n_esc Test Mean Squared Error: {n_test_mse}")
    n_train_mae = mean_absolute_error(n_train, n_train_pred)
    n_train_mse = root_mean_squared_error(n_train, n_train_pred)
    print(f"n_esc Train Mean Absolute Error: {n_train_mae}")
    print(f"n_esc Train Mean Squared Error: {n_train_mse}")

    print(f"Maximum n_esc Test Prediction: {max(n_test_pred)}")
    print(f"Minimum n_esc Test Prediction: {min(n_test_pred)}")

plt.style.use('./MNRAS_Style.mplstyle')
mpl.rcParams.update({'font.size': 20})
fig = plt.figure(figsize=(16, 10))
# Create subplots with explicit positions - this gives us total control
# [left, bottom, width, height] - all values are fractions of figure size
ax1 = fig.add_axes([0.1, 0.38, 0.3, 0.57])  # top left
ax2 = fig.add_axes([0.6, 0.38, 0.3, 0.57])   # top right
ax3 = fig.add_axes([0.1, 0.28, 0.3, 0.13])   # bottom left - smaller height
ax4 = fig.add_axes([0.6, 0.28, 0.3, 0.13])    # bottom right - smaller height
axes = np.array([[ax1, ax2], [ax3, ax4]])
hists = []

f_esc_str = "$\mathrm{Log}_{10}(f_\mathrm{esc})$"
f_esc_pred_str = "$\mathrm{Log}_{10}(f_\mathrm{esc} \; \mathrm{Predicted})$"
n_esc_str = "$\mathrm{Log}_{10}(\dot{n}_\mathrm{ion,esc})$"
n_esc_pred_str = "$\mathrm{Log}_{10}(\dot{n}_\mathrm{ion,esc} \; \mathrm{Predicted})$"

for ax_i in range(len(axes)):
    target_str = [f_esc_str, n_esc_str][ax_i]
    target_pred_str = [f_esc_pred_str, n_esc_pred_str][ax_i]
    test = [f_test, n_test][ax_i]
    test_pred = [f_test_pred, n_test_pred][ax_i]

    x_min, x_max = [(-5, 0), (46, 54)][ax_i]
    nbins = 60
    x_bins = np.linspace(x_min, x_max, nbins+1)
    
    axes[0, ax_i].set_xlim(x_min, x_max)
    axes[0, ax_i].set_ylim(x_min, x_max)
    axes[1, ax_i].set_xlim(x_min, x_max)
    # hide x labels for top plots to improve appearance
    plt.setp(axes[0, ax_i].get_xticklabels(), visible=False)

    # plots a 2d histogram of predicted test target against test target,
    # where the number of galaxies in a bin dictates it's colour
    hist, xedges, yedges = np.histogram2d(test_pred, test, bins=[x_bins, x_bins])
    hist = hist.T
    hist = np.ma.masked_where(hist == 0, hist)
    if log:
        hist = np.log10(hist)

    cmap = [truncate_colormap(plt.cm.Purples, 0.1, 1.0), truncate_colormap(plt.cm.Greens, 0.1, 1.0)][ax_i]
    # cmap.set_bad(color='#d3d3d3')
    h1 = axes[0, ax_i].imshow(hist, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                              origin='lower', aspect='auto', cmap=cmap, interpolation='nearest', zorder=1)
    hists.append(h1)
    # plt.colorbar(h1, label=color_label, ax=axes[0][ax_i], fraction=0.046, pad=0.04)
    axes[0, ax_i].set_ylabel(target_str)
    # ensure ylabel doesn't get cut off
    axes[0, ax_i].yaxis.set_label_coords(-0.10, 0.5)

    # seperates the galaxies into bins of predicted test target with each containing equal numbers of galaxies, 
    # then plots the median of predicted test target against the median of test target for each bin
    nbins = 30
    bins = np.quantile(test_pred, np.linspace(0, 1, nbins + 1))
    bin_indices = np.digitize(test_pred, bins)
    test_pred_medians = np.zeros(nbins) 
    test_medians = np.zeros(nbins)
    test_pred_mae  = np.zeros(nbins)
    test_pred_mse = np.zeros(nbins) 
    for i in range(1, len(bins)):
        bin_mask = bin_indices == i
        test_pred_medians[i-1] = np.median(test_pred[bin_mask])
        test_medians[i-1] = np.median(test[bin_mask])
        test_pred_mae[i-1] = mean_absolute_error(test[bin_mask], test_pred[bin_mask])
        test_pred_mse[i-1] = root_mean_squared_error(test[bin_mask], test_pred[bin_mask])
    axes[0, ax_i].plot(test_pred_medians, test_medians, c='r', linewidth=2.5, alpha=0.8, zorder=2)
    #axes[ax_i].plot(test_pred_medians, test_medians + test_pred_mae, c='b', linewidth=2.5, alpha=0.8, zorder=2)
    #axes[ax_i].plot(test_pred_medians, test_medians - test_pred_mae, c='b', linewidth=2.5, alpha=0.8, zorder=2)

    # plots the line of y = x
    axes[0, ax_i].plot((x_min, x_max), (x_min, x_max), c='black', linewidth=1.5, alpha=0.6, zorder=3)

    purple_max = mpl.colormaps['Purples'](1.0)
    green_max = mpl.colormaps['Greens'](1.0)
    axes[1, ax_i].plot(test_pred_medians, test_pred_mae, c=(purple_max, green_max)[ax_i], linewidth=3, alpha=0.9)
    axes[1, ax_i].set_ylim(0.275, 0.675)
    axes[1, ax_i].set_xlabel(target_pred_str)
    axes[1, ax_i].set_ylabel('MAE [dex]')
    axes[1, ax_i].yaxis.set_label_coords(-0.10, 0.5)

    axes[0, ax_i].set_box_aspect(1)
    axes[0, ax_i].grid(True, alpha=0.4, linestyle='--')
    axes[1, ax_i].grid(True, alpha=0.8, linestyle='--')
    # add grid lines in background of graph
    for ax in axes[:,ax_i]:
        ax.set_axisbelow(True)
        for line in ax.get_xgridlines() + ax.get_ygridlines():
            line.set_zorder(0)

if log:
    color_label = "$\mathrm{Log}_{10}(\mathrm{N_{bin}})$"
else:
    color_label = "$\mathrm{N_{bin}}$"
cbar_ax1 = fig.add_axes([0.1, 0.92, 0.3, 0.02])  # [left, bottom, width, height]
cbar_ax2 = fig.add_axes([0.6, 0.92, 0.3, 0.02])   # [left, bottom, width, height]
cbar_ax1.xaxis.set_label_coords(0.5, 3)
cbar_ax2.xaxis.set_label_coords(0.5, 3)
cbar1 = fig.colorbar(hists[0], cax=cbar_ax1, orientation='horizontal')
cbar2 = fig.colorbar(hists[1], cax=cbar_ax2, orientation='horizontal')
cbar1.set_label(color_label)
cbar2.set_label(color_label)
plt.subplots_adjust(top=0.9)  # Leave space at top for colorbars
# Move ticks and labels to the top of the colorbar
cbar1.ax.xaxis.set_ticks_position('top')
cbar1.ax.xaxis.set_label_position('top')
cbar2.ax.xaxis.set_ticks_position('top')
cbar2.ax.xaxis.set_label_position('top')

mpl.rcParams['figure.dpi'] = 500
folder = "final_graph_generation/"
fig.savefig(folder + "report_graphs/report_graph.png", bbox_inches='tight', dpi=500)
plt.show()