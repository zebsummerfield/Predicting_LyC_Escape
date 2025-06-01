import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
import matplotlib as mpl

folder = "final_model/"
file1 = folder + 'f_esc_rf_final_test_train.json'
file2 = folder + 'n_esc_rf_final_test_train.json'
file3 = folder + 'f_esc_rf_observational_test_train.json'
file4 = folder + 'n_esc_rf_observational_test_train.json' 
files = [file1, file2, file3, file4]
file_strs = ['Model A', 'Model B', 'Model C', 'Model D']

plt.style.use('./MNRAS_Style.mplstyle')
mpl.rcParams.update({'font.size': 20})
fig, axes = plt.subplots(2, 1, figsize=(8, 8))

for ax_i in range(len(axes)):
    with open([file3, file4][ax_i], 'r') as json_data:
        data = json.load(json_data)

    test = np.array(data['f_esc_test'])
    test_pred = np.array(data['f_esc_test_pred'])

    # seperates the galaxies into bins of predicted test data with each containing equal numbers of galaxies,
    # then plots the median of predicted test data against the median of several error functions.
    nbins = 30
    bins = np.quantile(test_pred, np.linspace(0, 1, nbins + 1))
    bin_indices = np.digitize(test_pred, bins)
    pred_medians, pred_mae, pred_mse, pred_rmse, pred_std = ([], [], [], [], [])
    for i in range(1, len(bins)):
        bin_mask = bin_indices == i
        pred_medians.append(np.median(test_pred[bin_mask]))
        pred_mae.append(mean_absolute_error(test[bin_mask], test_pred[bin_mask]))
        pred_mse.append(mean_squared_error(test[bin_mask], test_pred[bin_mask]))
        pred_rmse.append(root_mean_squared_error(test[bin_mask], test_pred[bin_mask]))
        pred_std.append(np.std(test_pred[bin_mask] - test[bin_mask]))
    purple_max = mpl.colormaps['Purples'](1.0)
    green_max = mpl.colormaps['Greens'](1.0)
    axes[ax_i].plot(pred_medians, pred_mae, c=(purple_max, green_max)[ax_i], linewidth=3, alpha=0.9)

    if pred_medians[0] < 0:
        axes[ax_i].set_xlabel("$\mathrm{Log}_{10}(f_\mathrm{esc} \; \mathrm{Predicted})$")
        axes[ax_i].set_xlim(-4, 0)
    else:
        axes[ax_i].set_xlabel("$\mathrm{Log}_{10}(\dot{n}_\mathrm{ion,esc} \; \mathrm{Predicted})$")
        axes[ax_i].set_xlim(46, 53)
    axes[ax_i].set_ylabel("MAE [dex]")
    axes[ax_i].set_ylim(0.25, 0.75)

    # add grid lines in background of graph
    axes[ax_i].grid(True, alpha=0.8, linestyle='--')
    axes[ax_i].set_axisbelow(True)
    for line in axes[ax_i].get_xgridlines() + axes[ax_i].get_ygridlines():
        line.set_zorder(0) 

mpl.rcParams['figure.dpi'] = 500
plt.tight_layout()
folder = 'final_graph_generation/'
fig.savefig(folder + "report_graphs/report_graph.png", dpi=500)
plt.show()