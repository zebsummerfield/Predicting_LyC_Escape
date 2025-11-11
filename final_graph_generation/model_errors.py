import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
import matplotlib as mpl

bars = True

folder = "final_rf_model/"
file1 = folder + 'f_esc_rf_final_test_train.json'
file2 = folder + 'n_esc_rf_final_test_train.json'
file3 = folder + 'f_esc_rf_observational_charlotte_test_train.json'
file4 = folder + 'n_esc_rf_observational_charlotte_test_train.json' 
files = [file3, file4, file1, file2]
file_strs = ['Model C', 'Model D', 'Model A', 'Model B']
colours = ['pink', 'lime', mpl.colormaps['Purples'](1.0), mpl.colormaps['Greens'](1.0)]

plt.style.use('./MNRAS_Style.mplstyle')
mpl.rcParams.update({'font.size': 20})
fig, axes = plt.subplots(2, 1, figsize=(8, 8))

for f_i in range(len(files)):
    ax_i = f_i % 2
    with open(files[f_i], 'r') as json_data:
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
    
    if not bars:
        axes[ax_i].bar(pred_medians, pred_mae, color=colours[f_i], alpha=0.9, width=0.1)
    
    else:
        bin_centres = (bins[:-1] + bins[1:]) / 2
        bin_widths = np.diff(bins)
        axes[ax_i].bar(bin_centres, pred_mae, width=bin_widths,
                       label=file_strs[f_i], color=colours[f_i], alpha=0.8, edgecolor='black')


axes[0].set_xlabel("$\mathrm{Log}_{10}(f_\mathrm{esc} \; \mathrm{Predicted})$")
axes[0].set_xlim(-5, 0)
axes[1].set_xlabel("$\mathrm{Log}_{10}(\dot{n}_\mathrm{ion,esc} \; \mathrm{Predicted})$")
axes[1].set_xlim(46, 55)
for ax in axes:
    ax.set_ylabel("MAE [dex]")
    ax.set_ylim(0.25, 0.75)
    ax.grid(False)
    # add grid lines in background of graph
    # axes[ax_i].grid(True, alpha=0.8, linestyle='--')
    # axes[ax_i].set_axisbelow(True)
    # for line in axes[ax_i].get_xgridlines() + axes[ax_i].get_ygridlines():
    #     line.set_zorder(0)
    ax.legend()

mpl.rcParams['figure.dpi'] = 500
plt.tight_layout()

folder = 'final_graph_generation/'
fig.savefig(folder + "report_graphs/report_graph.png", dpi=500)
plt.show()