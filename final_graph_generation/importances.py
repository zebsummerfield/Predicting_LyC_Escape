import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from functions import tableau20

folder = "final_model/"
file1 = folder + 'f_esc_rf_final_test_train.json'
file2 = folder + 'n_esc_rf_final_test_train.json'
file3 = folder + 'f_esc_rf_observational_test_train.json'
file4 = folder + 'n_esc_rf_observational_test_train.json' 

# True if model is generated to predict for an observational catalogue 
obvs = True

with open((file1, file3)[obvs], 'r') as json_data:
    f_data = json.load(json_data)
    f_keys = np.array(f_data['keys'])
    f_key_strs = np.array(['$\Delta\mathrm{MS}_{10}$', '$\mathrm{SFR}_{10}/\mathrm{SFR}_{100}$', '$M_*$', 
                        '$M_\mathrm{gas}/M_*$', '$M_*/M_\mathrm{vir}$', '$Z$',
                        '$M_\mathrm{UV}$', '$L_\mathrm{UV}/L_\mathrm{H\\alpha}$', '$R_\mathrm{UV}$', 
                        '$R_\mathrm{H\\alpha}$', '$R_\mathrm{SFR}$', '$R_\mathrm{SFR}/R_{M_*}$', 
                        '$\Sigma_\mathrm{SFR_{10}}$', '$1+z$', 'Rand'])
    if obvs:
        f_key_strs = f_key_strs[[0, 1, 2, 6, 7, 8, 9, 13, 14]]
    f_importances = np.array(f_data['importances'])
    f_std_importances = np.array(f_data['std_importances'])

with open((file2, file4)[obvs], 'r') as json_data:
    n_data = json.load(json_data)
    n_keys = np.array(n_data['keys'])
    n_key_strs = np.array(['$\mathrm{SFR}_{10}$', '$\mathrm{SFR}_{100}$', '$M_*$', 
                           '$M_\mathrm{gas}/M_*$', '$M_*/M_\mathrm{vir}$', '$Z$',
                           '$M_\mathrm{UV}$', '$L_\mathrm{H\\alpha}$', '$1+z$',
                           'Rand'])
    if obvs:
        n_key_strs = n_key_strs[[0, 1, 2, 6, 7, 8, 9]]
    n_importances = np.array(n_data['importances'])
    n_std_importances = np.array(n_data['std_importances'])


plt.style.use('./MNRAS_Style.mplstyle')
mpl.rcParams.update({'font.size': 20})
if obvs:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
else: 
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

for ax_i in range(len(axes)):
    axes[ax_i].tick_params(axis='x', which='both', bottom=False, top=False)
    axes[ax_i].grid(False)
    axes[ax_i].grid(True, axis='y', alpha=0.8)
    for line in axes[ax_i].get_ygridlines():
        line.set_zorder(0)

    keys = [f_keys, n_keys][ax_i]
    key_strs = [f_key_strs, n_key_strs][ax_i]
    importances = [f_importances, n_importances][ax_i]
    std_importances = [f_std_importances, n_std_importances][ax_i]
    sorted_indices = np.argsort(importances)[::-1]
    colors = np.array([tableau20[k] for k in keys])

    bar_width = (0.4, 0.8)[obvs]
    x = np.linspace(0, 10, len(keys))
    axes[ax_i].bar(x, importances[sorted_indices], yerr=std_importances[sorted_indices], 
                   width=bar_width, color=colors[sorted_indices], capsize=5, edgecolor='black', zorder=2)
    axes[ax_i].set_ylabel('Importance')
    if obvs:
        axes[ax_i].set_ylim(((0, 0.3), (0, 0.4))[ax_i])
    else:
        axes[ax_i].set_ylim(((0, 0.2), (0, 0.4))[ax_i])
    axes[ax_i].set_xticks(x)
    axes[ax_i].set_xticklabels(key_strs[sorted_indices], rotation='vertical')
    axes[ax_i].set_xlim(x[0] - bar_width, x[-1] + bar_width)

mpl.rcParams['figure.dpi'] = 500
plt.tight_layout()
folder = "final_graph_generation/"
fig.savefig(folder + "report_graphs/report_graph.png", dpi=500)
plt.show()