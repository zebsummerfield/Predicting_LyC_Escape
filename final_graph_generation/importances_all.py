import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from functions import tableau20

folder = "final_rf_model/"
file1 = folder + 'f_esc_rf_final_test_train.json'
file2 = folder + 'n_esc_rf_final_test_train.json'
file3 = folder + 'f_esc_rf_observational_charlotte_test_train.json'
file4 = folder + 'n_esc_rf_observational_charlotte_test_train.json' 
files = [file1, file2, file3, file4]
model_strs = ['Model A - Best $f_\mathrm{esc}$ Predictor',
              'Model B - Best $n_\mathrm{ion,esc}$ Predictor',
              'Model C - Observational $f_\mathrm{esc}$ Predictor',
              'Model D - Observational $n_\mathrm{ion,esc}$ Predictor']

f_key_strs = np.array(['$\Delta\mathrm{MS}_{10}$', '$\mathrm{sSFR}_{100}$', '$\mathrm{SFR}_{10}/\mathrm{SFR}_{100}$',
                       '$M_*$', '$M_\mathrm{gas}/M_*$', '$M_*/M_\mathrm{vir}$',
                       '$Z$', '$M_\mathrm{UV}$', '$L_\mathrm{UV}/L_\mathrm{H\\alpha}$',
                       '$R_\mathrm{UV}$', '$R_\mathrm{H\\alpha}$', '$R_\mathrm{SFR}$',
                       '$R_\mathrm{SFR}/R_{M_*}$', '$\Sigma_\mathrm{SFR_{10}}$', '$1+z$',
                       'Rand'])
f_obvs_key_strs = f_key_strs[[0, 1, 2, 3, 7, 14, 15]]
n_key_strs = np.array(['$\mathrm{SFR}_{10}$', '$\mathrm{SFR}_{100}$', '$M_*$', 
                        '$M_\mathrm{gas}/M_*$', '$M_*/M_\mathrm{vir}$', '$Z$',
                        '$M_\mathrm{UV}$', '$L_\mathrm{H\\alpha}$', '$1+z$',
                        'Rand'])
n_obvs_key_strs = n_key_strs[[0, 1, 2, 6, 8, 9]]
key_strs = [f_key_strs, n_key_strs, f_obvs_key_strs, n_obvs_key_strs]

plt.style.use('./MNRAS_Style.mplstyle')
mpl.rcParams.update({'font.size': 20})
fig = plt.figure(figsize=(20, 12))
# Create subplots with explicit positions - this gives us total control
# [left, bottom, width, height] - all values are fractions of figure size
ax1 = fig.add_axes([0.05, 0.75, 0.9, 0.2])  # Model A - f_esc best predictor
ax2 = fig.add_axes([0.05, 0.4, 0.9, 0.2])  # Model B - n_esc best predictor
ax3 = fig.add_axes([0.05, 0.075, 0.4, 0.2])  # Model C - f_esc obvs predictor
ax4 = fig.add_axes([0.55, 0.075, 0.4, 0.2])  # Model D - n_esc obvs predictor
axes = [ax1, ax2, ax3, ax4]

for m_i in range(len(model_strs)):
    ax = axes[m_i]
    with open(files[m_i], 'r') as json_data:
        data = json.load(json_data)
        keys = np.array(data['keys'])
    importances = np.array(data['importances'])
    std_importances = np.array(data['std_importances'])

    ax.tick_params(axis='x', which='both', bottom=False, top=False)
    ax.grid(False)
    ax.grid(True, axis='y', alpha=0.8)
    for line in ax.get_ygridlines():
        line.set_zorder(0)

    sorted_indices = np.argsort(importances)[::-1]
    colors = np.array([tableau20[k] for k in keys])

    bar_width = (0.4, 0.8)[m_i >= 2]
    x = np.linspace(0, 10, len(keys))
    ax.bar(x, importances[sorted_indices], yerr=std_importances[sorted_indices], 
                   width=bar_width, color=colors[sorted_indices], capsize=5, edgecolor='black', zorder=2)
    ax.set_ylabel('Importance')
    #ax.set_ylim((((0, 0.2), (0, 0.4))[m_i==1], (0, 0.5))[m_i>=2])
    ax.set_ylim(((0, 0.4), (0, 0.5))[m_i>=2])
    ax.set_xticks(x)
    ax.set_xticklabels(key_strs[m_i][sorted_indices], rotation='vertical')
    ax.set_xlim(x[0] - bar_width, x[-1] + bar_width)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=(4, 5)[m_i >= 2]))


    ax.text(0.5, (0.925, 0.95)[m_i >= 2], model_strs[m_i], 
                ha='center', va='top', transform=ax.transAxes)

mpl.rcParams['figure.dpi'] = 500
folder = "final_graph_generation/"
fig.savefig(folder + "report_graphs/report_graph.png", dpi=500)
plt.show()