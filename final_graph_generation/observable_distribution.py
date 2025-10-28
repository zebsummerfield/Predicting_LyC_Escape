import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from functions import *

folder = "final_graph_generation/"
file = 'cat.hdf5'

# 0 for f_esc, 1 for n_esc
f_or_n = 0
# True if model is generated to predict for an observational catalogue 
obvs = False

folder = "final_graph_generation/"
file = 'cat.hdf5'
keys, log_vars, log_f_esc, log_n_esc = prepare_data(file, f_or_n=f_or_n, obvs=obvs, eps=True,
                                                    add_vars=['stellar_mass_full', 'sfr_full_50'])
ssfr50 = ssfr_func(10**log_vars[-1], 10**log_vars[-2])

colors = [
    '#1b9e77',  # teal green
    '#d95f02',  # orange
    '#7570b3',  # muted purple
    '#e7298a'   # magenta-pink
]
plt.style.use('./MNRAS_Style.mplstyle')
mpl.rcParams.update({'font.size': 20})
fig = plt.figure(figsize=(20, 5))
gs = mpl.gridspec.GridSpec(1, 4, wspace=0)
ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1], sharey=ax0)
ax2 = fig.add_subplot(gs[2], sharey=ax1)
ax3 = fig.add_subplot(gs[3], sharey=ax2)
axes = [ax0, ax1, ax2, ax3]

ax0.set_ylim(0, 1)
ax0.set_ylabel('Probability Density [dex$^{-1}$]')

# here the distributions of variables in the filtered dataset are plotted in histograms
range = (-7, 0)
ax2.hist(log_f_esc, density=True, bins=100, alpha=0.8, color=colors[0])
ax2.set_xlim(range)
ax2.set_xlabel('$\mathrm{Log}_{10}(f_{\mathrm{esc}})$')
ax2.tick_params(labelleft=False)

range = (44.5, 54)
ax3.hist(log_n_esc, density=True, bins=100, alpha=0.8, color=colors[1])
ax3.set_xlim(range)
ax3.set_xlabel('$\mathrm{Log}_{10}(\dot{n}_{\mathrm{ion,esc}} \; [\mathrm{s^{-1}}])$')
ax3.tick_params(labelleft=False)

range = (5.5, 11)
ax0.hist(log_vars[-2], density=True, bins=100, alpha=0.8, color=colors[2])
ax0.set_xlim(range)
ax0.set_xlabel('$\mathrm{Log}_{10}(M_* \; [\mathrm{M_\odot}])$')

range = (-3.5, 2)
ax1.hist(np.log10(ssfr50), density=True, bins=100, alpha=0.8, color=colors[3])
ax1.set_xlabel('$\mathrm{Log}_{10}(\mathrm{sSFR_{50}} \; [\mathrm{Gyr^{-1}}])$')
ax1.set_xlim(range)
ax1.tick_params(labelleft=False)

for ax in axes:
    ax.grid(True, alpha=0.8, linestyle='--')
    ax.set_axisbelow(True)
    for line in ax.get_xgridlines() + ax.get_ygridlines():
        line.set_zorder(0)

mpl.rcParams['figure.dpi'] = 500
plt.tight_layout()
fig.savefig(folder + "report_graphs/report_graph.png", dpi=500)
plt.show()