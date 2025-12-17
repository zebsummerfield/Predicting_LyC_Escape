import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from functions import *
from matplotlib.ticker import MaxNLocator

folder = "final_graph_generation/"
file = 'cat.hdf5'

# 0 for f_esc, 1 for n_esc
f_or_n = 0
# True if model is generated to predict for an observational catalogue 
obvs = False
# True to split the data into redshift bins
split_redshift = True
# True to use the dusty Thesan-Zoom catalogue, False for dust-free catalogue
dusty = True

folder = "final_graph_generation/"
file = ['cat.hdf5', 'cat_dusttestszeb.hdf5'][dusty]
keys, log_vars, log_f_esc, log_n_esc = prepare_data(file, f_or_n=f_or_n, obvs=obvs, dusty=dusty, eps=True,
                                                    add_vars=['redshift_full', 'stellar_mass_full', 'sfr_full_50'])
ssfr50 = ssfr_func(10**log_vars[-1], 10**log_vars[-2])

colors = [
    '#1b9e77',  # teal green
    '#d95f02',  # orange
    '#7570b3',  # muted purple
    '#e7298a'   # magenta-pink
]
plt.style.use('./MNRAS_Style.mplstyle')
nbins = 50

if not split_redshift:
    mpl.rcParams.update({'font.size': 17})
    fig = plt.figure(figsize=(20, 5))
    gs = mpl.gridspec.GridSpec(1, 4, wspace=0.05)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharey=ax0)
    ax2 = fig.add_subplot(gs[2], sharey=ax0)
    ax3 = fig.add_subplot(gs[3], sharey=ax0)
    axes = np.array([ax0, ax1, ax2, ax3])

    ax0.set_ylim(0, (10000, 4000)[dusty])
    ax0.set_ylabel('Number of Galaxies per Bin')

    # here the distributions of variables in the filtered dataset are plotted in histograms
    range = (-7, 0)
    ax2.hist(log_f_esc, bins=nbins, alpha=0.8, color=colors[0])
    ax2.set_xlim(range)
    ax2.set_xlabel('$\mathrm{Log}_{10}(f_{\mathrm{esc}})$')
    ax2.tick_params(labelleft=False)

    range = (43, 55)
    ax3.hist(log_n_esc, bins=nbins, alpha=0.8, color=colors[1])
    ax3.set_xlim(range)
    ax3.set_xlabel('$\mathrm{Log}_{10}(\dot{n}_{\mathrm{ion,esc}} \; [\mathrm{s^{-1}}])$')
    ax3.tick_params(labelleft=False)

    range = (6, 11)
    ax1.hist(log_vars[-2], bins=nbins, alpha=0.8, color=colors[2], align='left')
    ax1.set_xlim(range)
    ax1.set_xlabel('$\mathrm{Log}_{10}(M_* \; [\mathrm{M_\odot}])$')
    ax1.tick_params(labelleft=False)

    range = (3, 16)
    ax0.hist(10**log_vars[-3], bins=nbins, alpha=0.8, color=colors[3])
    ax0.set_xlabel('$z$')
    ax0.set_xlim(range)

    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))


else:
    z1 = (log_vars[-3] < np.log10(6))
    z2 = (log_vars[-3] < np.log10(9)) & (log_vars[-3] >= np.log10(6))
    z3 = (log_vars[-3] >= np.log10(9))
    zranges = [z1, z2, z3]

    mpl.rcParams.update({'font.size': 17})
    fig = plt.figure(figsize=(20, 15))
    gs = mpl.gridspec.GridSpec(3, 4, wspace=0.05, hspace=0.05)
    axes = np.zeros((3, 4), dtype=object)
    for i in range(3):
        for j in range(4):
            if i == 0 and j == 0:
                ax = fig.add_subplot(gs[2-i, j])
            elif i == 0:
                ax = fig.add_subplot(gs[2-i, j], sharey=axes[0, 0])
                ax.tick_params(labelleft=False)
            elif j == 0:
                ax = fig.add_subplot(gs[2-i, j], sharex=axes[0, 0])
                ax.tick_params(labelbottom=False)
            else:
                ax = fig.add_subplot(gs[2-i, j], sharex=axes[0, j], sharey=axes[i, 0])
                ax.tick_params(labelbottom=False, labelleft=False)
            axes[i, j] = ax

            if j == 0:
                xrange = (6, 11)
                bin_edges = np.linspace(xrange[0]-0.1, xrange[1], nbins + 1)
                bin_width = bin_edges[1] - bin_edges[0]
                shifted_edges = bin_edges - bin_width / 2
                ax.hist(log_vars[-2][zranges[i]], bins=shifted_edges, alpha=0.8, color=colors[2], align='left')
                ax.set_xlim(xrange)
            elif j == 1:
                xrange = (-3.5, 1.5)
                bin_edges = np.linspace(xrange[0], xrange[1], nbins + 1)
                ax.hist(np.log10(ssfr50[zranges[i]]), bins=bin_edges, alpha=0.8, color=colors[3])
                ax.set_xlim(xrange)
            elif j == 2:
                xrange = (-7, 0)
                bin_edges = np.linspace(xrange[0], xrange[1], nbins + 1)
                ax.hist(log_f_esc[zranges[i]], bins=bin_edges, alpha=0.8, color=colors[0])
                ax.set_xlim(xrange)
            else:
                xrange = (43, 55)
                bin_edges = np.linspace(xrange[0], xrange[1], nbins + 1)
                ax.hist(log_n_esc[zranges[i]], bins=bin_edges, alpha=0.8, color=colors[1])
                ax.set_xlim(xrange)

            if i == 0:
                ax.text(0.1, 0.9, '$3 \\leq z < 6$', transform=ax.transAxes, ha='left', va='top')
            elif i == 1:
                ax.text(0.1, 0.9, '$6 \\leq z < 9$', transform=ax.transAxes, ha='left', va='top')
            elif i == 2:
                ax.text(0.1, 0.9, '$9 \\leq z \\leq 16$', transform=ax.transAxes, ha='left', va='top')

    axes[0,0].set_xlabel('$\mathrm{Log}_{10}(M_* \; [\mathrm{M_\odot}])$')
    axes[0,1].set_xlabel('$\mathrm{Log}_{10}(\mathrm{sSFR_{50}} \; [\mathrm{Gyr^{-1}}])$')
    axes[0,2].set_xlabel('$\mathrm{Log}_{10}(f_{\mathrm{esc}})$')
    axes[0,3].set_xlabel('$\mathrm{Log}_{10}(\dot{n}_{\mathrm{ion,esc}} \; [\mathrm{s^{-1}}])$')
    for ax in (axes[0,:]):
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    axes[0,0].set_ylabel('Number of Galaxies per Bin')
    axes[1,0].set_ylabel('Number of Galaxies per Bin')
    axes[2,0].set_ylabel('Number of Galaxies per Bin')
    axes[0,0].set_ylim(0, (4500, 2000)[dusty])
    axes[1,0].set_ylim(0, (4500, 2000)[dusty])
    axes[2,0].set_ylim(0, (2000, 500)[dusty])



for ax in axes.flatten():
    ax.grid(False)
    # ax.grid(True, alpha=0.8, linestyle='--')
    # ax.set_axisbelow(True)
    # for line in ax.get_xgridlines() + ax.get_ygridlines():
    #     line.set_zorder(0)

mpl.rcParams['figure.dpi'] = 500
plt.tight_layout()
fig.savefig(folder + "report_graphs/report_graph.png", dpi=500)
plt.show()