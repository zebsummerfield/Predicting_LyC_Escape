import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from functions import *

# 0 for f_esc, 1 for n_esc
f_or_n = 1

folder = "final_graph_generation/"
file = 'cat_dusttestszeb.hdf5'
with h5py.File(file, 'r') as hdf:
    f_esc = np.array(hdf['f_esc_vir_full']).astype('float32')
    n_esc = np.array(hdf['Ndot_LyC_vir_full'])
    redshift = np.array(hdf['redshift_full'])
    sfr50 = np.array(hdf['sfr_full_50'])
    sfr10 = np.array(hdf['sfr_full_10'])

    uv_lum_int_0 = np.array(hdf['uv_lum_int_full'])
    uv_lum_int_20 = np.array(hdf['uv_lum_int_fdust_20_full'])
    uv_lum_int_40 = np.array(hdf['uv_lum_int_fdust_40_full'])
    uv_lum_obs_0 = np.array(hdf['uv_lum_obs_full'])
    uv_lum_obs_20 = np.array(hdf['uv_lum_obs_fdust_20_full'])
    uv_lum_obs_40 = np.array(hdf['uv_lum_obs_fdust_40_full'])

    uv_vars = np.array([uv_lum_int_0, uv_lum_int_20, uv_lum_int_40,
                     uv_lum_obs_0, uv_lum_obs_20, uv_lum_obs_40])
    
    bad_indices = []
    for i in range(len(np.concatenate((uv_vars, [f_esc, n_esc], [sfr50])))):
        b_i = [index for index, val in enumerate(list(np.concatenate((uv_vars, [f_esc, n_esc], [sfr50]))[i]))
                        if (val == 0 or val == 1 or val == np.inf or val== -np.inf or val==np.nan)]
        print(f"feature {i+1} bad rows: {len(b_i)}")
        bad_indices += b_i
    bad_indices = list(set(bad_indices))[::-1]
    f_esc, n_esc = (np.delete(f_esc, bad_indices), np.delete(n_esc, bad_indices))
    redshift = np.delete(redshift, bad_indices)
    sfr10, sfr50 = (np.delete(sfr10, bad_indices), np.delete(sfr50, bad_indices))
    uv_vars = np.delete(uv_vars, bad_indices, axis=1)
    print(f'rows remaining: {len(f_esc)}')

    # calculates the UV attenuation for dust models 0, 20 and 40
    log_f_esc = np.log10(f_esc).astype('float32')
    log_n_esc = np.log10(n_esc).astype('float32')
    lum_to_tenpc = 4 * np.pi * (10 * 3.086e18)**2
    mag_vars = -2.5 * np.log10((uv_vars) / lum_to_tenpc) - 48.6
    attenuation_vars = mag_vars[3:6] - mag_vars[0:3]


plt.style.use('./MNRAS_Style.mplstyle')
mpl.rcParams.update({'font.size': 20})
fig, axes = plt.subplots(2, 3, figsize=(24, 12))

for ax_i in range(len(axes)):
    for ax_j in range(len(axes[0])):
        ax = axes[ax_i][ax_j]
        x = mag_vars[3:6][ax_j]
        if ax_i == 1:
            ax.set_xlabel(["0\% dust $M_\mathrm{UV,obs}$", "20\% dust $M_\mathrm{UV,obs}$", "40\% dust $M_\mathrm{UV,obs}$"][ax_j])
        if ax_i == 0:
            y = attenuation_vars[ax_j]
            ax.set_ylabel("$A_{1500}$")
            y_limits = (0, max(y))
        else:
            f_or_n_str = ['$\mathrm{Log}_{10}(f_{\mathrm{esc}})$',
                          '$\mathrm{Log}_{10}(\dot{n}_{\mathrm{ion,esc}} \; [\mathrm{s^{-1}}])$'][f_or_n]
            y = [log_f_esc, log_n_esc][f_or_n]
            ax.set_ylabel(f_or_n_str)
            y_limits = ((-6, 0), (45, 54))[f_or_n]

        # plots a 2d histogram of attenuation against y where the number of galaxies in a bin dictates it's colour
        nbins = 100
        hist, xedges, yedges = np.histogram2d(x, y, bins=nbins, range=((min(x), max(x)), (min(y), max(y))))
        hist = hist.T
        hist = np.log10(hist)
        hist[hist == -np.inf] = 0
        h1 = ax.imshow(hist, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                        origin='lower', aspect='auto', cmap='viridis', interpolation='nearest', vmin=0)

        # seperates the galaxies into bins of attenuation with each containing equal numbers of galaxies 
        nbins = 100
        bins = np.quantile(x, np.linspace(0, 1, nbins + 1))
        bin_indices = np.digitize(x, bins)
        x_medians, y_medians = ([], [])
        y_16th, y_84th = ([], [])
        for i in range(1, len(bins)):
            bin_mask = bin_indices == i
            x_medians.append(np.median(x[bin_mask]))
            y_medians.append(np.median(y[bin_mask]))
            y_16th.append(np.percentile(y[bin_mask], 16))
            y_84th.append(np.percentile(y[bin_mask], 84))

        # plots the median of log_x against the median of log_y for each bin
        ax.plot(x_medians, y_medians, c='r', linewidth=3, alpha=0.8, label="median $f_{esc}$")
        ax.fill_between(x_medians, y_16th, y_84th, color='r', alpha=0.2, label="16th-84th percentile", zorder=5)
        ax.set_xlim(min(x_medians), max(x_medians))
        ax.set_ylim(y_limits)
        ax.set_box_aspect(1)
        ax.tick_params(left=False, right=False, top=False, bottom=False)
        ax.minorticks_off()
        ax.grid(False)


fig.tight_layout()
cbar = fig.colorbar(h1, ax=axes, orientation='vertical', aspect=30, pad=0.03)
cbar.set_label("Number of Galaxies in Bin", labelpad=5)

mpl.rcParams['figure.dpi'] = 500
fig.savefig(folder + "report_graphs/report_graph.png", bbox_inches='tight', dpi=500)
plt.show()