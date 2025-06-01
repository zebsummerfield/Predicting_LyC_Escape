import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

c = 0

def sfms_func(xdata, s, b, u):
    redshift = xdata[0,:]
    stellar_mass = 10**xdata[1,:]
    return np.log10(s * (stellar_mass / 1e10)**b * (1 + redshift)**u)

def ssfr_func(sfr, mass):
    return np.array(sfr) / np.array(mass) * 1e9

s = [0.033, 0.067]
b = [0.041, 0.042]
u = [2.64, 2.57]

folder = "investigating_indicators/"
file = 'cat.hdf5'

with h5py.File(file, 'r') as hdf:
    with open(folder+"keys.txt", "w") as k:
        for key in hdf.keys():
            k.write(key + "\n")
    print(hdf.keys())

    f_esc = np.array(hdf['f_esc_vir_full']).astype('float32')
    n_esc = np.array(hdf['Ndot_LyC_vir_full'])

    redshift = np.array(hdf['redshift_full'])
    star_mass = np.array(hdf['stellar_mass_full'])
    ssfr10 = ssfr_func(hdf['sfr_full_10'], star_mass)
    ssfr50 = ssfr_func(hdf['sfr_full_50'], star_mass)
    ssfr100 = ssfr_func(hdf['sfr_full_100'], star_mass)
    vir_mass = np.array(hdf['M_vir_full'])
    gas_mass = np.array(hdf['gas_mass_full'])
    ha_lum = np.array(hdf['ha_lum_int_full'])
    uv_lum = np.array(hdf['uv_lum_int_full'])
    gas_met = np.array(hdf['gas_met_full'])
    star_met = np.array(hdf['star_met_full'])
    star_size = np.array(hdf['stellar_size_full'])
    sfr_size = np.array(hdf['sfr_size_full'])
    uv_size = np.array(hdf['uv_size_int_full'])
    ha_size = np.array(hdf['ha_size_int_full'])
    uv_projection = np.array(hdf['uv_size_int_2d_full'])
    ha_projection = np.array(hdf['ha_size_int_2d_full'])
    ssfr_density = ssfr100 / np.sqrt(uv_projection * ha_projection)
    uv_obs = np.array(hdf['uv_lum_obs_full'])

    vars = np.array([ssfr10, ssfr50, ssfr100, redshift, star_mass, gas_mass, star_mass/gas_mass, 
                     uv_lum, ha_lum, uv_lum/ha_lum, uv_lum*ha_lum, gas_met, star_met, gas_met*star_met,
                     sfr_size, star_size, sfr_size/star_size, uv_size, ha_size, 1/ssfr_density,
                     uv_obs/uv_lum, ha_size/uv_size])

    #variables we are investigating for an f_esc relationship
    x = star_size
    y = sfr_size

    # removes any rows that have zero, nan or infinity for the x, y, ssfr10, f_esc and n_esc
    for i in range(len(np.concatenate(([x, y], [f_esc, n_esc])))):
        nan_indices = [index for index, val in enumerate(list(np.concatenate(([x, y], [vars[0]], [f_esc, n_esc]))[i]))
                       if (val==0 or val == np.inf or val== -np.inf or np.isnan(val))][::-1]
        print(f"rows deleted: {len(nan_indices)}")
        f_esc, n_esc = (np.delete(f_esc, nan_indices), np.delete(n_esc, nan_indices))
        x, y = (np.delete(x, nan_indices), np.delete(y, nan_indices))
        vars = np.delete(vars, nan_indices, axis=1)
        redshift, star_mass = (np.delete(redshift, nan_indices), np.delete(star_mass, nan_indices))

    log_vars = np.log10(vars).astype('float32')
    log_f_esc = np.log10(f_esc).astype('float32')
    log_n_esc = np.log10(n_esc).astype('float32')

    # calculates the offset from the star-forming main sequence
    log_sfms10 =  log_vars[0] - sfms_func(np.array([redshift, np.log10(star_mass)]), s[0], b[0], u[0])
    log_sfms100 =  log_vars[2] - sfms_func(np.array([redshift, np.log10(star_mass)]), s[1], b[1], u[1])

    x = np.log10(x).astype('float32')
    y = np.log10(y).astype('float32')

matplotlib.rcParams.update({'font.size': (12, 10)[c]})
fig, axes = plt.subplots(2, 2, figsize=((14, 14), (8, 8))[c], gridspec_kw={'height_ratios': [1, 1.6]})

# seperates the galaxies into bins of variable x with each containing equal numbers of galaxies 
nbins = 50
bins = np.quantile(x, np.linspace(0, 1, nbins + 1))
bin_indices = np.digitize(x, bins)
x_medians, f_esc_medians = ([], [])
for i in range(1, len(bins)):
    bin_mask = bin_indices == i
    x_medians.append(np.median(x[bin_mask]))
    f_esc_medians.append(np.median(f_esc[bin_mask]))
# plots the median of x against the median of log_f_esc for each bin
axes[0][0].plot(x_medians, f_esc_medians, c='r', label="$Log_{10}$(X)")
axes[0][0].set_xlim(min(x_medians), max(x_medians))
axes[0][0].set_xlabel("$Log_{10}$(X)")
axes[0][0].set_ylabel("$f_{esc}$")
axes[0][0].legend(loc = 'upper left')

# seperates the galaxies into bins of variable y with each containing equal numbers of galaxies 
bins = np.quantile(y, np.linspace(0, 1, nbins + 1))
bin_indices = np.digitize(y, bins)
y_medians, f_esc_medians = ([], [])
for i in range(1, len(bins)):
    bin_mask = bin_indices == i
    y_medians.append(np.median(y[bin_mask]))
    f_esc_medians.append(np.median(f_esc[bin_mask]))
# plots the median of y against the median of log_f_esc for each bin on the same axes as the x plot
ax00 = axes[0][0].twiny()
ax00.plot(y_medians, f_esc_medians, c='g', label="$Log_{10}$(Y)")
ax00.set_xlim(min(y_medians), max(y_medians))
ax00.set_xlabel("$Log_{10}$(Y)")
ax00.legend(loc= 'upper right')

# plots a scatter of x against y where the colour and size of the points is determined by f_esc
sizes = np.where(f_esc > 0.2, 50, 5)
sorted_indices = np.argsort(f_esc)
color_scatter = axes[0][1].scatter(x[sorted_indices], y[sorted_indices], s=sizes[sorted_indices],
                                c=f_esc[sorted_indices], cmap='inferno', marker='.')
axes[0][1].set_xlabel("$Log_{10}$(X)")
axes[0][1].set_ylabel("$Log_{10}$(Y)")
fig.colorbar(color_scatter, label='$f_{esc}$')

# plots a 2d histogram of x against y where the colour of the bin is dictated by it's mean log_f_esc
nbins = 50
hist, xedges, yedges = np.histogram2d(x, y, bins=nbins, weights=log_f_esc)
hist_norm, xedges, yedges = np.histogram2d(x, y, bins=nbins)
hist = hist / hist_norm
vmax = np.log10(1)
vmin = np.log10(0.01)
hist = hist.T # transposes the histogram for proper orientation
color_hist = axes[1][0].imshow(hist, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                            origin='lower', aspect='auto', cmap='viridis', 
                            vmax=vmax, vmin=vmin, interpolation='nearest')
plt.colorbar(color_hist, label='$Log_{10}(f_{esc})$', ax=axes[1][0], shrink=(0.6, 0.8)[c])
axes[1][0].set_xlabel("$Log_{10}$(X)")
axes[1][0].set_ylabel("$Log_{10}$(Y)")

# plots a 2d histogram of x against y where the colour of the bin is dictated by it's mean n_esc
hist, xedges, yedges = np.histogram2d(x, y, bins=nbins, weights=n_esc)
hist_norm, xedges, yedges = np.histogram2d(x, y, bins=nbins)
hist = hist / hist_norm
hist = np.log10(hist)
hist = hist.T # transposes the histogram for proper orientation
color_hist = axes[1][1].imshow(hist, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                            origin='lower', aspect='auto', cmap='jet', interpolation='nearest')
plt.colorbar(color_hist, label='$Log_{10}$(Rate of LyC photon escape)', 
             ax=axes[1][1], shrink=(0.6, 0.8)[c])
axes[1][1].set_xlabel("$Log_{10}$(X)")
axes[1][1].set_ylabel("$Log_{10}$(Y)")

for axs in axes:
    for ax in axs:
        ax.set_box_aspect(1)

plt.tight_layout()
plt.show()