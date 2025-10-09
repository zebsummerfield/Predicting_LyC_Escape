import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from functions_old import *

c = 0

folder = "investigating_indicators/"
file = 'cat.hdf5'

with h5py.File(file, 'r') as hdf:
    with open(folder+"keys.txt", "w") as k:
        for key in hdf.keys():
            k.write(key + "\n")
    print(hdf.keys())

    f_esc = np.array(hdf['f_esc_vir_full'])
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
    uv_density = uv_lum / (uv_projection**2)
    uv_obs = np.array(hdf['uv_lum_obs_full'])

    vars = np.array([ssfr10, ssfr50, ssfr100, redshift, star_mass, gas_mass, star_mass/gas_mass, 
                     uv_lum, ha_lum, uv_lum/ha_lum, uv_lum*ha_lum, gas_met, star_met, gas_met*star_met,
                     sfr_size, star_size, sfr_size/star_size, uv_size, ha_size, uv_size/ha_size,
                     1/uv_density, uv_obs/uv_lum])

    #variable we are investigating for an f_esc relationship
    x = ssfr10/ssfr100

    # removes any rows that have zero, nan or infinity for the x, ssfr10, f_esc and n_esc
    for i in range(len(np.concatenate(([x], [vars[1]], [f_esc, n_esc])))):
        nan_indices = [index for index, val in enumerate(list(np.concatenate(([x], [vars[1]], [f_esc, n_esc]))[i]))
                       if (val==0 or val == np.inf or val== -np.inf or np.isnan(val))][::-1]
        print(f"rows deleted: {len(nan_indices)}")
        f_esc, n_esc = (np.delete(f_esc, nan_indices), np.delete(n_esc, nan_indices))
        x = np.delete(x, nan_indices)
        vars = np.delete(vars, nan_indices, axis=1)
        redshift, star_mass = (np.delete(redshift, nan_indices), np.delete(star_mass, nan_indices))

    log_vars = np.log10(vars).astype('float32')
    log_f_esc = np.log10(f_esc).astype('float32')
    log_n_esc = np.log10(n_esc).astype('float32')

    # calculates the offset from the star-forming main sequence
    log_sfms10 =  log_vars[0] - sfms_func(np.array([redshift, np.log10(star_mass)]), s[0], b[0], u[0])
    log_sfms100 =  log_vars[2] - sfms_func(np.array([redshift, np.log10(star_mass)]), s[1], b[1], u[1])

    log_x = np.log10(x).astype('float32') 

matplotlib.rcParams.update({'font.size': (18, 10)[c]})
fig, axes = plt.subplots(1, 2, figsize=((16, 8), (12, 6))[c])

# plots a 2d histogram of log_x against log_f_esc where the number of galaxies in a bin dictates it's colour
nbins = 100
hist, xedges, yedges = np.histogram2d(log_x, log_f_esc, bins=nbins)
hist = hist.T
hist = np.log10(hist)
h1 = axes[0].imshow(hist, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                  origin='lower', aspect='auto', cmap='viridis', interpolation='nearest')
cbar = fig.colorbar(h1, label="Number of Galaxies in Bin", ax=axes[0], shrink=(0.6, 0.8)[c])
axes[0].set_xlabel("$Log_{10}$(X)")
axes[0].set_ylabel("$Log_{10}(f_{esc})$")

# plots a 2d histogram of log_x against log_n_esc where the number of galaxies in a bin dictates it's colour
hist, xedges, yedges = np.histogram2d(log_x, log_n_esc, bins=nbins)
hist = hist.T
hist = np.log10(hist)
h2 = axes[1].imshow(hist, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                  origin='lower', aspect='auto', cmap='viridis', interpolation='nearest')
cbar = fig.colorbar(h2, label="Number of Galaxies in Bin", ax=axes[1], shrink=(0.6, 0.8)[c])
axes[1].set_xlabel("$Log_{10}$(X)")
axes[1].set_ylabel("$Log_{10}$(Rate of LyC photon escape)")

# seperates the galaxies into bins of variable log_x with each containing equal numbers of galaxies 
nbins = 100
bins = np.quantile(log_x, np.linspace(0, 1, nbins + 1))
bin_indices = np.digitize(log_x, bins)
x_medians, f_esc_medians, n_esc_medians = ([], [], [])
for i in range(1, len(bins)):
    bin_mask = bin_indices == i
    x_medians.append(np.median(log_x[bin_mask]))
    f_esc_medians.append(np.median(log_f_esc[bin_mask]))
    n_esc_medians.append(np.median(log_n_esc[bin_mask]))

# plots the median of log_x against the median of log_f_esc for each bin
axes[0].plot(x_medians, f_esc_medians, c='r', linewidth=3)
axes[0].set_xlim(min(x_medians), max(x_medians))
axes[0].set_ylim(-5, 0)

# plots the median of log_x against the median of log_n_esc for each bin
axes[1].plot(x_medians, n_esc_medians, c='r', linewidth=3)
axes[1].set_xlim(min(x_medians), max(x_medians))
axes[1].set_ylim(46, 54)

for ax in axes:
    ax.set_box_aspect(1)

plt.tight_layout()
plt.show()