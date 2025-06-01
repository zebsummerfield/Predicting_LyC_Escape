import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from investigating_indicators.functions_old import *

folder = "investigating_indicators/"
file = 'cat.hdf5'

# if c = 0 then small graphs are plotted, if c = 1 then large graphs are plotted
c = 0

# 0 for f_esc, 1 for n_esc
f_or_n = 0

# loads the catalogue of galaxies and their variables
with h5py.File(file, 'r') as hdf:

    f_esc = np.array(hdf['f_esc_vir_full']).astype('float32')
    n_esc = np.array(hdf['Ndot_LyC_vir_full'])
    print(len(f_esc))

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
    uv_lum_density = uv_lum / (uv_projection**2)
    uv_obs = np.array(hdf['uv_lum_obs_full'])

    random_variable = np.random.uniform(low=0, high=1, size=(len(f_esc)))

    f_esc_vars = np.array([ssfr10, ssfr100, star_mass, gas_mass, gas_met, 
                     uv_lum, ha_lum, uv_size, ha_size, uv_lum_density,
                     sfr_size, star_size, (1+redshift), random_variable])
    f_esc_keys = np.array(['offset10', 'ssfr10/ssfr100', 'star_mass', 'gas_mass/star_mass', 'gas_met', 
                     'uv_lum', 'uv_lum/ha_lum', 'uv_size', 'ha_size', 'uv_lum_density', 
                     'sfr_size', 'sfr_size/star_size', '1 + redshift', 'random_variable'])
    n_esc_vars = np.array([ssfr100, star_mass, vir_mass, gas_mass, star_mass, gas_met, 
                           uv_lum, ha_lum, uv_size, uv_lum_density, 
                           star_size, (1+redshift), random_variable])
    n_esc_keys = np.array(['ssfr100', 'star_mass', 'vir_mass', 'gas_mass', 'gas_mass/star_mass', 'gas_met', 
                           'uv_lum', 'ha_lum', 'uv_size', 'uv_lum_density', 
                           'star_size', '1 + redshift', 'random_variable'])
    
    # adds a small epsilon to the variables to avoid log(0) errors
    f_epsilons = np.array([min([v for v in var if v !=0])/10 for var in f_esc_vars])
    eps_array = np.zeros(f_esc_vars.shape)
    for i in range(len(eps_array)):
        eps_array[i].fill(f_epsilons[i])
    f_esc_vars = f_esc_vars + eps_array
    n_epsilons = np.array([min([v for v in var if v !=0])/10 for var in n_esc_vars])
    eps_array = np.zeros(n_esc_vars.shape)
    for i in range(len(eps_array)):
        eps_array[i].fill(n_epsilons[i])
    n_esc_vars = n_esc_vars + eps_array

    f_esc_vars[1] = f_esc_vars[0] / f_esc_vars[1]
    f_esc_vars[3] = f_esc_vars[3] / f_esc_vars[2]
    f_esc_vars[6] = f_esc_vars[5] / f_esc_vars[6]
    f_esc_vars[11] = f_esc_vars[10] / f_esc_vars[11]
    n_esc_vars[4] = n_esc_vars[3] / n_esc_vars[1]

    vars = [f_esc_vars, n_esc_vars][f_or_n]
    keys = [f_esc_keys, n_esc_keys][f_or_n]

    # removes any rows that have zero ,unity or infinity for the vars and f_esc
    # ssfr100 is included to remove galaxies that have had no recent star formation
    # here indices to be deleted are added to a list instead for plotting
    all_nan_indices = []
    f_esc[np.isnan(f_esc)] = 0
    for i in range(len(np.concatenate((vars, [f_esc], [ssfr50])))):
        nan_indices = [index for index, val in enumerate(list(np.concatenate((vars, [f_esc], [ssfr50]))[i]))
                       if (val == 0 or val == 1 or val == np.inf or val == -np.inf)][::-1]
        all_nan_indices += nan_indices
        print(f"rows deleted: {len(nan_indices)}")
        f_esc, n_esc = (np.delete(f_esc, nan_indices), np.delete(n_esc, nan_indices))
        vars = np.delete(vars, nan_indices, axis=1)
        redshift, star_mass = (np.delete(redshift, nan_indices), np.delete(star_mass, nan_indices))
        ssfr10, ssfr50, ssfr100 = (np.delete(ssfr10, nan_indices), 
                                   np.delete(ssfr50, nan_indices),
                                   np.delete(ssfr100, nan_indices))
    all_nan_indices = list(set(all_nan_indices))
    print(f'rows remaining: {len(f_esc)}')

    log_vars = np.log10(vars).astype('float32')
    # replaces ssfr10 with Offset from the star forming main sequence over 10Myrs
    log_sfms10 =  log_vars[0] - sfms_func(np.array([redshift, np.log10(star_mass)]), s[0], b[0], u[0])
    #log_vars[0] = log_sfms10.astype('float32')
    log_f_esc = np.log10(f_esc).astype('float32')
    log_n_esc = np.log10(n_esc).astype('float32')

    # density_threshold = 1
    # hist, bin_edges = np.histogram(log_f_esc, bins=100, density=True)
    # binc_centres = 0.5*(bin_edges[1:] + bin_edges[:-1])
    # hist_clipped = np.clip(hist, density_threshold)

    # carries out rejection sampling to cap the density of the f_esc distribution
    mean = np.mean(log_f_esc)
    std = np.std(log_f_esc)
    density_threshold = 0.25 * stats.norm.pdf(mean, mean, std)
    accepted_samples = []
    for i in range(len(log_f_esc)):
        density = stats.norm.pdf(log_f_esc[i], mean, std)
        if np.random.random() < density_threshold/density:
            accepted_samples.append(i)
    print(f"rows after rejection sampling: {len(accepted_samples)}")
    
mpl.rcParams.update({'font.size': (16, 10)[c]})
fig, axes = plt.subplots(1, 2, figsize=((16, 8), (12, 6))[c])

# here the distributions of variables in the filtered dataset are plotted in histograms
range = (-8, 0)
axes[0].hist(log_f_esc[accepted_samples], density=True, bins=100, alpha=0.7, color='b')
axes[0].set_xlim(range)
axes[0].set_title('f_esc distribution')
axes[0].set_xlabel('$Log_{10}$($f_{esc}$)')
axes[0].set_ylabel('Density')

range = (43, 55)
axes[1].hist(log_n_esc[accepted_samples], density=True, bins=100, alpha=0.7, color='r')
axes[1].set_xlim(range)
axes[1].set_title('n_esc distribution')
axes[1].set_xlabel('$Log_{10}$($N_{esc}$)')
axes[1].set_ylabel('Density')

plt.show()