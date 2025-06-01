import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from functions import *

folder = "final_graph_generation/"
file = 'cat.hdf5'

# if c = 0 then small graphs are plotted, if c = 1 then large graphs are plotted
c = 0

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

    vars = np.array([ssfr10, vir_mass, ssfr50, random_variable])
    
    # adds a small epsilon to the variables to avoid log(0) errors
    epsilons = np.array([min([v for v in var if v !=0])/10 for var in vars])
    eps_array = np.zeros(vars.shape)
    for i in range(len(eps_array)):
        eps_array[i].fill(epsilons[i])
    vars = vars + eps_array

    # removes any rows that have zero ,unity or infinity for the vars and f_esc
    # ssfr50 is included to remove galaxies that have had no recent star formation
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
    # log_sfms10 =  log_vars[0] - sfms_func(np.array([redshift, np.log10(star_mass)]), s[0], b[0], u[0])
    # log_vars[0] = log_sfms10.astype('float32')
    log_f_esc = np.log10(f_esc).astype('float32')
    log_n_esc = np.log10(n_esc).astype('float32')

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


colors = [
    '#1b9e77',  # teal green
    '#d95f02',  # orange
]
plt.style.use('./MNRAS_Style.mplstyle')
mpl.rcParams.update({'font.size': 20})
fig = plt.figure(figsize=(16, 8))
gs = mpl.gridspec.GridSpec(1, 2, wspace=0)
ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1], sharey=ax0)
axes = [ax0, ax1]

# here the distributions of variables in the filtered dataset are plotted in histograms
ax0.set_ylim(0,0.3)
range = (-8, 0)
ax0.hist(log_f_esc[accepted_samples], density=True, bins=100, alpha=0.8, color=colors[0])
ax0.set_xlim(range)
ax0.set_xlabel('$\mathrm{Log}_{10}$($f_{\mathrm{esc}}$)')
ax0.set_ylabel('Probability Density Distribution')

range = (43.5, 55.5)
ax1.hist(log_n_esc[accepted_samples], density=True, bins=100, alpha=0.8, color=colors[1])
ax1.set_xlim(range)
ax1.set_xlabel('$\mathrm{Log}_{10}$($N_{\mathrm{esc}}$)')
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