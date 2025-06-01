import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import joblib
from final_model.functions_old import *

folder = "final_model/"
file = 'cat.hdf5'

# 0 for f_esc, 1 for n_esc
f_or_n = 0

import h5py
import numpy as np
from astropy.cosmology import Planck15 as cosmo

with open('lola_cat.txt', 'r') as f:
    # loads the observational data from the text file and converts it to a dictionary
    lines = f.readlines()
    obvs_keys = lines[0].strip().split()
    print(obvs_keys)
    values = []
    for line in lines[1:]:
        values.append(line.strip().split())
    obvs_data = dict(zip(obvs_keys, np.array(values).T.astype(float)))

    obvs_redshift = np.array(obvs_data['z_spec'])
    print(len(obvs_redshift))
    obvs_star_mass = 10**np.array(obvs_data['logmstar'])
    obvs_ha_flux = 10**np.array(obvs_data['log_Ha_flux'])
    obvs_uv_flux = np.array(obvs_data['UV_flux'])
    obvs_sfr10 = 10**np.array(obvs_data['logSFR10'])
    obvs_sfr100 = 10**np.array(obvs_data['logSFR100'])
    obvs_ha_size = np.array(obvs_data['r_eff_Ha_kpc'])
    obvs_uv_size = np.array(obvs_data['r_eff_UV_kpc'])

    obvs_ssfr10 = ssfr_func(obvs_sfr10, obvs_star_mass)
    obvs_ssfr100 = ssfr_func(obvs_sfr100, obvs_star_mass)
    log10_offset10 = np.log10(obvs_ssfr10) - sfms_func(np.array([obvs_redshift, np.log10(obvs_star_mass)]), s[0], b[0], u[0])

    # converts the fluxes to magnitudes
    obvs_muv_app = -2.5 * obvs_uv_flux - 48.6
    obvs_mha_app = -2.5 * obvs_ha_flux - 48.6
    dist_mod = cosmo.distmod(obvs_redshift)
    obvs_uv_mag = np.array(obvs_muv_app - dist_mod.value)
    obvs_ha_mag = np.array(obvs_mha_app - dist_mod.value)

    random_variable = np.random.uniform(low=0, high=1, size=(len(obvs_redshift)))

    obvs_f_esc_vars = [obvs_ssfr10, obvs_ssfr10/obvs_ssfr100, obvs_star_mass, obvs_uv_flux,
                       obvs_uv_flux/obvs_ha_flux, obvs_uv_size, obvs_ha_size, 1+obvs_redshift, random_variable]
    obvs_f_esc_keys = ['offset10', 'ssfr10/ssfr100', 'star_mass', 'uv_mag',
                       'uv_flux/ha_flux', 'uv_size', 'ha_size', '1 + redshift', 'random_variable']
    obvs_n_esc_vars = [obvs_sfr10, obvs_sfr100, obvs_star_mass, obvs_uv_flux, obvs_ha_flux,
                       1+obvs_redshift, random_variable]
    obvs_n_esc_keys = ['sfr10', 'sfr100', 'star_mass', 'uv_mag', 'ha_mag',
                       '1 + redshift', 'random_variable']
    
    obvs_log_f_esc_vars = np.log10(obvs_f_esc_vars).astype('float32')
    obvs_log_n_esc_vars = np.log10(obvs_n_esc_vars).astype('float32')
    
    # replaces ssfr10 with offset from the star forming main sequence over 10Myrs
    obvs_log_f_esc_vars[0] = log10_offset10.astype('float32')
    # replaces the luminosities with magnitudes
    obvs_log_f_esc_vars[3] = obvs_uv_mag.astype('float32')
    obvs_log_n_esc_vars[2] = obvs_uv_mag.astype('float32')
    obvs_log_n_esc_vars[3] = obvs_ha_mag.astype('float32')

    obvs_vars = [obvs_f_esc_vars, obvs_n_esc_vars][f_or_n]
    obvs_log_vars = [obvs_log_f_esc_vars, obvs_log_n_esc_vars][f_or_n]
    obvs_keys = [obvs_f_esc_keys, obvs_n_esc_keys][f_or_n]
    print(obvs_keys)

    # each row contains the data for a single galaxy
    X = np.transpose(obvs_log_vars)

f_or_n_str = ['f_esc', 'n_esc'][f_or_n]
loaded_model = joblib.load(folder+f'{f_or_n_str}_rf_observational_model.pkl')
predictions = loaded_model.predict(X)

mpl.rcParams.update({'font.size': 16})
fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(np.log10(obvs_star_mass), predictions)
fig.tight_layout()
plt.show()