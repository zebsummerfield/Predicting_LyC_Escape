import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import joblib
from functions_old import *

folder = "final_rf_model/"
file = 'cat.hdf5'

# 0 for f_esc, 1 for n_esc
f_or_n = 1

import numpy as np
from astropy.io import fits

with fits.open("prosp_properties_GOODSN.fits") as hdul1:
    data1 = {name: hdul1[1].data[name] for name in hdul1[1].data.columns.names}
with fits.open("prosp_properties_GOODSS.fits") as hdul2:
    data2 = {name: hdul2[1].data[name] for name in hdul2[1].data.columns.names}
# with fits.open("masked_objects_gn.fits") as hdul1:
#     data1 = {name: hdul1[1].data[name] for name in hdul1[1].data.columns.names}
# with fits.open("masked_objects_gs.fits") as hdul2:
#     data2 = {name: hdul2[1].data[name] for name in hdul2[1].data.columns.names}
obvs_data = {key: np.concatenate([data1[key], data2[key]]) for key in data1}
print(obvs_data.keys())

import pdb; pdb.set_trace()

obvs_redshift = obvs_data['z']
obvs_star_mass = 10**(np.array(obvs_data['log(Mstar)']).astype('float64'))
obvs_sfr10 = np.array(obvs_data['SFR10'])
obvs_sfr100 = np.array(obvs_data['SFR100'])
obvs_uv_mag = np.array(obvs_data['M1500int'])

obvs_ssfr10 = ssfr_func(obvs_sfr10, obvs_star_mass)
obvs_ssfr100 = ssfr_func(obvs_sfr100, obvs_star_mass)
log10_offset10 = np.log10(obvs_ssfr10) - sfms_func(np.array([obvs_redshift, np.log10(obvs_star_mass)]), s[0], b[0], u[0])

random_variable = np.random.uniform(low=0, high=1, size=(len(obvs_redshift)))

obvs_f_esc_vars = [obvs_ssfr10, obvs_ssfr100, obvs_ssfr10/obvs_ssfr100, obvs_star_mass, random_variable, 1+obvs_redshift, random_variable]
obvs_f_esc_keys = ['offset10', 'ssfr100', 'ssfr10/ssfr100', 'star_mass', 'uv_mag', '1 + redshift', 'random_variable']
obvs_n_esc_vars = [obvs_sfr10, obvs_sfr100, obvs_star_mass, random_variable, 1+obvs_redshift, random_variable]
obvs_n_esc_keys = ['sfr10', 'sfr100', 'star_mass', 'uv_mag', '1 + redshift', 'random_variable']

obvs_log_f_esc_vars = np.log10(obvs_f_esc_vars).astype('float32')
obvs_log_n_esc_vars = np.log10(obvs_n_esc_vars).astype('float32')

# replaces ssfr10 with offset from the star forming main sequence over 10Myrs
obvs_log_f_esc_vars[0] = log10_offset10.astype('float32')
# replaces the luminosities with magnitudes
obvs_log_f_esc_vars[4] = obvs_uv_mag.astype('float32')
obvs_log_n_esc_vars[3] = obvs_uv_mag.astype('float32')

obvs_vars = [obvs_f_esc_vars, obvs_n_esc_vars][f_or_n]
obvs_log_vars = [obvs_log_f_esc_vars, obvs_log_n_esc_vars][f_or_n]
obvs_keys = [obvs_f_esc_keys, obvs_n_esc_keys][f_or_n]
print(obvs_keys)

# removes any rows that have zero, nan or infinity for the vars
bad_indices = []
print(f"SN(F44W) < 3 rows: {len(bad_indices)}")
for i in range(len(obvs_vars)):
    b_i = [index for index, val in enumerate(list((obvs_vars)[i]))
                    if (val == 0 or val == np.inf or val== -np.inf or val == np.nan)]
    print(f"feature {i+1} bad rows: {len(b_i)}")
    bad_indices += b_i
bad_indices = list(set(bad_indices))[::-1]
obvs_vars = np.delete(obvs_vars, bad_indices, axis=1)
obvs_log_vars = np.delete(obvs_log_vars, bad_indices, axis=1)
print(f'rows remaining: {len(obvs_vars[i])}')
import pdb; pdb.set_trace()

# each row contains the data for a single galaxy
X = np.transpose(obvs_log_vars)


f_or_n_str = ['f_esc', 'n_esc'][f_or_n]
loaded_model = joblib.load(folder+f'{f_or_n_str}_rf_observational_charlotte_model.pkl')
predictions = loaded_model.predict(X)

mpl.rcParams.update({'font.size': 20})
fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(obvs_log_vars[(3, 2)[f_or_n]], predictions)
ax.set_xlabel("Log$_{10}$ ($M_*$)")
ax.set_ylabel(["$f_{esc}$", "$n_{ion,esc}$"][f_or_n])
fig.tight_layout()
plt.show()