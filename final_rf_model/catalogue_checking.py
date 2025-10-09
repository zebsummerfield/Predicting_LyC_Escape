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
    obvs_star_mass = np.array(obvs_data['logmstar'])
    obvs_ha_flux = np.array(obvs_data['log_Ha_flux'])
    obvs_uv_flux = np.log10(np.array(obvs_data['UV_flux']))
    obvs_sfr10 = np.array(obvs_data['logSFR10'])
    obvs_sfr100 = np.array(obvs_data['logSFR100'])
    obvs_ha_size = np.array(obvs_data['r_eff_Ha_kpc'])
    obvs_uv_size = np.array(obvs_data['r_eff_UV_kpc'])
    print(np.mean(obvs_uv_flux - obvs_ha_flux))

    # converts the fluxes to magnitudes
    obvs_muv_app = -2.5 * obvs_uv_flux - 48.6
    obvs_mha_app = -2.5 * obvs_ha_flux - 48.6
    dist_mod = cosmo.distmod(obvs_redshift)
    obvs_uv_mag = np.array(obvs_muv_app - dist_mod.value)
    obvs_ha_mag = np.array(obvs_mha_app - dist_mod.value)

    obvs_vars = [obvs_redshift, obvs_star_mass, obvs_uv_mag, obvs_ha_mag,
                 obvs_sfr10, obvs_sfr100, obvs_ha_size, obvs_uv_size]   

with h5py.File('cat.hdf5', 'r') as hdf:
    f_esc = np.log10(np.array(hdf['f_esc_vir_full']))
    n_esc = np.log10(np.array(hdf['Ndot_LyC_vir_full']))
    
    redshift = np.array(hdf['redshift_full'])
    star_mass = np.log10(np.array(hdf['stellar_mass_full']))
    ha_lum = np.array(hdf['ha_lum_int_full']) * 5
    uv_lum = np.array(hdf['uv_lum_int_full'])
    sfr10 = np.log10(np.array(hdf['sfr_full_10']))
    sfr100 = np.log10(np.array(hdf['sfr_full_100']))
    ha_size = np.array(hdf['ha_size_int_2d_full'])
    uv_size = np.array(hdf['uv_size_int_2d_full'])
    import pdb;pdb.set_trace()


    # converts the luminosities to magnitudes
    lum_to_tenpc = 4 * np.pi * (10 * 3.086e18)**2
    uv_mag = -2.5 * np.log10((uv_lum) / lum_to_tenpc) - 48.6
    ha_mag = -2.5 * np.log10((ha_lum) / lum_to_tenpc) - 48.6

    vars = [redshift, star_mass, uv_mag, ha_mag,
            sfr10, sfr100, ha_size, uv_size]

    # removes any rows that have zero ,unity or infinity for the vars and f_esc
    f_esc[np.isnan(f_esc)] = 0
    for i in range(len(np.concatenate((vars, [f_esc])))):
        nan_indices = [index for index, val in enumerate(list(np.concatenate((vars, [f_esc]))[i]))
                       if (val == 0 or val == 1 or val == np.inf or val== -np.inf or np.isnan(val))][::-1]
        print(f"rows deleted: {len(nan_indices)}")
        f_esc, n_esc = (np.delete(f_esc, nan_indices), np.delete(n_esc, nan_indices))
        vars = np.delete(vars, nan_indices, axis=1)

    print([np.mean(var) for var in obvs_vars]) 
    print([np.mean(var) for var in vars])
    