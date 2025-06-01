import h5py
import matplotlib.pyplot as plt
import numpy as np

def sfms_func(xdata, s, b, u):
    redshift = xdata[0,:]
    stellar_mass = 10**xdata[1,:]
    return np.log10(s * (stellar_mass / 1e10)**b * (1 + redshift)**u)

def ssfr_func(sfr, mass):
    return np.array(sfr) / np.array(mass) * 1e9

s = [0.033, 0.067]
b = [0.041, 0.042]
u = [2.64, 2.57]

folder = "random_forest_testing/"
file = 'cat.hdf5'

with h5py.File(file, 'r') as hdf:
    with open(folder+"keys.txt", "w") as k:
        for key in hdf.keys():
            k.write(key + "\n")
    print(hdf.keys())
    print(hdf['f_esc_vir_full'][0:10])
    
    f_esc = np.array(hdf['f_esc_vir_full']).astype('float32')
    redshift = np.array(hdf['redshift_full']).astype('float32')
    mass = np.array(hdf['stellar_mass_full'])
    ssfr10 = ssfr_func(hdf['sfr_full_10'], mass)
    ssfr50 = ssfr_func(hdf['sfr_full_50'], mass)
    ssfr100 = ssfr_func(hdf['sfr_full_100'], mass)
    vars = np.array([ssfr10, ssfr50, ssfr100, redshift, mass])

    # removes any rows that have nan for f_esc
    nans = np.isnan(f_esc)
    nan_indices = [index for index, b in enumerate(list(nans)) if b == True][::-1]
    print(len(nan_indices))
    f_esc = np.delete(f_esc, nan_indices, 0)
    vars = np.delete(vars, nan_indices, axis=1)
    
    # removes any rows that have zero for any vars
    for v in range(len(vars)):
        zero_indices = [index for index, val in enumerate(list(vars[v])) if val == 0][::-1]
        print(len(zero_indices))
        f_esc = np.delete(f_esc, zero_indices, 0)
        vars = np.delete(vars, zero_indices, axis=1)

    log_vars = np.log10(vars).astype('float32')
    print(log_vars.shape)

    # calculates the offset from the star-forming main sequence
    sfms10 =  log_vars[0] - sfms_func(np.array([vars[3], log_vars[4]]), s[0], b[0], u[0])
    sfms100 =  log_vars[2] - sfms_func(np.array([vars[3], log_vars[4]]), s[1], b[1], u[1])

plt.scatter(f_esc, sfms100, s=4, c='b', marker='.')
plt.xlabel("$f_{esc}$")
plt.ylabel("$Log_{10}$(offset From The Star-Forming Main Sequence)")
plt.show()