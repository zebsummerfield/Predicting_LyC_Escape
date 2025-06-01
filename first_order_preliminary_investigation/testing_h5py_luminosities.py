import h5py
import matplotlib.pyplot as plt
import numpy as np

import os
folder = "first_order_testing/"
file1 = folder + "halo_188.hdf5"
file2 = folder + "halo-Ha-RHD_188.hdf5"
file3 = folder + "halo-ion-eq-RHD_188.hdf5"
file4 = folder + "halo-M1500-RHD_188.hdf5"
file5 = folder + "candidates_188.hdf5"

with h5py.File(file3, 'r') as hdf:
    f_escs = hdf["subhalo"]["f_esc"][:]
    print(list(f_escs)[0:10])

with h5py.File(file2, 'r') as hdf:
    keys = list(hdf.keys())
    luminosities = hdf['subhalo']['L'][:]
    print(luminosities[0:10])
    print(sum(luminosities)/832)


luminosities = np.array(luminosities)
f_escs = np.array(f_escs)
mask = (luminosities < 1e38) & (luminosities > 1e30)
plt.scatter(luminosities[mask], f_escs[mask])
plt.show()