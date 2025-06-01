import h5py
import matplotlib.pyplot as plt
import numpy as np

folder = "first_order_testing/"
file1 = folder + "halo_188.hdf5"
file2 = folder + "halo-Ha-RHD_188.hdf5"
file3 = folder + "halo-ion-eq-RHD_188.hdf5"
file4 = folder + "halo-M1500-RHD_188.hdf5"
file5 = folder + "candidates_188.hdf5"

with h5py.File(file3, 'r') as hdf:
    keys = list(hdf.keys())
    for k in keys:
        data = hdf[k]
        print(f"{k} : {len(data)}")
    print(hdf["subhalo"].keys())
    f_escs = hdf["subhalo"]["f_esc"][:]
    print(len(f_escs))
    print(list(f_escs)[0:10])
    import pdb
    pdb.set_trace()

with h5py.File(file2, 'r') as hdf:
    keys = list(hdf.keys())
    luminosities = hdf['subhalo']['L'][:]
    print(len(luminosities))
    print(luminosities[0:10])
    print(sum(luminosities)/832)

