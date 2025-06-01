import h5py
import matplotlib.pyplot as plt
import numpy as np

folder = "first_order_testing/"
file1 = folder + "halo_188.hdf5"
file2 = folder + "halo-Ha-RHD_188.hdf5"
file3 = folder + "halo-ion-eq-RHD_188.hdf5"
file4 = folder + "halo-M1500-RHD_188.hdf5"
file5 = folder + "candidates_188.hdf5"

with h5py.File(file5, 'r') as hdf:
    keys = list(hdf.keys())
    for k in keys:
        data = hdf[k]
        print(f"{k} : {len(data)}")
    print(hdf["Subhalo"].keys())
    print(len(hdf['Subhalo']['SubhaloGasMetallicity']))
    print(hdf['Subhalo']['SubhaloGasMetallicity'][0:10])
    for k in hdf['Subhalo']:
        try:
            print(f"{k} : {len([val for val in hdf['Subhalo'][k] if val>0 and not val == 1])}")
        except:
            pass
            
