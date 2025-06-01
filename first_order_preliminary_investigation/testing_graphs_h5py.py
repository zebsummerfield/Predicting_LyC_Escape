import h5py
import matplotlib.pyplot as plt
import numpy as np

folder = "first_order_testing/"
file1 = folder + "halo_188.hdf5"
file2 = folder + "halo-Ha-RHD_188.hdf5"
file3 = folder + "halo-ion-eq-RHD_188.hdf5"
file4 = folder + "halo-M1500-RHD_188.hdf5"
file5 = folder + "candidates_188.hdf5"

esc_dict = {}

with h5py.File(file3, 'r') as hdf:
    f_escs = hdf['subhalo']['f_esc'][:]
    esc_ids = hdf['subhalo']['id'][:]
    esc_dict = {id: esc for id, esc in zip(esc_ids, f_escs)}
    # print(esc_dict.keys())

with h5py.File(file5, 'r') as hdf:
    candidate_values = []
    for var in ['M_vir', 'SubhaloVmaxRad', 'SubhaloGasMetallicityMaxRad', 'SubhaloGasDustCarbonSiliconRatioMaxRad']:
        candidate_ids = hdf['Subhalo']['SubhaloID'][:]
        esc_indices = [index for index, id in enumerate(candidate_ids) if id in esc_ids]
        candidate = hdf['Subhalo'][var][:]
        candidate_values.append([candidate[index] for index in esc_indices])


    candidate_dict = {id: [a, b, c, d] for id, a, b, c, d in zip(esc_ids, 
        candidate_values[0], 
        candidate_values[1], 
        candidate_values[2],
        candidate_values[3])}

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        
axes[0, 0].scatter([can[0] for can in candidate_dict.values()], esc_dict.values(), s=2)
axes[0, 0].set_title("Virial Mass")

axes[0, 1].scatter([can[1] for can in candidate_dict.values()], esc_dict.values(), s=2)
axes[0, 1].set_title("Comoving Radius of Rotation Curve Maximum")

axes[1, 0].scatter([can[2] for can in candidate_dict.values()], esc_dict.values(), s=2)
axes[1, 0].set_title("Mass-Weighted Average Metallicity")

axes[1, 1].scatter([can[3] for can in candidate_dict.values()],esc_dict.values(), s=2)
axes[1, 1].set_title("Dust Carbon to Silicon Ratio")

plt.show()
