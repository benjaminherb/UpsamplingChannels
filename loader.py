import os.path
import h5py
import numpy as np


def load_mat(path):
    with h5py.File(path, 'r') as mat:
        spectral_image = np.array(mat['cube']).T
        bands = None
        if 'bands' in mat:
            bands = np.array(mat['bands']).squeeze()
    return spectral_image, bands
