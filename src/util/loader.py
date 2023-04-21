import os.path
import h5py
import numpy as np
import pandas as pd
import scipy


def load_mat(path):
    with h5py.File(path, 'r') as mat:
        spectral_image = np.array(mat['cube']).T
        bands = None
        if 'bands' in mat:
            bands = np.array(mat['bands']).squeeze()
    return spectral_image, bands

def load_observer(wavelengths, interpolation_method="linear"):
    observer_data = pd.read_csv('./res/observer/CIE_xyz_1931_2deg.csv', index_col=0,
                                header=None)
    interpolation_function = scipy.interpolate.interp1d(
        observer_data.index.values, observer_data.values, axis=0,
        kind=interpolation_method, fill_value=0, bounds_error=False)
    observer = interpolation_function(wavelengths)

    return observer.transpose() # return (3,31) array, same as bands

def load_data(directory):
    spectral_pixels, wavelengths = (np.empty((0, 31)), [])
    for file in os.listdir(directory):
        spectral_image, wavelengths = load_mat(os.path.join(directory, file))
        flat_spectral_image = spectral_image.reshape(-1, spectral_image.shape[-1])
        spectral_pixels = np.vstack((spectral_pixels, flat_spectral_image))

    return spectral_pixels, wavelengths
