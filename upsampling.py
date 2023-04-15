import os
import numpy as np
# from tensorflow import keras
# import tensorflow as tf
from loader import load_mat
import matplotlib.pyplot as plt
import scipy

MINIMUM_WAVELENGTH = 400
MAXIMUM_WAVELENGTH = 700


def main():
    spectral_pixels = load_data("./res/train/")
    band_infos = [
        {'wavelength': 400, 'sigma': 15},
        {'wavelength': 480, 'sigma': 20},
        {'wavelength': 530, 'sigma': 50},
        {'wavelength': 610, 'sigma': 8},
        {'wavelength': 660, 'sigma': 30},
        {'wavelength': 710, 'sigma': 17},
    ]

    bands, wavelenghts = get_bands(band_infos)
    for band in bands:
        plt.plot(wavelenghts, band)
    plt.xlabel('Wavelength')
    plt.ylabel('Response')

    plt.show()


def load_data(directory) -> list:
    data = []
    for file in os.listdir(directory):
        spectral_image, bands = load_mat(os.path.join(directory, file))
        flat_spectral_image = spectral_image.reshape(-1, spectral_image.shape[-1])
        spectral_pixels = np.split(flat_spectral_image, flat_spectral_image.shape[0])
        data.extend(spectral_pixels)

    return data


def convert_to_selected_bands(spectral_pixel, bands):
    pass


def get_band(center_wavelength, sigma):
    wavelenghts = np.linspace(MINIMUM_WAVELENGTH,
                              MAXIMUM_WAVELENGTH,
                              MAXIMUM_WAVELENGTH - MINIMUM_WAVELENGTH)
    values = np.exp(-(wavelenghts - center_wavelength) ** 2 / (2 * sigma ** 2)) / (
                sigma * np.sqrt(2 * np.pi))
    values = values / values.max()
    return values, wavelenghts


def get_bands(band_infos):
    bands = []
    wavelengths = []
    for band_info in band_infos:
        band, wavelengths = get_band(band_info['wavelength'], band_info['sigma'])
        bands.append(band)
    return bands, wavelengths


if __name__ == "__main__":
    main()
