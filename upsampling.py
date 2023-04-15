import os
import numpy as np
# from tensorflow import keras
# import tensorflow as tf
from loader import load_mat
import matplotlib.pyplot as plt
import scipy


def main():
    spectral_pixels = load_data("./res/train/")


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


def get_band(wavelength, sigma):
    x = np.linspace(-200 + wavelength, 200 + wavelength, 400)
    y = np.exp(-(x - wavelength) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    y = y / y.max()
    return np.stack((x, y), axis=-1)


if __name__ == "__main__":
    band1 = get_band(800, 2)
    band2 = get_band(700, 10)
    band3 = get_band(600, 30)
    band4 = get_band(500, 20)
    plt.plot(band1[:, 0], band1[:, 1])
    plt.plot(band2[:, 0], band2[:, 1])
    plt.plot(band3[:, 0], band3[:, 1])
    plt.plot(band4[:, 0], band4[:, 1])
    plt.xlabel('Wavelength')
    plt.ylabel('Response')

    plt.show()
