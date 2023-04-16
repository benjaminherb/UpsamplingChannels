import os

import numpy as np
from tensorflow import keras
import tensorflow as tf
from loader import load_mat
import matplotlib.pyplot as plt
import scipy


def main():
    spectral_pixels, wavelengths = load_data("./res/train/")

    band_infos = [
        {'wavelength': 400, 'sigma': 15},
        {'wavelength': 480, 'sigma': 20},
        {'wavelength': 530, 'sigma': 50},
        {'wavelength': 610, 'sigma': 8},
        {'wavelength': 660, 'sigma': 30},
        {'wavelength': 710, 'sigma': 17},
    ]

    bands = get_bands(band_infos, wavelengths)
    for band in bands:
        plt.plot(wavelengths, band)
    plt.xlabel('Wavelength')
    plt.ylabel('Response')

    plt.show()
    plt.plot(wavelengths, spectral_pixels[0].squeeze())
    resampled_pixels = resample(spectral_pixels, bands)
    resampled_wavelengths = [band_info['wavelength'] for band_info in band_infos]
    plt.plot(resampled_wavelengths, resampled_pixels[0])
    print(resample(np.ones(31), bands))
    plt.show()

    input_layer = keras.layers.Input(shape=(6,))
    hidden_layer1 = keras.layers.Dense(64, activation='relu')(input_layer)
    hidden_layer2 = keras.layers.Dense(64, activation='relu')(hidden_layer1)
    hidden_layer3 = keras.layers.Dense(31, activation='relu')(hidden_layer2)
    output_layer = keras.layers.Dense(31, activation='softmax')(hidden_layer3)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])

    model.fit(resampled_pixels, spectral_pixels, epochs=1, batch_size=64)

    prediction = model.predict(resampled_pixels[1000, :])
    print(prediction)
    print(spectral_pixels[1000, :])
    print(prediction - spectral_pixels[1000, :])

    plt.plot(wavelengths, spectral_pixels[1000, :].squeeze())
    plt.plot(wavelengths, prediction)
    plt.savefig("test.png")


def loss_function(ground_truth, prediction):
    squared_difference = tf.square(ground_truth - prediction)
    return tf.reduce_mean(squared_difference, axis=-1)


def load_data(directory):
    spectral_pixels, wavelengths = (np.empty((0, 31)), [])
    for file in os.listdir(directory):
        spectral_image, wavelengths = load_mat(os.path.join(directory, file))
        flat_spectral_image = spectral_image.reshape(-1, spectral_image.shape[-1])
        spectral_pixels = np.vstack((spectral_pixels, flat_spectral_image))

    return spectral_pixels, wavelengths


def get_band(center_wavelength, sigma, wavelengths):
    values = np.exp(-(wavelengths - center_wavelength) ** 2 / (2 * sigma ** 2)) / (
            sigma * np.sqrt(2 * np.pi))
    values = values  # / values.max()
    return values


def get_bands(band_infos, wavelengths):
    bands = []
    for band_info in band_infos:
        band = get_band(band_info['wavelength'], band_info['sigma'], wavelengths)
        bands.append(band)

    # scale all bands so the sum over all values is one
    bands = np.array(bands) \
            * (wavelengths[-1] - wavelengths[0]) / len(wavelengths)
    return bands


def resample(spectral_pixels, bands):
    # Assumes the bands and pixel bands match
    if len(spectral_pixels.shape) == 1:  # for single pixels
        return np.sum(spectral_pixels * bands, axis=1)
    return np.sum((spectral_pixels[:, np.newaxis, :] * bands), axis=2)


if __name__ == "__main__":
    main()
