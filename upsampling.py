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
    plt.savefig("camera_response.png")

    resampled_pixels = resample(spectral_pixels, bands)
    resampled_wavelengths = [band_info['wavelength'] for band_info in band_infos]

    # split some for testing
    test_spectral_pixels = spectral_pixels[-100:,:]
    test_resampled_pixels = resampled_pixels[-100:,:]
    spectral_pixels = spectral_pixels[:-100,:]
    resampled_pixels = resampled_pixels[:-100,:]

    input_layer = keras.layers.Input(shape=(6,))
    hidden_layer1 = keras.layers.Dense(64, activation='relu')(input_layer)
    hidden_layer2 = keras.layers.Dense(64, activation='relu')(hidden_layer1)
    hidden_layer3 = keras.layers.Dense(31, activation='relu')(hidden_layer2)
    output_layer = keras.layers.Dense(31, activation='softmax')(hidden_layer3)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    optimizer = keras.optimizers.Adam(learning_rate=1.0)
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['MeanSquaredError', 'RootMeanSquaredError'])

    model.fit(resampled_pixels, spectral_pixels, validation_split=0.2, epochs=3, batch_size=64)
    model.evaluate(test_resampled_pixels, test_spectral_pixels)

    pos = 17
    prediction = model.predict(test_resampled_pixels[pos, :].reshape(-1,6))
    print(prediction)
    print(test_spectral_pixels[pos, :])
    print(prediction.flatten() - test_spectral_pixels[pos, :].squeeze())

    plt.figure()
    plt.plot(wavelengths, spectral_pixels[pos, :].squeeze(), label='Spectral')
    plt.plot(wavelengths, prediction.flatten(), label='Prediction')
    plt.legend()
    plt.savefig(f"test_{pos}.png")


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
