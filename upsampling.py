import os
from datetime import datetime

import numpy as np
from tensorflow import keras
import tensorflow as tf
from loader import load_mat
import matplotlib.pyplot as plt
import scipy


def main():
    if not os.path.isdir("./out"):
        os.mkdir("./out")
    output_directory = os.path.join("./out", datetime.now().strftime("%y%m%d_%H%M%S"))
    os.mkdir(os.path.join(output_directory))

    spectral_pixels, wavelengths = load_data("./res/train/")
    np.random.shuffle(spectral_pixels) # reorder to get a diverse test/validate set

    band_infos = [
        {'wavelength': 425, 'sigma': 30},
        {'wavelength': 475, 'sigma': 30},
        {'wavelength': 525, 'sigma': 30},
        {'wavelength': 575, 'sigma': 30},
        {'wavelength': 625, 'sigma': 30},
        {'wavelength': 675, 'sigma': 30},
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
    test_spectral_pixels = spectral_pixels[-100:, :]
    test_resampled_pixels = resampled_pixels[-100:, :]
    spectral_pixels = spectral_pixels[:-100, :]
    resampled_pixels = resampled_pixels[:-100, :]

    input_layer = keras.layers.Input(shape=(6,))
    hidden_layer1 = keras.layers.Dense(32, activation='relu')(input_layer)
    hidden_layer2 = keras.layers.Dense(64, activation='relu')(hidden_layer1)
    output_layer = keras.layers.Dense(31, activation='linear')(hidden_layer2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    optimizer = keras.optimizers.Adam(learning_rate=0.1)
    model.compile(optimizer=optimizer, loss=loss_function,
                  metrics=['MeanSquaredError', 'RootMeanSquaredError'])

    model.fit(resampled_pixels, spectral_pixels, validation_split=0.2, epochs=1, batch_size=64)
    model.evaluate(test_resampled_pixels, test_spectral_pixels)

    for pos in [0, 25, 50, 75, 99]:
        prediction = model.predict(test_resampled_pixels[pos, :].reshape(-1, 6)).flatten()
        ground_truth = test_spectral_pixels[pos, :].squeeze()
        plt.figure()
        plt.plot(wavelengths, ground_truth, label='Ground Truth')
        plt.plot(wavelengths, prediction, label='Prediction')
        plt.legend()
        plt.savefig(os.path.join(
            output_directory,
            f"test_prediction_{pos}_{np.mean(tf.square(ground_truth - prediction)):.6f}.png"))


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


def shuffle_along_axis(array, axis):
    indices = np.random.rand(*array.shape).argsort(axis=axis)
    return np.take_along_axis(array, indices, axis=axis)


if __name__ == "__main__":
    main()
