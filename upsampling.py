import os
from datetime import datetime

import numpy as np
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy

from loader import load_mat, load_observer


def main():
    if not os.path.isdir("./out"):
        os.mkdir("./out")
    output_directory = os.path.join("./out", datetime.now().strftime("%y%m%d_%H%M%S"))
    os.mkdir(os.path.join(output_directory))

    spectral_pixels, wavelengths = load_data("./res/train/")
    np.random.shuffle(spectral_pixels)  # reorder to get a diverse test/validate set

    band_infos_six = [
        {'wavelength': 425, 'sigma': 30},
        {'wavelength': 475, 'sigma': 30},
        {'wavelength': 525, 'sigma': 30},
        {'wavelength': 575, 'sigma': 30},
        {'wavelength': 625, 'sigma': 30},
        {'wavelength': 675, 'sigma': 30},
    ]

    band_infos_five = [
        {'wavelength': 420, 'sigma': 30},
        {'wavelength': 485, 'sigma': 30},
        {'wavelength': 550, 'sigma': 30},
        {'wavelength': 615, 'sigma': 30},
        {'wavelength': 680, 'sigma': 30},
    ]
    band_infos_four = [
        {'wavelength': 430, 'sigma': 30},
        {'wavelength': 510, 'sigma': 30},
        {'wavelength': 590, 'sigma': 30},
        {'wavelength': 670, 'sigma': 30},
    ]
    band_infos_three = [
        {'wavelength': 450, 'sigma': 30},
        {'wavelength': 550, 'sigma': 30},
        {'wavelength': 650, 'sigma': 30},
    ]

    band_infos = band_infos_three
    band_count = len(band_infos)

    bands = get_bands(band_infos, wavelengths)
    for band in bands:
        plt.plot(wavelengths, band)
    plt.xlabel('Wavelength')
    plt.ylabel('Response')
    plt.savefig(f"camera_response_{band_count}_bands.png")

    resampled_pixels = resample(spectral_pixels, bands)
    resampled_wavelengths = [band_info['wavelength'] for band_info in band_infos]

    spectral_pixels = spectral_pixels[:-100, :]
    resampled_pixels = resampled_pixels[:-100, :]

    input_layer = keras.layers.Input(shape=(band_count,))
    hidden_layer1 = keras.layers.Dense(32, activation='relu')(input_layer)
    hidden_layer2 = keras.layers.Dense(64, activation='relu')(hidden_layer1)
    output_layer = keras.layers.Dense(31, activation='linear')(hidden_layer2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=loss_function,
                  metrics=['RootMeanSquaredError'])

    model.fit(resampled_pixels, spectral_pixels, validation_split=0.2, epochs=10, batch_size=1024)

    # load test images
    test_spectral_pixels, wavelengths = load_data("./res/test")
    test_resampled_pixels = resample(test_spectral_pixels, bands)

    model.evaluate(test_resampled_pixels, test_spectral_pixels)

    for pos in [0, 100, 1000, 10000, 100000]:
        prediction = model.predict(test_resampled_pixels[pos, :].reshape(-1, band_count)).flatten()
        ground_truth = test_spectral_pixels[pos, :].squeeze()
        plt.figure()
        plt.plot(wavelengths, ground_truth, label='Ground Truth')
        plt.plot(wavelengths, prediction, label='Prediction')
        plt.legend()
        plt.savefig(os.path.join(
            output_directory,
            f"test_prediction_{band_count}_bands_{pos}_{np.mean(tf.square(ground_truth - prediction)):.8f}.png"))

    test_spectral_image, wavelengths = load_mat("./res/test/ARAD_1K_0059.mat")
    test_spectral_image_flat = test_spectral_image.reshape(-1, test_spectral_image.shape[-1])
    test_ground_truth_rgb_image_flat = resample(test_spectral_image_flat,
                                                load_observer(wavelengths))
    test_ground_truth_rgb_image = test_ground_truth_rgb_image_flat.reshape(
        (test_spectral_image.shape[0], test_spectral_image.shape[1], 3))
    plt.imsave("test_ground_truth_rgb_image.png",
               test_ground_truth_rgb_image / test_ground_truth_rgb_image.max())

    # Prediction
    test_resampled_image_flat = resample(test_spectral_image_flat, bands)
    test_predicted_spectral_image_flat = model.predict(test_resampled_image_flat)
    test_predicted_rgb_image_flat = resample(test_predicted_spectral_image_flat, load_observer(wavelengths))
    test_predicted_rgb_image = test_predicted_rgb_image_flat.reshape(
        (test_spectral_image.shape[0], test_spectral_image.shape[1], 3))
    plt.imsave("test_predicted_rgb_image.png",
               test_predicted_rgb_image / test_predicted_rgb_image.max())



def loss_function(ground_truth, prediction):
    squared_difference = tf.square(ground_truth - prediction)
    return tf.reduce_mean(squared_difference, axis=-1)


def load_data(directory):
    spectral_pixels, wavelengths = (np.empty((0, 31)), [])
    for file in os.listdir(directory):
        print(file)
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


if __name__ == "__main__":
    main()
