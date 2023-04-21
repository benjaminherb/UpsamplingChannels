import os
from datetime import datetime

import numpy as np
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy

from loader import load_mat, load_observer, load_data
from bands import get_bands, get_test_band_infos
from processing import resample


def main():
    spectral_pixels, wavelengths = load_data("./res/train/")

    band_infos = get_test_band_infos("3")
    band_count = len(band_infos)

    bands = get_bands(band_infos, wavelengths)

    resampled_pixels = resample(spectral_pixels, bands)
    resampled_wavelengths = [band_info['wavelength'] for band_info in band_infos]

    input_layer = keras.layers.Input(shape=(band_count,))
    hidden_layer1 = keras.layers.Dense(32, activation='relu')(input_layer)
    hidden_layer2 = keras.layers.Dense(64, activation='relu')(hidden_layer1)
    output_layer = keras.layers.Dense(31, activation='linear')(hidden_layer2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=loss_function,
                  metrics=['RootMeanSquaredError'])

    model.fit(resampled_pixels, spectral_pixels, validation_split=0.2, epochs=5, batch_size=1024)

    # load test images
    # test_spectral_pixels, wavelengths = load_data("./res/test")
    # test_resampled_pixels = resample(test_spectral_pixels, bands)
    # model.evaluate(test_resampled_pixels, test_spectral_pixels)

    # Evaluation
    if not os.path.isdir("./out"):
        os.mkdir("./out")
    output_directory = os.path.join(
        "./out", f"{datetime.now().strftime('%y%m%d_%H%M%S')}_{band_count}_bands")
    os.mkdir(os.path.join(output_directory))

    test_directory = "./res/test"
    for file_name in os.listdir(test_directory):
        name = os.path.splitext(file_name)[0]
        # ground truth
        ground_truth_spectral, wavelengths = load_mat(os.path.join(test_directory, file_name))
        ground_truth_xyz = resample(ground_truth_spectral, load_observer(wavelengths))

        # prediction
        resampled = resample(ground_truth_spectral, bands)
        prediction_spectral = predict_image(resampled, model)
        prediction_xyz = resample(prediction_spectral, load_observer(wavelengths))

        max_value = max(ground_truth_xyz.max(), prediction_xyz.max())
        plt.imsave(os.path.join(
            output_directory, f"{name}_xyz_ground_truth.png"), ground_truth_xyz / max_value)
        plt.imsave(os.path.join(
            output_directory, f"{name}_xyz_prediction.png"), prediction_xyz / max_value)

        # plot mean
        save_plot(wavelengths, np.mean(ground_truth_spectral, axis=(0, 1)),
                  np.mean(prediction_spectral, axis=(0, 1)),
                  os.path.join(output_directory, f"{name}_spectral_mean.png"))

        # plot example pixel
        pos = 42
        save_plot(wavelengths, ground_truth_spectral[pos, pos, :],
                  prediction_spectral[pos, pos, :],
                  os.path.join(output_directory, f"{name}_spectral_{pos}_{pos}"))

    for band in bands:
        plt.plot(wavelengths, band)
    plt.xlabel('Wavelength')
    plt.ylabel('Response')
    plt.savefig(os.path.join(output_directory, "camera_response.png"))


def loss_function(ground_truth, prediction):
    squared_difference = tf.square(ground_truth - prediction)
    return tf.reduce_mean(squared_difference, axis=-1)


def predict_image(image, model):
    prediction = model.predict(image.reshape(-1, image.shape[-1]))
    return prediction.reshape((image.shape[0], image.shape[1], prediction.shape[1]))


def save_plot(wavelengths, ground_truth, prediction, path):
    plt.figure()
    plt.plot(wavelengths, ground_truth, label='Ground Truth')
    plt.plot(wavelengths, prediction, label='Prediction')
    plt.legend()
    plt.savefig(path)


if __name__ == "__main__":
    main()
