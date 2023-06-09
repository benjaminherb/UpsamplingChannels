import os
from datetime import datetime

import numpy as np
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import colour

from src.util.loader import load_mat, load_observer
from src.util.processing import resample, XYZ_to_RGB, linear_to_sRGB


def train(spectral_pixels, bands):
    band_count = bands.shape[0]
    resampled_pixels = resample(spectral_pixels, bands)

    input_layer = keras.layers.Input(shape=(band_count,))
    hidden_layer1 = keras.layers.Dense(32, activation='relu')(input_layer)
    hidden_layer2 = keras.layers.Dense(64, activation='relu')(hidden_layer1)
    output_layer = keras.layers.Dense(31, activation='linear')(hidden_layer2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=loss_function,
                  metrics=['RootMeanSquaredError'])

    model.fit(resampled_pixels, spectral_pixels, validation_split=0.2, epochs=10, batch_size=1024)

    return model


def evaluate(model, bands, test_directory):
    # load test images
    # test_spectral_pixels, wavelengths = load_data("./res/test")
    # test_resampled_pixels = resample(test_spectral_pixels, bands)
    # model.evaluate(test_resampled_pixels, test_spectral_pixels)

    band_count = bands.shape[0]
    # Evaluation
    if not os.path.isdir("../out"):
        os.mkdir("../out")
    output_directory = os.path.join(
        "./out", f"{datetime.now().strftime('%y%m%d_%H%M%S')}_{band_count}_bands")
    os.mkdir(os.path.join(output_directory))
    model.save(os.path.join(output_directory, "model"))

    for file_name in os.listdir(test_directory):
        name = os.path.splitext(file_name)[0]

        # ground truth
        ground_truth_spectral, wavelengths = load_mat(os.path.join(test_directory, file_name))
        ground_truth_xyz = resample(ground_truth_spectral, load_observer(wavelengths))
        ground_truth_rgb = XYZ_to_RGB(ground_truth_xyz)
        ground_truth_rgb = ground_truth_rgb - ground_truth_rgb.min()

        # prediction
        resampled = resample(ground_truth_spectral, bands)
        prediction_spectral = predict_image(resampled, model)
        prediction_xyz = resample(prediction_spectral, load_observer(wavelengths))
        prediction_rgb = XYZ_to_RGB(prediction_xyz)
        prediction_rgb = prediction_rgb - prediction_rgb.min()

        # scaling
        max_value = max(ground_truth_rgb.max(), prediction_rgb.max())
        ground_truth_rgb = linear_to_sRGB(ground_truth_rgb / max_value)
        prediction_rgb = linear_to_sRGB(prediction_rgb / max_value)

        delta_E = colour.delta_E(colour.XYZ_to_Lab(ground_truth_xyz),
                                 colour.XYZ_to_Lab(prediction_xyz),
                                 method='CIE 2000')

        plt.imsave(os.path.join(
            output_directory, f"{name}_rgb_ground_truth.png"), ground_truth_rgb)
        plt.imsave(os.path.join(
            output_directory, f"{name}_rgb_prediction_{band_count}.png"),
            prediction_rgb)

        # plot mean
        save_plot(wavelengths, np.mean(ground_truth_spectral, axis=(0, 1)),
                  np.mean(prediction_spectral, axis=(0, 1)),
                  os.path.join(output_directory, f"{name}_spectral_mean_{band_count}.png"),
                  delta_E)

        # plot example pixel
        pos = 42
        save_plot(wavelengths, ground_truth_spectral[pos, pos, :],
                  prediction_spectral[pos, pos, :],
                  os.path.join(output_directory, f"{name}_spectral_{pos}_{pos}_{band_count}"))

    plt.figure()
    for band in bands:
        plt.plot(wavelengths, band)
    plt.xlabel('Wavelength')
    plt.ylabel('Response')
    plt.savefig(os.path.join(output_directory, f"camera_response_{band_count}.png"))
    plt.close()


def loss_function(ground_truth, prediction):
    squared_difference = tf.square(ground_truth - prediction)
    return tf.reduce_mean(squared_difference, axis=-1)


def predict_image(image, model):
    prediction = model.predict(image.reshape(-1, image.shape[-1]))
    return prediction.reshape((image.shape[0], image.shape[1], prediction.shape[1]))


def save_plot(wavelengths, ground_truth, prediction, path, delta_E=None):
    plt.figure()
    plt.plot(wavelengths, ground_truth, label='Ground Truth')
    plt.plot(wavelengths, prediction, label='Prediction')
    if delta_E is not None:
        plt.title(
            f"ΔE(2000): {np.mean(delta_E):.6f} (mean) / {delta_E.min():.6f} (min) / {delta_E.max():.6f} (max)")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Value [0-1]")
    plt.legend()
    plt.savefig(path)
    plt.close()
