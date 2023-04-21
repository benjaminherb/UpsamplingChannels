import numpy as np


def resample(data, bands):
    # Assumes the bands and pixel bands match

    if len(data.shape) == 1:  # for single pixels
        return np.sum(data * bands, axis=1)

    elif len(data.shape) == 3:  # for 2d images
        # converts n*m*d array to (n*m)*d array, resamples and then converts
        # back to the original image size n*m*len(bands) with new band count
        return np.sum(
            (data.reshape(-1, data.shape[-1])[:, np.newaxis, :] * bands), axis=2).reshape(
            (data.shape[0], data.shape[1], len(bands)))

    else:  # 1d array of pixels
        return np.sum((data[:, np.newaxis, :] * bands), axis=2)
