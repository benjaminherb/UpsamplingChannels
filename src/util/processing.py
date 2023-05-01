import numpy as np
from src.util.loader import load_primaries, load_whitepoint


# SAMPLING

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


# TRISTIMULI CONVERSIONS

def XYZ_to_RGB(XYZ_image):
    primaries = load_primaries("sRGB")
    whitepoint = load_whitepoint("D65")
    XYZ_to_RGB_matrix = get_XYZ_to_RGB_matrix(primaries, whitepoint)
    RGB_image = np.dot(XYZ_image, XYZ_to_RGB_matrix.T)
    return RGB_image


def RGB_to_XYZ(RGB_image):
    primaries = load_primaries("sRGB")
    whitepoint = load_whitepoint("D65")
    RGB_to_XYZ_matrix = get_RGB_to_XYZ_matrix(primaries, whitepoint)
    XYZ_image = np.dot(RGB_image, RGB_to_XYZ_matrix.T)
    return XYZ_image


#  TRANSFER CURVES

def linear_to_sRGB(v):
    return ((v > 0.0031308) * (1.055 * np.power(v, (1 / 2.4)) - 0.055)
            + (v <= 0.0031308) * (v * 12.92))


def sRGB_to_linear(v):
    return ((v > 0.04045) * np.power((0.055 + v) / 1.055, 2.4)
            + (v <= 0.04045) * (v / 12.92))


# UTILITY FUNCTIONS

def get_RGB_to_XYZ_matrix(primaries, whitepoint):
    Xr, Yr, Zr = primaries['red']
    Xg, Yg, Zg = primaries['green']
    Xb, Yb, Zb = primaries['blue']
    Xw, Yw, Zw = whitepoint

    XYZ_matrix = np.array([[Xr, Xg, Xb], [Yr, Yg, Yb], [Zr, Zg, Zb]])
    XwYwZw_vector = np.array([Xw, Yw, Zw])
    SrSgSb_vector = np.linalg.solve(XYZ_matrix, XwYwZw_vector)
    RGB_to_XYZ_matrix = XYZ_matrix * np.array([SrSgSb_vector, SrSgSb_vector, SrSgSb_vector]).T
    return RGB_to_XYZ_matrix


def get_XYZ_to_RGB_matrix(primaries, whitepoint):
    return np.linalg.inv(get_RGB_to_XYZ_matrix(primaries, whitepoint))


def xyY_to_XYZ(x, y, Y):
    # http://www.brucelindbloom.com/index.html?Eqn_Spect_to_XYZ.html
    if y == 0:
        return 0, 0, 0
    X = x * Y / y
    Z = (1 - x - y) * Y / y
    return X, Y, Z
