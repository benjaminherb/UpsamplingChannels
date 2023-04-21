import numpy as np

test_band_infos = {
    "6": [{'wavelength': 425, 'sigma': 30},
          {'wavelength': 475, 'sigma': 30},
          {'wavelength': 525, 'sigma': 30},
          {'wavelength': 575, 'sigma': 30},
          {'wavelength': 625, 'sigma': 30},
          {'wavelength': 675, 'sigma': 30}],

    "5": [{'wavelength': 420, 'sigma': 30},
          {'wavelength': 485, 'sigma': 30},
          {'wavelength': 550, 'sigma': 30},
          {'wavelength': 615, 'sigma': 30},
          {'wavelength': 680, 'sigma': 30}],

    "4": [{'wavelength': 430, 'sigma': 30},
          {'wavelength': 510, 'sigma': 30},
          {'wavelength': 590, 'sigma': 30},
          {'wavelength': 670, 'sigma': 30}],

    "3": [{'wavelength': 450, 'sigma': 30},
          {'wavelength': 550, 'sigma': 30},
          {'wavelength': 650, 'sigma': 30}]
}


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


def get_test_band_infos(test):
    return test_band_infos[test]
