from src.upsampling import train, evaluate
from src.util.bands import get_bands, get_test_band_infos
from src.util.loader import load_data


def main():
    # prepare data
    spectral_pixels, wavelengths = load_data("./res/train/")
    band_infos = get_test_band_infos("6")
    bands = get_bands(band_infos, wavelengths)

    # train model
    model = train(spectral_pixels, bands)

    # evaluate model
    evaluate(model, bands, "./res/test")


if __name__ == "__main__":
    main()
