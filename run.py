import keras.models

from src.upsampling import train, evaluate, loss_function
from src.util.bands import get_bands, get_test_band_infos
from src.util.loader import load_data, load_observer


def main():
    # prepare data
    spectral_pixels, wavelengths = load_data("./res/train/")
    for bandcount in ["3", "4", "5", "6"]:
        band_infos = get_test_band_infos(bandcount)
        bands = get_bands(band_infos, wavelengths)
        # bands = load_observer(wavelengths)

        # train model
        model = train(spectral_pixels, bands)

        # load trained model
        # model = keras.models.load_model("./out/230501_092447_6_bands/model", custom_objects={'loss_function': loss_function})

        # evaluate model
        evaluate(model, bands, "./res/test")


if __name__ == "__main__":
    main()
