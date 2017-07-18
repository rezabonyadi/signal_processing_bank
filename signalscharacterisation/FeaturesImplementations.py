import numpy as np
from timeit import default_timer as timer
from scipy import stats
from signalscharacterisation import FeaturesCalcHelper


class FeaturesImplementations:
    features_list = ["accumulated_energy", "moments_channels", "freq_bands_measures", "dyadic_spectrum_measures",
                     "spectral_edge_freq"]

    @staticmethod
    def get_features_list():
        return FeaturesImplementations.features_list

    @staticmethod
    def accumulated_energy(x, settings):
        """
        :param x: the input signal. Its size is (number of channels, samples).
        :param settings: it provides a dictionary that includes an attribute "window_size" in which the energy is
            calculated.
        :return: is a dictionary that includes:
            "final_values":     is an array with the size "number of channels", each indicates the accumulated energy
            in that channel.
             "function_name":   that is the name of this function ("accumulated_energy").
             "time":            the amount of time in seconds that the calculations required.
        """
        window_size = settings["window_size"]

        total_time = timer()
        x_size = x.shape

        k = 0
        variances = np.zeros((x_size[0], int(np.floor(x_size[1] / window_size))), dtype=np.float32)
        for i in range(0, x_size[1] - window_size, window_size):
            variances[:, k] = np.var(x[:, i:i + window_size], axis=1)
            k = k + 1

        total_time = timer() - total_time
        results = FeaturesCalcHelper.fill_results(["energy"],
                                                  [np.sum(variances, axis=1)], "accumulated_energy", [total_time])
        return results

    @staticmethod
    def moments_channels(x, settings):
        """
        :param x:
        :param settings:
        :return:
        """

        t = timer()
        mean_values = np.nanmean(x, axis=1)
        mean_time = timer() - t
        t = timer()
        variance_values = np.nanvar(x, axis=1)
        variance_time = timer() - t
        t = timer()
        skewness_values = stats.skew(x, axis=1)
        skewness_time = timer() - t
        t = timer()
        kurtosis_values = stats.kurtosis(x, axis=1)
        kurtosis_time = timer() - t

        results = FeaturesCalcHelper.fill_results(["mean", "variance", "skewness", "kurtosis"],
                                        [mean_values, variance_values, skewness_values, kurtosis_values],
                                        "moments_channels", [mean_time, variance_time, skewness_time, kurtosis_time])

        return results

    @staticmethod
    def freq_bands_measures(x, settings):
        """
        :param x:
        :param settings:
        :return:
        """
        sampling_freq = settings["sampling_freq"]
        time = [0, 0]
        x = np.transpose(x)

        t = timer()
        freq_levels = FeaturesCalcHelper.eeg_standard_freq_bands()
        power_spectrum = FeaturesCalcHelper.calc_spectrum(x, freq_levels, sampling_freq)
        time[0] = timer() - t
        t = timer()
        shannon_entropy = -1 * np.sum(np.multiply(power_spectrum, np.log(power_spectrum)), axis=0)
        time[1] = timer() - t
        results = FeaturesCalcHelper.fill_results(["power spectrum", "shannon entropy"],
                                                  [power_spectrum, shannon_entropy],
                                                  "freq_bands_measures", time)

        return results

    @staticmethod
    def dyadic_spectrum_measures(x, settings):
        """
        :param x:
        :param settings:
        :return:
        """

        sampling_freq = settings["sampling_freq"]
        time = [0, 0]
        x = np.transpose(x)
        n_samples = x.shape[0]
        lvl_d = np.floor(n_samples/2)
        n_lvls = int(np.floor(np.log2(lvl_d)))
        dyadic_freq_levels = np.zeros(n_lvls + 1)
        coef = sampling_freq/n_samples

        for i in range(n_lvls + 1):
            dyadic_freq_levels[i] = lvl_d * coef
            lvl_d = np.floor(lvl_d/2)

        dyadic_freq_levels = np.flipud(dyadic_freq_levels)
        t = timer()
        power_spectrum = FeaturesCalcHelper.calc_spectrum(x, dyadic_freq_levels, sampling_freq)
        time[0] = timer() - t
        t = timer()
        shannon_entropy = -1 * np.sum(np.multiply(power_spectrum, np.log(power_spectrum)), axis=0)
        time[1] = timer() - t
        results = FeaturesCalcHelper.fill_results(["power spectrum", "shannon entropy"],
                                                  [power_spectrum, shannon_entropy],
                                                  "dyadic_spectrum_measures", time)

        return results

    @staticmethod
    def spectral_edge_freq(x, settings):
        """
        :param x:
        :param settings:
        :return:
        """
        sfreq = settings["sampling_freq"]
        tfreq = settings["tfreq"]
        ppow = settings["power_coef"]
        x = np.transpose(x)
        n_samples = x.shape[0]

        t = timer()
        topfreq = int(round(n_samples / sfreq * tfreq)) + 1

        D = FeaturesCalcHelper.calc_normalized_fft(x)
        A = np.cumsum(D[:topfreq, :], axis=0)
        B = A - (A.max() * ppow)
        spedge = np.min(np.abs(B), axis=0)
        spedge = (spedge - 1) / (topfreq - 1) * tfreq

        t = timer() - t

        results = FeaturesCalcHelper.fill_results(["spectral edge freq"], [spedge],
                                                  "spectral_edge_freq", [t])
        return results


