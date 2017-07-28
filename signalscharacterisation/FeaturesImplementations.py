from timeit import default_timer as timer

from scipy.stats import skew, kurtosis, mstats
from scipy.signal import resample
import numpy as np
# from spectrum import *
from statsmodels.tsa.ar_model import AR
from numba import autojit
from signalscharacterisation import FeaturesCalcHelper


class FeaturesImplementations:
    """
    This class implements a set of methods, each provide some characteristics of their input signal, x, according to
    the provided settings. It is assumed that the signal x is m by n, where m is the number of channels and n is the
    number of time points (samples).
    """
    features_list = ["accumulated_energy", "moments_channels", "freq_bands_measures", "dyadic_spectrum_measures",
                     "spectral_edge_freq", "correlation_channels_time", "correlation_channels_freq", "h_jorth",
                     "hjorth_fractal_dimension", "petrosian_fractal_dimension", "katz_fractal_dimension",
                     "hurst_fractal_dimension", "detrended_fluctuation", "autocorrelation", "autoregression",
                     "maximum_cross_correlation", "frequency_harmonise"]

    @staticmethod
    def get_features_list():
        """
        The function essentially returns the features_list list. This is a list of available functions to be used.

        :return: a list of the name of available functions as features.
        """
        return FeaturesImplementations.features_list

    @staticmethod
    def accumulated_energy(x, settings):
        """
        Calculates the accumulated energy of the signal. See ??? for more information.

        :param x: the input signal. Its size is (number of channels, samples).
        :param settings: it provides a dictionary that includes an attribute "window_size" in which the energy is
            calculated.
        :return: is a dictionary that includes:
            "final_values":     is an array with the size "number of channels", each indicates the accumulated energy
            in that channel.
             "function_name":   that is the name of this function ("accumulated_energy").
             "time":            the amount of time in seconds that the calculations required.
        """
        window_size = settings["energy_window_size"]

        total_time = timer()
        x_size = x.shape

        k = 0
        variances = np.zeros((x_size[0], int(np.floor(x_size[1] / window_size))), dtype=np.float32)
        for i in range(0, x_size[1] - window_size, window_size):
            variances[:, k] = np.var(x[:, i:i + window_size], axis=1)
            k = k + 1

        total_time = timer() - total_time
        final_values = np.sum(variances, axis=1)

        # if settings["is_normalised"] == 1:
        #     final_values /= final_values.sum()

        results = FeaturesCalcHelper.fill_results(["energy"], [final_values],
                                                  "accumulated_energy", [total_time], settings["is_normalised"])
        return results

    @staticmethod
    @autojit
    def moments_channels(x, settings):
        """
        Calculates mean, variance, skewness, and kurtosis of the given signal.

        :param x: the input signal. Its size is (number of channels, samples).
        :param settings: it provides a dictionary, empty for this function.
        :return:
        """

        t = timer()
        mean_values = np.nanmean(x, axis=1)
        mean_time = timer() - t
        t = timer()
        variance_values = np.nanvar(x, axis=1)
        variance_time = timer() - t
        t = timer()
        skewness_values = skew(x, axis=1)
        skewness_time = timer() - t
        t = timer()
        kurtosis_values = kurtosis(x, axis=1)
        kurtosis_time = timer() - t

        results = FeaturesCalcHelper.fill_results(["mean", "variance", "skewness", "kurtosis"],
                                                  [mean_values, variance_values, skewness_values, kurtosis_values],
                                                  "moments_channels",
                                                  [mean_time, variance_time, skewness_time, kurtosis_time],
                                                  settings["is_normalised"])

        return results

    @staticmethod
    def freq_bands_measures(x, settings):
        """

        :param x: the input signal. Its size is (number of channels, samples).
        :param settings: it provides a dictionary that includes an attribute "sampling_freq" in which sampling
        frequency of the data.
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
                                                  "freq_bands_measures", time, settings["is_normalised"])

        return results

    @staticmethod
    def dyadic_spectrum_measures(x, settings):
        """
        This function calculates the dyadic spectrum measures (power, shannon entropy, and correlation of powers in
        each frequency band among each pair of channels).

        :param x: the input signal. Its size is (number of channels, samples).
        :param settings: it provides a dictionary that includes an attribute "sampling_freq" in which sampling
        frequency of the data and "corr_type" that is the correlation type to use.
        :return:
        """

        sampling_freq = settings["sampling_freq"]
        type_corr = settings["corr_type"]  # TODO: the corr type is not used at the moment, fix this
        time = [0, 0, 0]
        x = np.transpose(x)
        n_samples = x.shape[0]
        lvl_d = np.floor(n_samples / 2)
        n_lvls = int(np.floor(np.log2(lvl_d)))
        dyadic_freq_levels = np.zeros(n_lvls + 1)
        coef = sampling_freq / n_samples

        for i in range(n_lvls + 1):
            dyadic_freq_levels[i] = lvl_d * coef
            lvl_d = np.floor(lvl_d / 2)

        dyadic_freq_levels = np.flipud(dyadic_freq_levels)
        t = timer()
        power_spectrum = FeaturesCalcHelper.calc_spectrum(x, dyadic_freq_levels, sampling_freq)
        time[0] = timer() - t
        t = timer()
        shannon_entropy = -1 * np.sum(np.multiply(power_spectrum, np.log(power_spectrum)), axis=0)
        time[1] = timer() - t
        t = timer()
        power_spec_corr = FeaturesCalcHelper.calc_corr(np.transpose(power_spectrum))
        iu = np.triu_indices(power_spec_corr.shape[0], 1)
        power_spec_corr = power_spec_corr[iu]
        time[2] = timer() - t
        results = FeaturesCalcHelper.fill_results(["power spectrum", "shannon entropy", "dyadic powers corr"],
                                                  [power_spectrum, shannon_entropy, power_spec_corr],
                                                  "dyadic_spectrum_measures", time, settings["is_normalised"])

        return results

    @staticmethod
    def spectral_edge_freq(x, settings):
        """

        :param x: the input signal. Its size is (number of channels, samples).
        :param settings: it provides a dictionary that includes an attribute "sampling_freq" in which sampling
        frequency of the data, "spectral_edge_tfreq" (40) and the "spectral_edge_power_coef"  (0.5).
        :return:
        """
        sfreq = settings["sampling_freq"]
        tfreq = settings["spectral_edge_tfreq"]
        ppow = settings["spectral_edge_power_coef"]
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
                                                  "spectral_edge_freq", [t], settings["is_normalised"])
        return results

    @staticmethod
    def correlation_channels_time(x, settings):
        """
        This function calculates the correlation between channels for the provided data. It also provides
        the eigen values related to the correlation matrix.

        :param x: the input signal. Its size is (number of channels, samples).
        :param settings: it provides a dictionary, empty for this function.
        :return:
        """

        time = [0, 0]

        t = timer()
        channels_correlations = FeaturesCalcHelper.calc_corr(x)
        time[0] = timer() - t

        t = timer()
        eigs = FeaturesCalcHelper.calc_eigens(channels_correlations)
        channels_correlations_eigs = eigs["lambda"]
        time[1] = timer() - t

        iu = np.triu_indices(channels_correlations.shape[0], 1)
        channels_correlations = channels_correlations[iu]

        results = FeaturesCalcHelper.fill_results(["correlation_channels", "lambda"],
                                                  [channels_correlations, channels_correlations_eigs],
                                                  "correlation_channels_time", [time], settings["is_normalised"])
        return results

    @staticmethod
    def correlation_channels_freq(x, settings):
        """
        This function provides the correlation between each pair of channels in the frequency domain. It also provides
        the eigen values related to the correlation matrix.

        :param x: the input signal. Its size is (number of channels, samples).
        :param settings: it provides a dictionary, empty for this function.
        :return:
        """
        # Calculate correlation matrix and its eigenvalues (b/w channels)
        time = [0, 0]

        t = timer()
        d = np.transpose(FeaturesCalcHelper.calc_normalized_fft(np.transpose(x)))
        channels_correlations = FeaturesCalcHelper.calc_corr(d)
        time[0] = timer() - t
        t = timer()
        eigs = FeaturesCalcHelper.calc_eigens(channels_correlations)
        channels_correlations_eigs = eigs["lambda"]
        time[1] = timer() - t

        iu = np.triu_indices(channels_correlations.shape[0], 1)
        channels_correlations = channels_correlations[iu]

        results = FeaturesCalcHelper.fill_results(["correlation_channels_freq", "lambda"],
                                                  [channels_correlations, channels_correlations_eigs],
                                                  "correlation_channels_freq", [time], settings["is_normalised"])
        return results

    @staticmethod
    def h_jorth(x, settings):
        """
        This function calculates h-jorth parameters, activity, mobility, and complexity.
        See ???? for information
        :param x: the input signal. Its size is (number of channels, samples).
        :param settings: settings (dummy for this function)
        :return: h-jorth parameters in a dictionary, including time, values, and function name.
        """
        time = [0, 0, 0]

        t = timer()
        activity = np.var(x, axis=1)
        time[0] = t - timer()

        t = timer()
        x_diff = np.diff(x, axis=1)
        x_var = activity
        x_diff_var = np.var(x_diff, axis=1)
        mobility = np.sqrt(np.divide(x_diff_var, x_var))
         # = calc_mobility(x, x_diff)
        time[1] = t - timer()

        t = timer()
        x_diff2_var = np.var(np.diff(x_diff, axis=1))
        complexity = np.divide(x_var * x_diff2_var, x_diff_var * x_diff_var)
        time[2] = t - timer()

        results = FeaturesCalcHelper.fill_results(["activity", "mobility", "complexity"],
                                                  [activity, mobility, complexity], "h_jorth", [t],
                                                  settings["is_normalised"])
        return results

    @staticmethod
    def hjorth_fractal_dimension(x, settings):
        """
        Compute Hjorth Fractal Dimension of a time series X, kmax
        is an HFD parameter. Kmax is basically the scale size or time offset.
        So you are going to create Kmax versions of your time series.
        The K-th series is every K-th time of the original series.
        This code was taken from pyEEG, 0.02 r1: http://pyeeg.sourceforge.net/

        :param x:  the input signal. Its size is (number of channels, samples).
        :param settings: dummy for hjorth FD.
        :return: petrosian fractal dimension for each channel.
        """

        t = timer()
        dimensions_channels = np.apply_along_axis(FeaturesCalcHelper.calc_hjorth_fractal_dimension, 1,
                                                  x, settings["hjorth_fd_k_max"])
        t = timer() - t
        results = FeaturesCalcHelper.fill_results(["h-jorth-FD"],
                                                  [dimensions_channels], "hjorth_fractal_dimension", [t],
                                                  settings["is_normalised"])
        return results

    @staticmethod
    def petrosian_fractal_dimension(x, settings):
        """
        Petrosian fractal dimension, see https://www.seas.upenn.edu/~littlab/Site/Publications_files/Esteller_2001.pdf

        :param x:  the input signal. Its size is (number of channels, samples).
        :param settings: dummy for petrosian.
        :return: petrosian fractal dimension for each channel.
        """
        t = timer()
        dimensions_channels = np.apply_along_axis(FeaturesCalcHelper.calc_petrosian_fractal_dimension, 1, x)
        t = timer() - t
        results = FeaturesCalcHelper.fill_results(["petrosian-FD"],
                                                  [dimensions_channels], "petrosian_fractal_dimension", [t],
                                                  settings["is_normalised"])
        return results

    @staticmethod
    def katz_fractal_dimension(x, settings):
        """
        Kartz fractal dimension, see https://www.seas.upenn.edu/~littlab/Site/Publications_files/Esteller_2001.pdf

        :param x: the input signal. Its size is (number of channels, samples).
        :param settings: dummy for kartz
        :return: kartz exponent for each channel
        """

        def get_kartz(x): return np.log(np.abs(x - x[0]).max()) / np.log(len(x))

        t = timer()
        dimensions_channels = np.apply_along_axis(get_kartz, 1, x)
        t = timer() - t

        results = FeaturesCalcHelper.fill_results(["kartz-FD"],
                                                  [dimensions_channels], "katz_fractal_dimension", [t],
                                                  settings["is_normalised"])
        return results

    @staticmethod
    def hurst_fractal_dimension(x, settings):
        """
        Hurst fractal dimension, see https://en.wikipedia.org/wiki/Hurst_exponent

        :param x:  the input signal. Its size is (number of channels, samples).
        :param settings: dummy for hurst.
        :return: petrosian fractal dimension for each channel.
        """
        t = timer()
        dimensions_channels = np.apply_along_axis(FeaturesCalcHelper.calc_hurst, 1, x)
        t = timer() - t
        results = FeaturesCalcHelper.fill_results(["hurst-FD"],
                                                  [dimensions_channels], "hurst_fractal_dimension", [t],
                                                  settings["is_normalised"])
        return results

    @staticmethod
    def detrended_fluctuation(x, settings):
        """
        This calculates the detrended fluctuation analysis. See
        https://en.wikipedia.org/wiki/Detrended_fluctuation_analysis.

        :param x: the input signal. Its size is (number of channels, samples).
        :param settings: the setting dictionary that includes an attribute "dfa_overlap" that is boolean for
        overlapping/non-overlapping calculations, and "dfa_order" that is the order for fitting.
        :return:
        """

        n_samples = x.shape[1]
        nvals = FeaturesCalcHelper.calc_logarithmic_n(4, 0.1 * n_samples, 1.2)
        overlap = settings["dfa_overlap"]
        order = settings["dfa_order"]

        t = timer()
        dfa_channels = 0
        # TODO: the following codes is extremely slow, we need to improve its performance.
        dfa_res = np.apply_along_axis(FeaturesCalcHelper.calc_dfa, 1, x, n_vals=nvals, overlap=overlap, order=order)
        t = timer() - t
        results = FeaturesCalcHelper.fill_results(["detrended_fluctuation"],
                                                  [dfa_res], "detrended_fluctuation", [t], settings["is_normalised"])
        return results

    @staticmethod
    def autocorrelation(x, settings):
        """
        This function calculates the autocorrelation of signals for each channel.

        :param x: the input signal. Its size is (number of channels, samples).
        :param settings: a dictionary with one attribute, "autocorr_n_lags", that is the max lag for autocorrelation.
        :return:
        """
        n_channels, n_samples = x.shape
        n_lag = settings["autocorr_n_lags"]
        autocorrs = np.zeros((n_channels, settings["autocorr_n_lags"]))

        t = timer()
        for i in range(0, n_channels):
            temp = FeaturesCalcHelper.crosscorr(x[i, :], x[i, :], lag=n_lag, both_sides=0)
            autocorrs[i, :] = temp[1:]

        # The following is much slower
        # autocorrs = np.apply_along_axis(acf, 1, x, unbiased=False, nlags=settings["autocorr_n_lags"])

        t = timer() - t
        results = FeaturesCalcHelper.fill_results(["autocorrelation"],
                                                  [autocorrs], "autocorrelation", [t], settings["is_normalised"])
        return results

    @staticmethod
    def autoregression(x, settings):
        """
        This function calculates the autoregression for each channel.

        :param x: the input signal. Its size is (number of channels, samples).
        :param settings: a dictionary with one attribute, "autoreg_lag", that is the max lag for autoregression.
        :return: the "final_value" is a matrix (number of channels, autoreg_lag) indicating the parameters of
         autoregression for each channel.
        """

        autoreg_lag = settings["autoreg_lag"]
        n_channels = x.shape[0]
        t = timer()
        channels_regg = np.zeros((n_channels, autoreg_lag + 1))
        for i in range(0, n_channels):
            fitted_model = AR(x[i, :]).fit(autoreg_lag)  # This is not the same as Matlab's for some reasons!
            # kk = ARMAResults(fitted_model)
            # autore_vals, dummy1, dummy2 = arburg(x[i, :], autoreg_lag) # This looks like Matlab's but slow
            channels_regg[i, 0: len(fitted_model.params)] = np.real(fitted_model.params)

        t = timer() - t
        results = FeaturesCalcHelper.fill_results(["autoregression"],
                                                  [channels_regg], "autoregression", [t], settings["is_normalised"])
        return results

    @staticmethod
    def maximum_cross_correlation(x, settings):
        """
        This calculates the maximum correlation measure.

        :param x: the input signal. Its size is (number of channels, samples).
        :param settings: a dictionary including the downsampling rate and the correlation lag.
        :return: the maximum correlation measure.
        """
        tau = settings["max_xcorr_downsample_rate"]
        lag = settings["max_xcorr_lag"]

        n_channels, n_samples = x.shape
        if tau < 1:
            x = np.apply_along_axis(resample, 1, x, int(tau * n_samples))

        cross_cor_matrix = np.zeros((n_channels, n_channels))
        t = timer()
        for i in range(0, n_channels-1):
            for j in range(i+1, n_channels-1):
                x_corr = FeaturesCalcHelper.crosscorr(x[i, :], x[j, :], lag)
                cc_abs = np.abs(x_corr)
                cross_cor_matrix[i, j] = max(cc_abs)

        t = timer() - t

        iu = np.triu_indices(cross_cor_matrix.shape[0], 1)
        cross_cor_matrix = cross_cor_matrix[iu]
        results = FeaturesCalcHelper.fill_results(["maximum_cross_correlation"],
                                                  [cross_cor_matrix], "maximum_cross_correlation", [t],
                                                  settings["is_normalised"])

        return results

    @staticmethod
    def frequency_harmonise(x, settings):
        """
        This was used by Michael Hills for the seizure detection competition in 2014 in Kaggle.
        See https://github.com/MichaelHills/seizure-detection/raw/master/seizure-detection.pdf

        :param x: the input signal. Its size is (number of channels, samples).
        :param settings: a dictionary including the "freq_hramonies_max_freq".
        :return:
        """

        time = [0, 0, 0]
        t = timer()
        m_x = x - np.mean(x, axis=1, keepdims=True)
        x_mgn = np.log10(np.absolute(np.fft.rfft(m_x, axis=1)[:, 1:settings["freq_hramonies_max_freq"]]))
        time[0] = timer() - t
        x_zscored = mstats.zscore(x_mgn, axis=1)
        channels_correlations = FeaturesCalcHelper.calc_corr(x_zscored)
        eigs = FeaturesCalcHelper.calc_eigens(channels_correlations)
        time[1] = timer() - t
        time[2] = time[1]

        channels_corrs_eig_values = eigs["lambda"]
        channels_corrs_eigs_vectors = eigs["vectors"]
        results = FeaturesCalcHelper.fill_results(["frequency_harmonise", "lambdas", "eigen_vectors"],
                                                  [x_mgn, channels_corrs_eig_values, channels_corrs_eigs_vectors],
                                                  "frequency_harmonise", time,
                                                  settings["is_normalised"])
        return results

