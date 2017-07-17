import numpy as np
from timeit import default_timer as timer
from scipy import stats


class SignalsFeatures:
    features_list = []

    def __init__(self):
        self.features_list = ["accumulated_energy", "moments_channels"]

    def get_features_by_indexes(self, features_indexes, x, settings):
        length = features_indexes.shape
        return_list = [dict() for i in range(length[0])]
        for i in features_indexes:
            return_list[i] = self.get_feature_by_name(self.features_list[i], x, settings)

        return return_list

    def get_features_names(self):
        return self.features_list

    def get_feature_by_name(self, feature_name, x, settings):
        feature = getattr(self, feature_name)
        return feature(x, settings)

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
        results = dict()
        results["measures_names"] = ["energy"]
        results["final_values"] = [np.sum(variances, axis=1)]
        results["function_name"] = "accumulated_energy"
        results["time"] = [total_time]
        return results

    @staticmethod
    def autocorrelation_channels(x, settings):
        """
        :param x:
        :param settings:
        :return:
        """
        x_size = x.shape
        num_lag = settings["num_lag"]

        # for i in range(0, x_size[0], 1): # for each channel

    @staticmethod
    def moments_channels(x, settings):
        """
        :param x:
        :param settings:
        :return:
        """
        x_size = x.shape

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

        results = dict()
        results["measures_names"] = ["mean", "variance", "skewness", "kurtosis"]
        results["final_values"] = [mean_values, variance_values, skewness_values, kurtosis_values]
        results["function_name"] = "accumulated_energy"
        results["time"] = [mean_time, variance_time, skewness_time, kurtosis_time]
        results["function_name"] = "moments_channels"

        return results
