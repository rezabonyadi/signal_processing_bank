import numpy as np
from timeit import default_timer as timer


class SignalMeasuresCalculators:

    @staticmethod
    def accumulated_energy(x, settings):
        """
        :param x: the input signal. Its size is (number of channels, samples).
        :param settings: it provides a dictionary that includes an attribute "window_size" in which the energy is
            calculated.
        :return: is a dictionary that includes:
            "final_values" that is an array with the size "number of channels", each indicates the accumulated energy
            in that channel.
             "function_name": that is the name of this function ("accumulated_energy").
             "time": the amount of time in seconds that the calculations required.
        """
        window_size = settings["window_size"]

        total_time = timer()
        x_size = x.shape
        k = 0
        vars = np.zeros((x_size[0], int(np.floor(x_size[1] / window_size))), dtype=np.float32)
        for i in range(0, x_size[1] - window_size, window_size):
            vars[:, k] = np.var(x[:, i:i + window_size], axis=1)
            k = k + 1

        total_time = timer() - total_time
        results = dict()
        results["final_values"] = np.sum(vars, axis=1)
        results["function_name"] = "accumulated_energy"
        results["time"] = total_time
        return results


