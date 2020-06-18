from scipy import signal
import numpy as np
# from numba import cuda
import numpy as np
# from numba import cuda
from scipy import signal
from timeit import default_timer as timer

from signalscharacterisation import GpuHelperClass


def calc_normalized_fft(x, axis=0):
    """
    Calculates the magnitude of FFT of the input signal. Removes the DC component and normalizes the area to 1.

    :param x:
    :param axis:
    :return:
    """

    D = np.absolute(np.fft.fft(x, n=None, axis=axis))
    D[0, :] = 0  # set the DC component to zero
    normalized_d = D / np.sum(D, axis=axis)
    return normalized_d


def eeg_standard_freq_bands():
    """
    EEG waveforms are divided into frequency groups. These groups seem to be related to mental activity.
    alpha waves = 8-13 Hz = Awake with eyes closed
    beta waves = 14-30 Hz = Awake and thinking, interacting, doing calculations, etc.
    gamma waves = 30-45 Hz = Might be related to conciousness and/or perception (particular 40 Hz)
    theta waves = 4-7 Hz = Light sleep
    delta waves < 3.5 Hz = Deep sleep

    There are other EEG features like sleep spindles and K-complexes, but I think for this analysis
    we are just looking to characterize the waveform based on these basic intervals.
    :return:
    """
    return np.array([0.1, 4, 8, 14, 30, 45, 70, 180])  # Frequency levels in Hz


def calc_spectrum(x, freq_levels, sampling_freq):
    n_samples = x.shape[0]
    n_channels = x.shape[1]

    D = calc_normalized_fft(x)
    freq_levels = freq_levels[freq_levels <= sampling_freq]
    spectrum_levels = np.round(n_samples / sampling_freq * freq_levels).astype('int')
    dspect = np.zeros((len(spectrum_levels) - 1, n_channels))
    for j in range(0, len(spectrum_levels) - 1):
        dspect[j, :] = 2 * np.sum(D[spectrum_levels[j]:spectrum_levels[j + 1], :], axis=0)

    return dspect


def fill_results(measures_names, final_values, function_name, time, normalise=0):
    results = dict()
    if normalise == 1:
        for measure in final_values:
            measure /= measure.sum()

    results["measures_names"] = measures_names
    results["final_values"] = final_values
    results["function_name"] = function_name
    results["time"] = time
    results["is_normalised"] = normalise
    return results


def calc_corr(data):
    """
    This function returns the correlation between the rows of the data.

    :param data:
    :return:
    """

    C = np.array(np.corrcoef(data))
    C[np.isnan(C)] = 0  # Replace any NaN with 0
    C[np.isinf(C)] = 0  # Replace any Infinite values with 0

    return C


def calc_eigens(x):
    w, v = np.linalg.eig(x)
    # print(w)
    w = np.sort(w)
    w = np.real(w)
    return {"lambda": w, "vectors": v}


def calc_hjorth_fractal_dimension(x, k_max=3):
    """
    Compute Hjorth Fractal Dimension of a time series X, kmax
    is an HFD parameter. Kmax is basically the scale size or time offset.
    So you are going to create Kmax versions of your time series.
    The K-th series is every K-th time of the original series.
    This code was taken from pyEEG, 0.02 r1: http://pyeeg.sourceforge.net/
    """
    L = []
    y = []
    N = len(x)
    for k in range(1, k_max):
        l_k = []

        for m in range(k):
            l_mk = 0
            floor_value = int(np.floor((N - m) / k))
            for i in range(1, floor_value):
                l_mk += np.abs(x[m + i * k] - x[m + i * k - k])

            l_mk = l_mk * (N - 1) / floor_value / k
            l_k.append(l_mk)

        L.append(
            np.log(np.nanmean(l_k)))  # Using the mean value in this window to compare similarity to other windows
        y.append([np.log(float(1) / k), 1])

    (p, r1, r2, s) = np.linalg.lstsq(y, L)  # Numpy least squares solution

    return p[0]


def calc_petrosian_fractal_dimension(X, D=None):
    """Compute Petrosian Fractal Dimension of a time series from either two
    cases below:
        1. X, the time series of type list (default)
        2. D, the first order differential sequence of X (if D is provided,
           recommended to speed up)

    In case 1, D is computed by first_order_diff(X) function of pyeeg

    To speed up, it is recommended to compute D before calling this function
    because D may also be used by other functions whereas computing it here
    again will slow down.

    This code was taken from pyEEG, 0.02 r1: http://pyeeg.sourceforge.net/
    """

    # If D has been previously calculated, then it can be passed in here
    #  otherwise, calculate it.
    if D is None:  ## Xin Liu
        D = np.diff(X)  # Difference between one data point and the next

    # The old code is a little easier to follow
    N_delta = 0  # number of sign changes in derivative of the signal
    for i in range(1, len(D)):
        if D[i] * D[i - 1] < 0:
            N_delta += 1

    n = len(X)

    # This code is a little more compact. It gives the same
    # result, but I found that it was actually SLOWER than the for loop
    # N_delta = sum(np.diff(D > 0))

    return np.log10(n) / (np.log10(n) + np.log10(n / n + 0.4 * N_delta))


def calc_hurst(x):
    """Returns the Hurst Exponent of the time series vector ts"""
    # Create the range of lag values
    lags = range(2, 100)

    # Calculate the array of the variances of the lagged differences
    tau = [np.sqrt(np.nanstd(np.subtract(x[lag:], x[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = np.polyfit(np.log(lags), np.log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0] * 2.0


def calc_logarithmic_n(min_n, max_n, factor):
    """
    Creates a list of values by successively multiplying a minimum value min_n by
    a factor > 1 until a maximum value max_n is reached.

    Non-integer results are rounded down.

    Args:
    min_n (float): minimum value (must be < max_n)
    max_n (float): maximum value (must be > min_n)
    factor (float): factor used to increase min_n (must be > 1)

    Returns:
    list of integers: min_n, min_n * factor, min_n * factor^2, ... min_n * factor^i < max_n
                      without duplicates
    """
    assert max_n > min_n
    assert factor > 1

    # stop condition: min * f^x = max
    # => f^x = max/min
    # => x = log(max/min) / log(f)

    max_i = int(np.floor(np.log(1.0 * max_n / min_n) / np.log(factor)))
    ns = [min_n]

    for i in range(max_i + 1):
        n = int(np.floor(min_n * (factor ** i)))
        if n > ns[-1]:
            ns.append(n)

    return ns


def calc_dfa(data, n_vals=None, overlap=True, order=1, gpu=False, debug_plot=False, plot_file=None):
    total_N = len(data)
    if n_vals is None:
        n_vals = calc_logarithmic_n(4, 0.1 * total_N, 1.2)

    # create the signal profile (cumulative sum of deviations from the mean => "walk")
    walk = np.nancumsum(data - np.nanmean(data))
    fluctuations = np.zeros(len(n_vals))
    i = 0

    for n in n_vals:
        # subdivide data into chunks of size n
        if overlap:
            # step size n/2 instead of n
            d = np.array([walk[i:i + n] for i in range(0, len(walk) - n, n // 2)])
        else:
            # non-overlapping windows => we can simply do a reshape
            d = walk[:total_N - (total_N % n)]
            d = d.reshape((total_N // n, n))

        # calculate local trends as polynomes
        x = np.arange(n)
        flucs = cpu_calc_fluc(x, d, order)
        
        # TODO: No GPU support for now
        # if gpu is False:
            # flucs = cpu_calc_fluc(x, d, order)
        # else:
            # flucs = GpuHelperClass.gpu_calc_dfa(x, d, order)

        # calculate mean fluctuation over all subsequences
        f_n = np.nansum(flucs) / len(flucs)
        fluctuations[i] = f_n
        i += 1

    fluctuations = np.array(fluctuations)
    # filter zeros from fluctuations
    nonzero = np.where(fluctuations != 0)
    n_vals = np.array(n_vals)[nonzero]
    fluctuations = fluctuations[nonzero]
    if len(fluctuations) == 0:
        # all fluctuations are zero => we cannot fit a line
        poly = [np.nan, np.nan]
    else:
        poly = np.polyfit(np.log(n_vals), np.log(fluctuations), 1)
        # if debug_plot:
        # plo.plot_reg(np.log(nvals), np.log(fluctuations), poly, "log(n)", "std(X,n)", fname=plot_file)

    return poly[0]


def cpu_calc_fluc(x, d, order):
    n = len(x)
    tpoly = np.array([np.polyfit(x, d[i], order) for i in range(len(d))])
    trend = np.array([np.polyval(tpoly[i], x) for i in range(len(d))])
    # calculate standard deviation ("fluctuation") of walks in d around trend
    flucs = np.sqrt(np.nansum((d - trend) ** 2, axis=1) / n)
    return flucs


def calc_corrlation(x, y):
    co = signal.correlate(x, y, mode='valid')
    return co


def crosscorr(x, y, lag=1, both_sides=1):
    denom = np.std(x) * np.std(y) * len(x)
    m_x = np.mean(x)
    m_y = np.mean(y)
    right_xcoor = [calc_corrlation(x[i:] - m_x, y[0:len(x) - i] - m_y) / denom for i in range(0, lag + 1)]
    if both_sides == 1:
        left_xcoor = [calc_corrlation(x[0:len(x) - i] - m_x, y[i:] - m_y) / denom for i in range(1, lag + 1)]
    else:
        left_xcoor = []
    # right_xcoor = [xcorr(x[i:], y) for i in range(1, lag)]
    return left_xcoor + right_xcoor
