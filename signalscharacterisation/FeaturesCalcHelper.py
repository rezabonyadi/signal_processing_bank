import numpy as np


def calc_normalized_fft(x):
    '''
    Calculates the FFT of the epoch signal. Removes the DC component and normalizes the area to 1
    '''
    D = np.absolute(np.fft.fft(x, n=None, axis=0))
    D[0, :] = 0  # set the DC component to zero
    normalized_d = D/np.sum(D, axis=0)
    return normalized_d


def eeg_standard_freq_bands():
    '''
    EEG waveforms are divided into frequency groups. These groups seem to be related to mental activity.
    alpha waves = 8-13 Hz = Awake with eyes closed
    beta waves = 14-30 Hz = Awake and thinking, interacting, doing calculations, etc.
    gamma waves = 30-45 Hz = Might be related to conciousness and/or perception (particular 40 Hz)
    theta waves = 4-7 Hz = Light sleep
    delta waves < 3.5 Hz = Deep sleep

    There are other EEG features like sleep spindles and K-complexes, but I think for this analysis
    we are just looking to characterize the waveform based on these basic intervals.
    '''
    return (np.array([0.1, 4, 8, 14, 30, 45, 70, 180]))  # Frequency levels in Hz



def calc_spectrum(x, freq_levels, sampling_freq):
    n_samples = x.shape[0]
    n_channels = x.shape[1]

    D = calc_normalized_fft(x)
    freq_levels = freq_levels[freq_levels <= sampling_freq]
    spectrum_levels = np.round(n_samples/sampling_freq*freq_levels).astype('int')
    dspect = np.zeros((len(spectrum_levels) - 1, n_channels))
    for j in range(0, len(spectrum_levels) - 1):
        dspect[j, :] = 2 * np.sum(D[spectrum_levels[j]:spectrum_levels[j + 1], :], axis=0)

    return dspect


def fill_results(measures_names, final_values, function_name, time):
    results = dict()
    results["measures_names"] = measures_names
    results["final_values"] = final_values
    results["function_name"] = function_name
    results["time"] = time
    return results


def corr(data, type_corr):
    C = np.array(data.corr(type_corr))
    C[np.isnan(C)] = 0  # Replace any NaN with 0
    C[np.isinf(C)] = 0  # Replace any Infinite values with 0
    w, v = np.linalg.eig(C)
    # print(w)
    x = np.sort(w)
    x = np.real(x)
    return x