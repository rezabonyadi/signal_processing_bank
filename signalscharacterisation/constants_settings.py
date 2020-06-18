from signalscharacterisation import features_implementations as fi


features_dict = {"accumulated_energy": fi.accumulated_energy,
                 "moments_channels": fi.moments_channels,
                 "freq_bands_measures": fi.freq_bands_measures,
                 "dyadic_spectrum_measures": fi.dyadic_spectrum_measures,
                 "spectral_edge_freq": fi.spectral_edge_freq,
                 "autoregression": fi.autoregression,
                 "correlation_channels_time": fi.correlation_channels_time,
                 "h_jorth": fi.h_jorth,
                 "correlation_channels_freq": fi.correlation_channels_freq,
                 "hjorth_fractal_dimension": fi.hjorth_fractal_dimension,
                 "petrosian_fractal_dimension": fi.petrosian_fractal_dimension,
                 "katz_fractal_dimension": fi.katz_fractal_dimension,
                 "autocorrelation": fi.autocorrelation,
                 "detrended_fluctuation": fi.detrended_fluctuation,
                 "hurst_fractal_dimension": fi.hurst_fractal_dimension,
                 "maximum_cross_correlation": fi.maximum_cross_correlation,
                 "frequency_harmonies": fi.frequency_harmonies}

settings = {'accumulated_energy': {"energy_window_size": 10},
            'moments_channels': {},
            'freq_bands_measures': {"sampling_freq": 100},
            'dyadic_spectrum_measures': {"sampling_freq": 100, 'corr_type': 'Pearson'},
            'spectral_edge_freq': {"spectral_edge_tfreq": 40, "spectral_edge_power_coef": 0.5, "sampling_freq": 100},
            'correlation_channels_time': {},
            'correlation_channels_freq': {},
            'detrended_fluctuation': {"dfa_overlap": False, "dfa_order": 1, "dfa_gpu": False},
            'h_jorth': {},
            'autocorrelation': {"autocorr_n_lags": 10},
            'autoregression': {"autoreg_lag": 10},
            'maximum_cross_correlation': { "max_xcorr_downsample_rate": 1, "max_xcorr_lag": 20},
            'hjorth_fractal_dimension': {"hjorth_fd_k_max": 3},
            'petrosian_fractal_dimension': {},
            'katz_fractal_dimension': {},
            'hurst_fractal_dimension': {},
            'frequency_harmonies': {"freq_harmonies_max_freq": 48}
            }