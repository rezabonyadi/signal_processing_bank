import numpy as np
from signalscharacterisation import SignalsFeatures

x = np.random.rand(16, 10000)
settings = {"energy_window_size": 10, "sampling_freq": 100, "spectral_edge_tfreq": 40,
            "spectral_edge_power_coef": 0.5, "corr_type": "pearson", "autocorr_n_lags": 10
            , "hjorth_fd_k_max": 3, "dfa_overlap": False, "dfa_order": 1, "autoreg_lag": 10,
            "max_xcorr_downsample_rate": 1, "max_xcorr_lag": 20, "freq_hramonies_max_freq": 48}
# res = SignalsFeatures.accumulated_energy(x, settings)
# res = SignalsFeatures.moments_channels(x, settings)
# s = SignalsFeatures()
fea_list = np.array([0, 1])

# res1 = s.get_features_by_indexes(fea_list, x, settings)
# print(s.get_features_names())

# SignalsFeatures.call_feature_by_name("dyadic_spectrum_measures", x, settings)

all_features = SignalsFeatures.get_features_list()
for i in range(0, len(all_features)):
    res = SignalsFeatures.call_feature_by_name(all_features[i], x, settings, normalise=1)
    print({res["function_name"]: res["final_values"]})
    # print({res["function_name"]: res["time"]})


x = 0

# print(res["final_values"])
# print(res1[1]["final_values"])
# print(res["time"])
