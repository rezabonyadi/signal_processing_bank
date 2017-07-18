import numpy as np
from signalscharacterisation import SignalsFeatures

x = np.random.rand(16, 10000)
settings = {"window_size": 10, "sampling_freq": 100, "tfreq": 40, "power_coef": 0.5, "corr_type": "pearson"}
# res = SignalsFeatures.accumulated_energy(x, settings)
# res = SignalsFeatures.moments_channels(x, settings)
# s = SignalsFeatures()
fea_list = np.array([0, 1])

# res1 = s.get_features_by_indexes(fea_list, x, settings)
# print(s.get_features_names())

# SignalsFeatures.call_feature_by_name("dyadic_spectrum_measures", x, settings)

all_features = SignalsFeatures.get_features_list()
for i in range(0, len(all_features)):
    res = SignalsFeatures.call_feature_by_name(all_features[i], x, settings)
    print({res["function_name"]: res["final_values"]})

x = 0

# print(res["final_values"])
# print(res1[1]["final_values"])
# print(res["time"])
