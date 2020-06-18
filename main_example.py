import numpy as np
from signalscharacterisation import features_bank, constants_settings


x = np.random.rand(16, 1000)

settings = constants_settings.settings

# res = SignalsFeatures.accumulated_energy(x, settings)
# res = SignalsFeatures.moments_channels(x, settings)
# s = SignalsFeatures()
fea_list = np.array([0, 1])

# res1 = s.get_features_by_indexes(fea_list, x, settings)
# print(s.get_features_names())

# SignalsFeatures.call_feature_by_name("dyadic_spectrum_measures", x, settings)

all_features = features_bank.get_features_list()
for feature in all_features:
    print('calculating ', feature)
    res = features_bank.call_feature_by_name(feature, x, settings[feature], normalise=1)
    print({res["function_name"]: res["final_values"]})
    # print({res["function_name"]: res["time"]})


x = 0

# print(res["final_values"])
# print(res1[1]["final_values"])
# print(res["time"])
