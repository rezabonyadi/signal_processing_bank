import numpy as np
from signalscharacterisation.FeaturesBank import SignalsFeatures


x = np.random.rand(16, 10000)
settings = {"window_size": 10}
res = SignalsFeatures.accumulated_energy(x, settings)
# res = SignalsFeatures.moments_channels(x, settings)
s = SignalsFeatures()
fea_list = np.array([0, 1])
res1 = s.get_features_by_indexes(fea_list, x, settings)
print(s.get_features_names())

print(res["final_values"])
print(res1[1]["final_values"])
# print(res["time"])
