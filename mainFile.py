import numpy as np
from signalscharacterisation import FeaturesBank


x = np.random.rand(16, 10000)
settings = {"window_size": 10}
res = FeaturesBank.accumulated_energy(x, settings)

print(res["final_values"])
print(res["time"])
