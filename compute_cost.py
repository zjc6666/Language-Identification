import numpy as np


c_miss = np.array([1, 1])
c_fa = np.array([1, 1])
p_target=np.array([0.1, 0.5])
beta = (c_fa / c_miss) * ((1 - p_target) / p_target)
theta = np.log(beta)
print(beta)