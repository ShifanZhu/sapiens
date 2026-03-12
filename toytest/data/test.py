import numpy as np

a = np.zeros((2, 3))
b = np.zeros((2, 3))
print(np.stack([a, ], axis=0).shape)  # (2, 2, 3)

