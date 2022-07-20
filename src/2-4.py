import numpy as np
from pca import pca

data = [[2, 2], [1, -1], [-1, 1], [-2, -2]]
x = np.array( data)
P, T = pca(x)

print(P)
print(T)