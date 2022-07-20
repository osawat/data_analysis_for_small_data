import numpy as np
import scale

# 既存データと未知サンプルをndarray 型で定義します
X = np.array([[1, 2, 3],
			[4, 5, 6],
			[7, 8, 9]])

x = np.array([[10, 11, 12]])

# X を標準化します
Xscale, meanX, stdX = scale.autoscale(X)
print(Xscale)