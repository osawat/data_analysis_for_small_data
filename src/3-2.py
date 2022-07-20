import numpy as np
from linear_regression import least_squares, ls_est

# データを定義します
X1 = np.array([[0.01, 0.50, -0.12],
			[0.97, -0.63, 0.02],
			[0.41, 1.15, -1.17],
			[-1.38, -1.02, 1.27]])

X2 = np.array([[-0.01, 0.52, -0.12],
			[0.96, -0.64, 0.03],
			[0.43, 1.14, -1.17],
			[-1.38, -1.01, 1.27]])


y = np.array([[0.25], [0.08], [1.03], [-1.37]])
x = np.array([1, 0.7, -0.2])

# 回帰係数を求めます．
beta = least_squares(X1, y)
print(beta)

# 未知サンプルから出力を予測します
y_hat = ls_est(x, beta)
print(y_hat)

# X1, X2 それぞれでの回帰係数を計算します
beta1 = least_squares(X1, y)
print(beta1)

beta2 = least_squares(X2, y)
print(beta2)