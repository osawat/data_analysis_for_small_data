import numpy as np
from linear_regression import least_squares, pcr, ridge, nipals_pls1, simpls

# データを定義します
X1 = np.array([[-1.12, -0.51, 0.69],
			[-0.43, -1.12, 1.02],
			[0.37, 1.10, -0.98],
			[1.19, 0.53, -0.73]])

X2 = np.array([[-1.12, -0.51, 0.70],
			[-0.43, -1.12, 1.01],
			[0.36, 1.10, -0.98],
			[1.20, 0.53, -0.73]])

y = np.array([0.4, 1.17, -1.14, -0.42])

## 3.6 ----------------------
# X1， X2 それぞれでの回帰係数を計算します
beta1 = least_squares(X1, y)
print(beta1)

beta2 = least_squares(X2, y)
print(beta2)

# 主成分数を2 に設定します
R = 2

beta1 = pcr(X1, y, R)
print(beta1)

beta2 = pcr(X2, y, R)
print(beta2)

## 3.7 ----------------------

# 主成分数を1 に設定します
R = 1

beta1 = pcr(X1, y, R)
print(beta1)

beta2 = pcr(X2, y, R)
print(beta2)


## 3.9 ----------------------
# パラメータを0.1 に設定します
mu = 0.1
beta1 = ridge(X1, y, mu)
print(beta1)

beta2 = ridge(X2, y, mu)
print(beta2)

# パラメータを1 に設定します
mu = 1
beta1 = ridge(X1, y, mu)
print(beta1)

beta2 = ridge(X2, y, mu)
print(beta2)

## 3.11 ----------------------
# 潜在変数が2 のとき
R = 2
beta1 = nipals_pls1(X1, y, R)
print(beta1)

beta2 = nipals_pls1(X2, y, R)
print(beta2)

# 潜在変数が1 のとき
R = 1
beta1 = nipals_pls1(X1, y, R)
print(beta1)

beta2 = nipals_pls1(X2, y, R)
print(beta2)

## 3.14 ----------------------
# 潜在変数が2 のとき
R = 2
beta1 = simpls(X1, y, R)
print(beta1)

beta2 = simpls(X2, y, R)
print(beta2)

# 潜在変数が1 のとき
R = 1
beta1 = simpls(X1, y, R)
print(beta1)

beta2 = simpls(X2, y, R)
print(beta2)