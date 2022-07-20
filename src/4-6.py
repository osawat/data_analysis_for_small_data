import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import SpectralClustering

from linear_regression import kmean
from ncsc_vs import ncsc


# データを生成します
np.random.seed(10)
a = [2, -1, 0.5]
b = [0.1, 0.1, 0.1]
X = np.random.rand(300) - 0.5

Y0 = a[0] * X[:100] + b[0] * np.random.rand(100)
Y1 = a[1] * X[100:200] + b[1] * np.random.rand(100)
Y2 = a[2] * X[200:] + b[2] * np.random.rand(100)
Y = np.vstack([Y0, Y1, Y2])
X = np.hstack([X.reshape(-1, 1), Y.reshape(-1, 1)])

label = np.vstack([np.zeros([100, 1]), np.ones([100, 1]), 2*np.ones([100, 1])])

# k- 平均法
n_clusters = 3
labels_km = kmean(X, n_clusters)

# ガウシアンカーネルによるSC
labels_sc = SpectralClustering(n_clusters = n_clusters, affinity = 'rbf', random_state = 0).fit_predict(X)

# NCSC
n_clusters = 3 # 分割グループ数
labels_ncsc = ncsc(X, n_clusters, gamma = 0.99)

# 結果のプロット
sz = 3
fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax1.scatter(X[:, 0], X[:, 1], c = label, s = sz)
ax1.set_title("True_label")

ax2 = fig.add_subplot(2, 2, 2)
ax2.scatter(X[:, 0], X[:, 1], c = labels_km, s = sz)
ax2.set_title("k- means")

ax3 = fig.add_subplot(2, 2, 3)
ax3.scatter(X[:, 0], X[:, 1], c = labels_sc, s = sz)
ax3.set_title("SC")

ax4 = fig.add_subplot(2, 2, 4)
ax4.scatter(X[:, 0], X[:, 1], c = labels_ncsc, s = sz)
ax4.set_title("NCSC")

plt.tight_layout()