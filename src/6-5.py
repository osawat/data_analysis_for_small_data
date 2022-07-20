import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from autoencorder import Autoencoder, AE_T2RE
from mspc import mspc_CL, mspc_ref, mspc_T2Q
from scale import autoscale, scaling

## ------------------------6.5 ------------------------
# 正常, 異常データの読み込み
train_data = pd.read_csv('./normal_data.csv')
# 対象とする異常によって読み込むファイルを変更してください．
faulty_data = pd.read_csv('./idv1 _data.csv')

## ------------------------6.6 ------------------------
# 正常データを用いたモデルの学習
model = LocalOutlierFactor(n_neighbors = 50, novelty = True, contamination = 0.01)
model.fit(train_data)

# 管理限界の取得
CL_lof = model.offset_

# 異常データのLOF スコアの計算
score_lof = model.score_samples(faulty_data)

## ------------------------ 6.7 ------------------------
# 正常データを用いたモデルの学習
model = IsolationForest(contamination = 0.05)
model.fit(train_data)

# 管理限界の取得
CL_if = model.offset_

# 異常データのiForest スコアの計算
score_if = model.score_samples(faulty_data)

## ------------------------6.8 ------------------------
# 正常データを用いたモデルの学習
meanX, stdX, U, S, V = mspc_ref(train_data, numPC = 17)

# 管理限界の決定
T2_train, Q_train = mspc_T2Q(train_data, meanX, stdX, U, S, V)
CL_T2_mspc, CL_Q_mspc = mspc_CL(T2, Q, alpha = 0.99)

# 異常データのT2 統計量とQ 統計量の計算
T2_mspc, Q_mspc = mspc_T2Q(faulty_data, meanX, stdX, U, S, V)



## ------------------------　6.10 ------------------------
# ハイパーパラメータの設定
z_dim = 17 # 中間層の次元
model = Autoencoder(z_dim = z_dim)
criterion = nn.MSELoss() # 誤差関数(平均二乗誤差)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001) # オプティマイザ
num_epochs = 110 # エポック数
batch_size = 20 # バッチサイズ

# 学習用データの前処理
train_data, mean_train, std_train = autoscale(train_data)
train_data = train_data.astype('float 32')
train_data = train_data.values
train_data = torch.tensor(train_data) # Tensor 型への変換します
dataloader = DataLoader(train_data, batch_size = batch_size, shuffle = True)

# インスタンスの作成
model = Autoencoder(z_dim = z_dim)

# 学習ループ
for epoch in range(num_epochs):
    for data in dataloader:
        xhat, z = model.forward(data)
        loss = criterion(xhat, data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


## ------------------------ 6.11 ------------------------
x_hat, z = model.forward(train_data)
z = z.detach().numpy()
z_bar = np.average(z, axis = 0)
z_bar = np.reshape(z_bar, (len(z_bar), 1))
S_z = np.cov(z.T)

## ------------------------ 6.13 ------------------------
# 管理限界の決定
train_data = pd.read_csv('./normal_data.csv')# 正常データの読み込み
T2_train, RE_train = AE_T2RE(train_data, z_hat, S_z)
CL_T2_AE, CL_RE_AE = mspc_CL(T2_train, RE_train, alpha = 0.99)

## ------------------------ 6.14 ------------------------
# 前処理
faulty_data = scaling(faulty_data, mean_train, std_train)
faulty_data = faulty_data.astype('float 32')
faulty_data = faulty_data.values
faulty_data = torch.tensor(faulty_data)

# T2 統計量, RE の計算
T2_AE, RE_AE = AE_T2RE(faulty_data, z_hat, S_z)

## ------------------------ 6.15 ------------------------
plt.plot(list(range(1, 961)), abs(score_lof)) # LOF とiForest では絶対値を計算する
plt.xlim(1, 960)
plt.ylim(0, 10)
plt.hlines(abs(CL_lof), 1, 960, "r", linestyles = 'dashed') # 管理限界の表示
plt.vlines(160, 0, 15, "g") # 異常発生時刻
plt.xlabel('Sample')
plt.ylabel('Score')
plt.title('IDV_1')

## ------------------------ 6.16 ------------------------
# 寄与の計算
cont_T2, cont_Q = cont_T2Q(faulty_data, meanX, stdX, U, S, V)

# 異常発生後100 サンプルの寄与の平均を計算
fault_cont_T2 = np.average(cont_T2[160:260, :], axis = 0)
fault_cont_Q = np.average(cont_Q[160:260, :], axis = 0)

# プロット
plt.figure()
plt.bar(range(1, 53), fault_cont_T2)
plt.title('contributions␣of␣T2')
plt.xlabel('varible')
plt.ylabel('contribution')
plt.figure()
plt.bar(range(1,53), fault_cont_Q)
plt.xlabel('variable')
plt.ylabel('contribution')
plt.title('contributions_of_Q')