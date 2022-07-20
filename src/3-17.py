import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from linear_regression import pcr, pls_beta, pls_cv, pls_vip, pred_eval, lasso, ridge, simpls
from ncsc_vs import ncsc, valselect
from scale import autoscale, scaling, rescaling


# MATLAB 形式のファイルを読み込みます
CNGATEST_dict = scipy.io.loadmat('CNGATEST.mat')

# データは辞書型として保存されているので，キーを確認します
print(dict.keys())

# 辞書型から配列を取り出します
cn_sd_hl = CNGATEST_dict['cn_sd_hl']
cn_sd_ll_a = CNGATEST_dict['cn_sd_ll_a']
cn_sd_ll_b = CNGATEST_dict['cn_sd_ll_b']
cn_y_hl = CNGATEST_dict['cn_y_hl']
cn_y_ll_a = CNGATEST_dict['cn_y_ll_a']
cn_y_ll_b = CNGATEST_dict['cn_y_ll_b']

# 学習用データと検証用データを用意します
Xtrain = np.vstack([cn_sd_ll_a, cn_sd_hl])
Xval = cn_sd_ll_b

ytrain = np.vstack([cn_y_ll_a, cn_y_hl])
yval = cn_y_ll_b

# 配列のサイズを確認します
print(Xtrain.shape)



# プロットの横軸の値を用意します
wave = np.arange(750, 1551, 2)

fig ,ax = plt.subplots(facecolor ="w")
plt.xlabel('Wavelength␣[nm]')
plt.ylabel('Intensity')
ax.plot(wave, Xtrain[26,:], label ='26th ,CN=53.9')
ax.plot(wave, Xtrain[32,:], label ='32th ,CN=42.8')
ax.legend()
plt.show()



# データの標準化
X_, meanX, stdX = autoscale(Xtrain)
y_, meany, stdy = autoscale(ytrain)
Xval_ = scaling(Xval, meanX, stdX)
yval_ = scaling(yval, meany, stdy)

# リッジ回帰
beta = ridge(X_, y_, mu = 1)
y_hat_ = Xval_ @ beta
y_hat = rescaling(y_hat_, meany, stdy)
rmse, r = pred_eval(yval, y_hat, True)

# PCR
beta = pcr(X_, y_, R = 5)
y_hat_ = Xval_ @ beta
y_hat = rescaling(y_hat_, meany, stdy)
rmes, r = pred_eval(yval, y_hat, plotflg = True)


# PLS(R =20)------------------------------------------
beta, _, _, _, _, _, _ = simpls(X_, y_, R = 5)
y_hat_ = Xval_ @ beta
y_hat = scaling(y_hat_, meany, stdy)
rmse, r = pred_eval(yval, y_hat, True)

print([r, rmse])

# クロスバリデーション------------------------------------------
optLV, press = pls_cv(X_,  y_, maxLV = 30, K = 10)

print(optLV)

# 検量線の学習
beta, _, _, _, _, _, _ = simpls(X_, y_, optLV)
y_hat_ = Xval_ @ beta
y_hat = rescaling(y_hat_, meany, stdy)
rmse, r = pred_eval(yval, y_hat, True)

# Lasso
beta_lasso = lasso(X_, y_, mu = 10, epsilon = 0.01)
y_hat_ = Xval_ @ beta_lasso
y_hat_lasso = rescaling(y_hat_, meany, stdy)
rmse_lasso, r_lasso = pred_eval(yval, y_hat_lasso)

# PLS - beta(20%)
beta, _, _, _, _, _,_ = simpls(X_, y_, R = 7) # 前章のクロスバリデーションの結果より
sel_var_beta20 = pls_beta(beta, share = 0.50)

# データセットの再構築
Xtrain_sel = Xtrain [:, sel_var_beta20]
Xval_sel = Xval [:, sel_var_beta20]
X_sel_, meanX_sel ,stdX_sel = autoscale(Xtrain_sel)
Xval_sel_ = scaling(Xval_sel, meanX_sel, stdX_sel)

# 検量線の学習
optLV_beta20, _ = pls_cv(X_sel_, y_, 20, 10)
beta_beta20, _, _, _, _, _,_ = simpls(X_sel_, y_, optLV_beta20)
y_hat_ = Xval_sel_ @ beta_beta20
y_hat_beta20 = rescaling(y_hat_, meany, stdy)
rmse_beta20, r_beta20 = pred_eval(yval, y_hat_beta20)

# PLS - beta(50%)
beta, _, _, _, _, _,_ = simpls(X_, y_, R = 7)
sel_var_beta50 = pls_beta(beta, share = 0.50)

# データセットの再構築
Xtrain_sel = Xtrain [:, sel_var_beta50]
Xval_sel = Xval [:, sel_var_beta50]
X_sel_, meanX_sel , stdX_sel = autoscale(Xtrain_sel)
Xval_sel_ = scaling(Xval_sel, meanX_sel ,stdX_sel)

# 検量線の学習
optLV_beta50, _ = pls_cv(X_sel_, y_, 20, 10)
beta_beta50, _, _, _, _, _,_ = simpls(X_sel_, y_, optLV_beta50)
y_hat_ = Xval_sel_ @ beta_beta50
y_hat_beta50 = rescaling(y_hat_, meany, stdy)
rmse_beta50, r_beta50 = pred_eval(yval, y_hat_beta50)

# VIP
beta, W, P, Q, T, U, cont = simpls(X_, y_, R = 7)
sel_var_vip, vip = pls_vip(W, Q.T, T, threshold = 1)

# データセットの再構築
Xtrain_sel = Xtrain [:, sel_var_vip]
Xval_sel = Xval [:, sel_var_vip]
X_sel_, meanX_sel, stdX_sel = autoscale(Xtrain_sel)
Xval_sel_ = scaling(Xval_sel, meanX_sel, stdX_sel)

# 検量線の学習
optLV_vip, _ = pls_cv(X_sel_, y_, 20, 10)
beta_vip, _, _, _, _, _, _ = simpls(X_sel_, y_, optLV_vip)
y_hat_ = Xval_sel_ @ beta_vip
y_hat_vip = rescaling(y_hat_, meany, stdy)
rmse_vip, r_vip = pred_eval(yval, y_hat_vip)

# NCSC -VS ------------------------------------------
n_clusters = 10 # NCSC によって分割する変数グループ数
n_sel_clusters = 5 # 検量線に使用する変数グループ数
labels = ncsc(Xtrain.T, n_clusters, gamma = 0.95)
_, sel_var_ncsc = valselect(X_, y_, 7, labels, n_clusters, n_sel_clusters)

# データセットの再構築
Xtrain_sel = Xtrain [:, sel_var_ncsc]
Xval_sel = Xval [:, sel_var_ncsc]
X_sel_, meanX_sel, stdX_sel = autoscale(Xtrain_sel)
Xval_sel_ = scaling(Xval_sel, meanX_sel, stdX_sel)

# 検量線の学習
optLV_ncsc, _ = pls_cv(X_sel_, y_, 20, 10)
beta_ncsc, _, _, _, _, _,_ = simpls(X_sel_, y_, optLV_ncsc)
y_hat_ = Xval_sel_ @ beta_ncsc
y_hat_ncsc = rescaling(y_hat_, meany, stdy)
rmse_ncsc, r_ncsc = pred_eval(yval, y_hat_ncsc)



fig, ax = plt.subplots()
ax.set_xlabel('wavelength')
ax.set_ylabel('Intensity')
for i in range(Xtrain.shape[0]):
    ax.plot(wave, Xtrain [i, :])

xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
ax.vlines(wave [sel_var_ncsc], ymin, ymax, linewidth = 0.5, alpha = 0.2)

num_text = f'#Var={len(sel_var_ncsc)}'
posx =(xmax - xmin)*0.1 + xmin
posy =(ymax - ymin)*0.8 + ymin
ax.text(posx, posy, num_text)

plt.show()