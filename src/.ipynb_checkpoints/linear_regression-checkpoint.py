import matplotlib.pyplot as plt
import numpy as np

from pca import pca
from scale import autoscale, scaling, rescaling


def least_squares(X, y):
    """
    最小二乗法を用いて回帰係数を計算します. 

    パラメータ
    ----------
    X: 入力データ
    y: 出力データ

    戻り値
    -------
    beta: 回帰係数
    """

    # yをベクトル化します
    y = y.reshape(-1, 1)

    # 正規方程式
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    return beta

def ls_est(x, beta):
    """
    線形回帰モデルを用いて出力を予測します. 

    パラメータ
    ----------
    x: 未知サンプル
    beta: 回帰係数

    戻り値
    -------
    y_hat: 予測値
    """

    y_hat = beta.T @ x
    return y_hat

def pcr(X, y, R):
    """
    PCR を用いて回帰係数を計算します. 

    パラメータ
    ----------
    X: 入力データ
    y: 出力データ
    R: 主成分の数

    戻り値
    -------
    beta: 回帰係数
    """

    # yをベクトル化します
    y = y.reshape(-1, 1)

    # 主成分分析を行います
    P, T = pca(X)

    # R 番目までの主成分得点行列を取り出します
    T = T[:,:R]

    # 最小二乗法により回帰係数を求めます
    beta_R = least_squares(T, y)
    beta = P[:,:R] @ beta_R
    return beta

def ridge(X, y, mu=0.1):
    """
    リッジ回帰を用いて回帰係数を計算します. 

    パラメータ
    ----------
    X: 入力データ
    y: 出力データ
    mu: パラメータ(デフォルト: 0.1)

    戻り値
    -------
    beta_ridge: 回帰係数
    """
    # yをベクトル化します
    y = y.reshape(-1, 1)

    # リッジ回帰の回帰係数を求めます
    I = np.eye(X.shape [1])
    beta_ridge = np.linalg.inv(X.T @ X + mu * I)@ X.T @ y
    return beta_ridge

def nipals_pls1(X, y, R):
    """
    NIPALS アルゴリズムを用いて回帰係数を計算します (PLS1).

    パラメータ
    ----------
    X: 入力データ
    y: 出力データ
    R: 潜在変数の数

    戻り値
    -------
    beta: 回帰係数
    W, P, D: PLS 1 モデルのパラメータ
    T: 潜在変数
    """

    # yをベクトル化します
    y = y.reshape(-1, 1)

    # パラメータを保存する変数を作成します
    W = np.zeros((X.shape [1], R))
    P = np.zeros((X.shape [1], R))
    D = np.zeros((R, 1))
    T = np.zeros((X.shape [0], R))

    # NIPALS を計算します
    for r in range(R):
        # 重みを求めます
        w = (X.T @ y)/ np.linalg.norm(X.T @ y)
        # 潜在変数を計算します
        t = X @ w
        # ローディングベクトルを求めます
        p =(X.T @ t)/(t.T @ t)
        # 回帰係数を求めます
        d =(t.T @ y)/(t.T @ t)
        # デフレーションを行います
        X = X - t.reshape(-1, 1) @ p.T
        y = y - t @ d
        # パラメータを格納します
        W[:,r] = w.T
        P[:,r] = p.T
        D[r] = d.T
        T[:,r] = t.T

    # 回帰係数を計算します
    beta = W @ np.linalg.inv(P.T @ W)@ D
    return beta, W, P, D, T


# PLS 2 本体の関数
def nipals_pls2(X, Y, R, epsilon=0.01):
    """
    NIPALS アルゴリズムを用いて回帰係数を計算します（PLS2）．

    パラメータ
    ----------
    X: 入力データ
    Y: 出力データ
    R: 潜在変数の数
    epsilon: （デフォルト: 0.01）

    戻り値
    -------
    beta: 回帰係数
    W, P, Q: PLS 2 モデルのパラメータ
    T: 潜在変数
    """

    # パラメータを保存する変数を作成します
    W = np.zeros((X.shape [1], R))
    P = np.zeros((X.shape [1], R))
    Q = np.zeros((Y.shape [1], R))
    T = np.zeros((X.shape [0], R))

    for r in range(R):
        # アルゴリズム2 よりw, c, t を求めます
        w, c, t = calc_parameter(X, Y, epsilon)
        # ローディングベクトルを求めます
        p =(X.T @ t)/(t.T @ t)
        # 回帰係数を求めます
        q =(Y.T @ t)/(t.T @ t)
        # デフレーションを行います
        X = X - np.outer(t, p)
        Y = Y - np.outer(t, q)
        # パラメータを保存します
        W[:,r] = w
        P[:,r] = p
        Q[:,r] = q
        T[:,r] = t

    # 回帰係数を計算します
    beta = W @ np.linalg.inv(P.T @ W)@ Q.T

    return beta, W, P, Q, T

# アルゴリズム2 を計算する関数
def calc_parameter(X, Y, epsilon):
    u = Y[:,0]
    while True :
        # 重みベクトルw を更新します
        w = X.T @ u / np.linalg.norm(X.T @ u)
        # 潜在変数t を更新します
        t = X @ w
        # 重みベクトルc を更新します
        c = Y.T @ t /np.linalg.norm(Y.T @ t)
        # 潜在変数u を更新します
        u_new = Y @ c
        # 収束判定を行います
        if np.linalg.norm(u_new - u) < epsilon: break
        u = u_new
    return w, c, t

def simpls(X, Y, R):
    """
    SIMPLS アルゴリズムを用いて回帰係数を計算します. 

    パラメータ
    ----------
    X: 入力データ
    Y: 出力データ
    R: 潜在変数の数

    戻り値
    -------
    beta: 回帰係数
    W, P, Q: PLS 1 モデルのパラメータ
    T, U: 潜在変数
    cont: 寄与率
    """
    # yをベクトル化します
    Y = Y.reshape(-1, 1)

    # パラメータを保存する変数を作成します
    W = np.zeros((X.shape [1], R))
    P = np.zeros((X.shape [1], R))
    Q = np.zeros((Y.shape [1], R))
    T = np.zeros((X.shape [0], R))
    U = np.zeros((Y.shape [0], R))
    ssq = np.zeros([R,2])
    ssqX = np.sum(X**2)
    ssqY = np.sum(Y**2)

    for r in range(R):
        # 特異値分をします
        u, s, v = np.linalg.svd(Y.T @ X)
        # 最大特異値に対応する右特異値ベクトルを求めます
        w = v[0,:].T
        # 潜在変数t を求めます
        t = X @ w
        # ローディングベクトルを求めます
        p =(X.T @ t)/(t.T @ t)
        # 回帰係数　q　を求めます
        q =(Y.T @ t)/(t.T @ t)
        # 潜在変数u を求めます
        u = Y @ q
        # デフレーションを行います
        X = X - np.outer(t, p)
        Y = Y - np.outer(t, q)
        ssq [r,0] = np.sum(X**2)/ ssqX
        ssq [r,1] = np.sum(Y**2)/ ssqY
        # パラメータを保存します
        W[:,r] = w.T
        P[:,r] = p.T
        Q[:,r] = q.T
        T[:,r] = t.T
        U[:,r] = u.T

    # 回帰係数を計算します
    beta = W @ np.linalg.inv(P.T @ W)@ Q.T

    # 寄与率を計算します
    cont = np.zeros([R,2])
    cont [0,:] = 1 - ssq [0,:]
    for r in range(1,R):
        cont [r,:] = ssq [r-1,:] - ssq[r,:]

    return beta, W, P, Q, T, U, cont

def pls_cv(X, Y, maxLV , K=10):
    """
    クロスバリデーションでPLS の最適な潜在変数の数を探索します.

    パラメータ
    ----------
    X: 入力データ
    Y: 出力データ
    maxLV: 探索する潜在変数の最大値
    K: データ分割数（デフォルト10）

    戻り値
    -------
    optR: 最適な潜在変数の数
    press: PRESS
    """

    n, m = X.shape
    n, l = Y.shape
    R = np.arange(1, maxLV + 1)
    all_index = [i for i in range(n)]
    # 分割されたデータのサンプル数
    validation_size = n // K
    # クロスバリデーションの結果を保存する変数
    result = np.matrix(np.zeros((K, len(R))))

    # 配列をシャッフルします
    Z = np.hstack((X, Y))
    rng = np.random.default_rng()
    rng.shuffle(Z, axis = 0)
    X = Z[:,0:m]
    Y = Z[:,m:m+l]

    # 各潜在変数に対してクロスバリデーションにてPRESS を 算します
    for i, r in enumerate(R):
        for k in range(K):
            # クロスバリデーション用にデータをK 分割し 
            # 学習用データと検証用データを選択します
            if k != K - 1:
                val_index = all_index [k * validation_size :(k+1)* validation_size - 1]
            else :
                val_index = all_index [k * validation_size :]
            train_index = [i for i in all_index if not i in val_index]
            X_train = X[train_index,:]
            X_val = X[val_index,:]
            Y_train = Y[train_index,:]
            Y_val = Y[val_index,:]

            # 各データを標準化します
            X_train, meanX, stdX = autoscale(X_train)
            Y_train, meanY, stdY = autoscale(Y_train)
            X_val = scaling(X_val, meanX, stdX)

            # 学習用データを用いて回帰係数を計算します
            beta, _, _, _, _, _, _ = simpls(X_train, Y_train, r)

            # 計算した回帰係数からX_val の予測値Y_hat を計算し
            # 元のスケールに戻します
            Y_hat = X_val @ beta
            J = Y_hat.shape [0]
            for j in range(J):
                Y_hat[j,:] = rescaling(Y_hat[j,:], meanY, stdY)

            # PRESS を計算し保存します
            press_val = PRESS(Y_val, Y_hat)
            result [k, i] = press_val
            press = np.sum(result, axis = 0)

    # PRESS が最小となったとき潜在変数を探索します
    optR = R[np.argmin(press)]
    return optR, press

def PRESS(y, y_hat):
    press_val = np.diag((y - y_hat).T @(y - y_hat))
    return press_val


def pred_eval(y, y_hat, plotflg = False):
    """
    RMSE と相関係数を計算します

    パラメータ
    ----------
    y: 真値
    y_hat: 予測値
    pltflg: プロットのOn/ Off(デフォルト: False)

    戻り値
    -------
    rmse: RMSE
    r: 相関係数
    """

    rmse = np.linalg.norm(y - y_hat)/np.sqrt(len(y))
    r = np.corrcoef(y, y_hat)[0,1]

    if plotflg:
        # 散布図をプロットします
        fig, ax = plt.subplots()
        plt.xlabel('Reference ')
        plt.ylabel('Prediction ')
        plt.scatter(y, y_hat)

        # プロットの範囲を取得します
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        plt.plot([xmin,xmax], [ymin,ymax], color = "darkgreen", linestyle = "dashed")

        # 相関係数とRMSE をプロット中に表示します
        # f '' は文字列のフォーマットを指定しています
        r_text = f'r={r:.2f}'
        rmse_text = f'rmse ={ rmse :.2f}'

        posx = (xmax - xmin)*0.1 + xmin
        posy_1 = (ymax - ymin)*0.8 + ymin
        posy_2 = (ymax - ymin)*0.75 + ymin
        ax.text(posx, posy_1, r_text)
        ax.text(posx, posy_2, rmse_text)

        plt.show()

    return rmse, r

def lasso(X, y, mu, epsilon = 0.01):
    """
    Lasso を用いて回帰係数を計算します

    パラメータ
    ----------
    X: 入力データ
    y: 出力データ
    mu: パラメータ
    epsilon: 収束の閾値(デフォルト: 0.01)

    戻り値
    -------
    beta: 回帰係数
    """

    X = np.matrix(X)
    y = np.matrix(y)

    # 解の初期値を適当に定めます
    beta = np.random.rand(X.shape [1])

    j = 1
    while True:
        # リッジ回帰近似
        B = np.diag(beta)
        beta_new = np.linalg.inv(X.T @ X + mu * np.linalg.pinv(abs(B)))@ X.T @ y

        # 収束判定
        if np.linalg.norm(beta - beta_new.T) < epsilon: break

    beta = np.reshape(np.array(beta_new.T), X.shape [1])
    beta = beta.reshape([-1,1])
    return beta

def pls_beta(beta , share = 0.5):
    """
    PLS - beta を用いて, 入力変数を選択します

    パラメータ
    ----------
    beta: PLS の回帰係数
    share: 選択した入力変数の回帰係数が全体の回帰係数に占める割合(デフォルト: 50 %)

    戻り値
    -------
    sel_var: 選択された入力変数のインデックス
    """

    sort_index = np.argsort(abs(beta), axis = 0)[:: -1]
    sort_beta = np.sort(abs(beta), axis = 0)[:: -1]
    cum_sort_beta = np.cumsum(abs(sort_beta))
    sel_var = sort_index [cum_sort_beta <= cum_sort_beta [-1]* share].ravel()
    return sel_var

def pls_vip(W, D, T):
    """
    PLS のVIP を計算します

    パラメータ
    ----------
    W, D: PLS のモデルパラメータ
    T: PLS の潜在変数行列

    戻り値
    -------
    sel_var: 選択された入力変数のインデックス
    vips: 入力変数のVIP
    """

    M, R = W.shape
    weight = np.zeros([R])
    vips = np.zeros([M])

    # 分母の計算
    ssy = np.diag(T.T @ T @ D @ D.T)
    total_ssy = np.sum(ssy)

    # 分子の計算
    for m in range(M):
        for r in range(R):
            weight [r] = np.array([(W[m,r] / np.linalg.norm(W[:,r]))**2])

        vips [m] = np.sqrt(M*(ssy @ weight) / total_ssy)

    sel_var = np.arange(M)
    sel_var = sel_var [vips >=1]
    return sel_var, vips

def kmean(X, num_cls, num_change_lim = 10):
    """
    k- 平均法

    パラメータ
    ----------
    X: 入力データ
    num_cls: 潜在変数の数
    num_change_lim: クラスタ割り当て個数の閾値

    戻り値
    -------
    cls_labels: クラスタのラベル
    """

    np.random.seed(0)
    N,P = X.shape
    # ランダムにクラスタを割り振る
    cls_labels = np.random.randint(0, num_cls, N)

    # クラスタ再割り当て個数が閾値以下になるまで繰り返す
    num_chage = N
    while num_chage > num_change_lim:
        # クラスタ平均を求める
        cls_centers = np.zeros([num_cls,P])
        for c in range(num_cls):
            cls_centers [c,:] = np.mean(X[cls_labels == c,:], 0)

        # クラスタ平均とサンプル間距離を計算し，ラベルを再割り当てする
        dist_all = np.zeros([N,num_cls])
        for c in range(num_cls):
            dist = X - cls_centers [c,:]
            dist_all [:,c] = np.sum(dist**2, 1)
        new_cls_labels = np.argmin(dist_all, 1)

        # 再割り当て個数を計算する
        num_chage = np.sum(new_cls_labels != cls_labels)
        # ラベルを更新する
        cls_labels = new_cls_labels

    return cls_labels

