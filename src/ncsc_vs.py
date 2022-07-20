import numpy as np
from sklearn.cluster import SpectralClustering
from linear_regression import simpls

def nc(X, gamma=0.99):
    """
    NC 法を用いた類似度行列の計算

    パラメータ
    ----------
    X: 入力データ
    gamma : 相係関係の有無の判定の閾値(デフォルト: 0.99)

    戻り値
    -------
    S: 類似度行列    
    """

    N, _ = X.shape
    S = np.zeros([N,N])
    X = X.T

    # 類似度行列を計算します
    for i in range(N):
        # クエリをすべてのサンプルから引きます
        Xq = X[:,i].reshape(-1, 1)
        Xs = X
        Xm = Xs - Xq

    # 相関係数を計算します
    V =(Xm.T @ Xm)/(N - 1)
    d = np.sqrt(np.diag(V)).reshape(-1, 1)
    D = d @ d.T
    R = np.divide(V, D, out = np.zeros_like(V), where = D != 0)
    R = np.nan_to_num(R) # NaN を除去します
    ZERO_DIAG =(np.eye(N) - 1)* -1
    R = R * ZERO_DIAG

    # 相関関係を有するサンプルのペアに重みを与えます
    R = np.abs(R)
    R[R > gamma] = 1
    R[R < gamma] = 0

    # 類似度行列を更新します
    S += R
    return S


def ncsc(X, n_clusters, gamma=0.99):
    """
    NCSC を用いて相関関係に基づいたクラスタリングを実行します

    パラメータ
    ----------
    X: 入力データ
    n_clusters: 分割するクラスタの数
    gamma: 相関関係の有無の判定の閾値(デフォルト: 0.99)

    戻り値
    -------
    labels: それぞれのサンプルのクラスタラベル
    """

    # NC 法による類似度行列の構築
    S = nc(X)

    # スペクトラルクラスタリングの実行
    clustering = SpectralClustering(n_clusters, affinity = 'precomputed',
                                    assign_labels ='discretize',
                                    random_state =0).fit(S)
    labels = clustering.labels_
    return labels


def valselect(X, Y, R, labels, n_clusters, n_sel_clusters):
    """
    NCSC の結果を用いて変数選択します．

    パラメータ
    ----------
    X: 入力データ
    Y: 出力データ
    R: 潜在変数の数
    labels: それぞれのサンプルごとのクラスタラベル(NCSC の出力)
    n_clusters: NCSC によって分割されたクラスタの数
    n_sel_clusters: 選択するクラスタの数

    戻り値
    -------
    sel_culsters: 選択された変数グループのインデックス
    sel_var: 選択された変数グループに属する変数のインデックス
    """

    labels = np.array(labels)
    sumcont = np.zeros([n_clusters])
    for c in range(n_clusters):
        idx = labels == c
        X_cls = X[:,idx]
        # 変数クラスタごとにPLS を学習させて，寄与率を計算します．
        if len(X_cls) != 0:
            _, _, _, _, _, _, cont = simpls(X_cls, Y, R)
            sumcont [c] = np.sum(cont [:,1])

    # 計算された寄与率から変数グループを選択します
    idx = np.argsort(sumcont)[:: -1]
    sel_culsters = idx [: n_sel_clusters]
    sel_var = [i for i, l in enumerate(labels)if l in sel_culsters]

    return sel_culsters, sel_var