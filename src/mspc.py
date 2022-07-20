import math

import numpy as np
from scale import autoscale, scaling

def mspc_ref(X, numPC):
    """
    SDV を用いてMSPC モデルを学習します

    パラメータ
    ----------
    X: 正常データ行列
    numPC: 採用する主成分数

    戻り値
    -------
    meanX: 正常データの平均
    stdX: 正常データの標準偏差
    U: 左特異ベクトル
    S: 特異値
    V: 右特異ベクトル
    """

    # 標準化
    X, meanX, stdX = autoscale(X)

    # PCA
    U, S, V = np.linalg.svd(X)
    U = U[:,: numPC]
    S = np.diag(S[: numPC])
    V = V[:,: numPC]

    return meanX, stdX, U, S, V

def mspc_T2Q(X, meanX, stdX, U, S, V):
    """
    学習させたMSPCモデルより, 監視対象サンプルの
    T2, Q 統計量を計算します.

    パラメータ
    ----------
    X: 監視対象データ
    meanX: 正常データの平均
    stdX: 正常データの標準偏差
    U: 左特異ベクトル
    S: 特異値
    V: 右特異ベクトル

    戻り値
    -------
    T2: T2 統計量
    Q: Q 統計量
    """

    # 学習データに合わせて標準化する
    X = scaling(X, meanX, stdX)
    
    I = np.eye(X.shape[1])

    # T2，Q の計算
    T2 = np.diag(X @(V @ np.linalg.inv(S @ S)@ V.T)@ X.T)
    Q = np.diag(X @(I - V @ V.T)@ X.T)
    return T2, Q

def cont_T2Q(X, meanX, stdX, U, S, V):
    """
    パラメータ
    ----------
    X: 監視対象サンプル
    meanX: 正常データの平均
    stdX: 正常データの標準偏差
    U: 左特異ベクトル
    S: 特異値
    V: 右特異ベクトル

    戻り値
    -------
    cont_T2: T2 統計量の寄与
    cont_Q: Q 統計量の寄与
    """

    X = scaling(X, meanX, stdX)

    # 寄与の計算
    cont_T2 = np.multiply(X, X @ V @ np.linalg.inv(S @ S/X.shape [0])@ V.T)
    cont_Q = np.power(X @(np.eye(X.shape [1]) - V @ V.T), 2)

    return cont_T2, cont_Q


def mspc_CL(T2, Q, alpha=0.99):
    """
    T2 統計量とQ 統計量の管理限界を計算します

    パラメータ
    ----------
    T2: 管理限界決定用T2 統計量
    Q: 管理限界決定用Q 統計量
    alpha: 信頼区間（デフォルト99 %）

    戻り値
    -------
    meanX: 正常データの平均
    stdX: 正常データの標準偏差
    U: 左特異ベクトル
    S: 特異値
    V: 右特異ベクトル
    """

    sort_T2 = sorted(T2)
    CL_T2 = T2[math.floor(alpha * len(train_data))]

    sort_Q = sorted(Q)
    CL_Q = Q[math.floor(alpha * len(train_data))]