import numpy as np


def lda(X1, X2):
    """
    固有値問題を用いてLDA の射影軸を計算します．

    パラメータ
    ----------
    X1: クラス1 のデータ
    X2: クラス2 のデータ

    戻り値
    -------
    w: 射影軸
    """

    N, M = X1.shape

    # 各クラスの平均ベクトルを求める
    m1 = np.mean(X1, axis = 0).reshape(-1, 1)
    m2 = np.mean(X2, axis = 0).reshape(-1, 1)

    # クラス内変動行列を求める
    Sw = np.zeros([M, M])
    for i in range(N):
        S1 =(X1[:, i].reshape(-1, 1) - m1)@(X1[:, i].reshape(-1, 1) - m1).T
        S2 =(X2[:, i].reshape(-1, 1) - m2)@(X2[:, i].reshape(-1, 1) - m2).T
        Sw = Sw + S1 + S2

    # クラス内変動行列を求める
    SB = np.outer((m2 - m1), (m2 - m1))

    # 固有値と固有ベクトルを求める
    lam, v = np.linalg.eig(np.linalg.inv(Sw)@ SB)

    # 最大固有に対応する固有ベクトルw を求める
    w = v[:, np.argmax(lam)]

    return w

def calc_score(pred, target):
    """
    分類問題の性能を計算します

    パラメータ
    -------
    pred : 予測ラベル
    target : 正解ラベル

    戻り値
    -------
    sensitivity : 感度
    specificity : 特異度
    g_mean : G-mean
    """

    # 真陽性, 陰性, 偽陽性, 偽陰性
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(0, len(pred)):
        # 真陰性
        if pred [i] == 0 and target [i] == 0:
            TN += 1
        # 偽陰性
        if pred [i] == 0 and target [i] == 1:
            FN += 1
        # 偽陽性
        if pred [i] == 1 and target [i] == 0:
            FP += 1
        # 真陽性
        if pred [i] == 1 and target [i] == 1:
            TP += 1

    # 感度，特異度の計算，ゼロ割を考慮して，if 文で場合分け
    if(TP+FN != 0)and(FP+TN != 0):
        sensitivity = TP /(TP+FN)
        specificity = TN /(FP+TN)
        PPV =TP /(TP+FP)
    else :
        if TP+FN == 0 and FP+TN == 0:
            sensitivity = 0
            specificity = 0
        elif TP+FN == 0 and FP+TN != 0:
            sensitivity = 0
            specificity = TN /(FP+TN)
        else:
            sensitivity = TP /(TP+FN)
            specificity = 0

    # G-mean の計算
    g_mean = np.sqrt(sensitivity * specificity)
    return sensitivity, specificity, g_mean

# データセットを整理するための関数
def make_dataset(df, num):
    """
    データセットのラベルを整理します

    パラメータ
    -------
    df: データセット
    num: 少数データのラベル

    戻り値
    -------
    df: 多数データを0, 少数データを1 としてラベル付けしたデータセット
    """

    for i in range(0, len(df)):
        df_class = df.iloc [i,-1] # データセットの最終列がラベル
        if df_class == num :
            df.iloc [i,-1] = 1 # 少数クラス
        else :
            df.iloc [i,-1] = 0 # 多数クラス

    return df

