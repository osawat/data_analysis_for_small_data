import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RUSBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from imblearn.ensemble import RUSBoostClassifier
from imblearn.over_sampling import ADASYN, SMOTE

from func5 import calc_score
from scale import autoscale

# Covertype を読み込みます
# ファイル名は変更してください．
df = pd.read_csv("data/cover_type_im.csv")

# モデルインスタンス生成
model = RandomForestClassifier()

# model = AdaBoostClassifier()

# model = RUSBoostClassifier ()

# # モデルインスタンス生成
# ovs = SMOTE(random_state = 42)
# model = DecisionTreeClassifier()
# ovs = ADASYN(random_state = 42)
# model = DecisionTreeClassifier()


# 結果をまとめるリスト
_result_lst = []

# データセットの準備
# 学習用データと検証用データを10 回ランダムに組み替え
kf = KFold(n_splits = 10, shuffle = True, random_state = 42)
Y = df.iloc [:, -1] # 正解ラベルの抽出
X = df.drop(df.columns [-1], axis = 1)# 正解ラベルの削除

# 学習と検証の繰り返し
for train_index, test_index in kf.split(X):
    # train データとなる行のみ抽出します
    X_train = X.iloc [train_index, :].values
    # train データの正解ラベル
    Y_train = Y.iloc [train_index]
    # test データの行を抽出します
    X_test = X.iloc [test_index, :].values
    # test データの正解ラベル
    Y_test = Y.iloc [test_index]

    # # オーバーサンプリング
    # X_train, Y_train = ovs.fit_resample(X_train, Y_train)

# データの標準化
X_train, mean, std = autoscale(X_train)
X_test = autoscale(X_test, mean, std)

# モデルを学習させます
model.fit(X_train, Y_train)

# 検証用データのラベルの予測
Y_pred = model.predict(X_test)
# 性能評価
sense, spec, g_mean = calc_score(Y_pred, Y_test)