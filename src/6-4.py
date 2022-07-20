import rdata
import pandas as pd

# RData 形式の読み込み
train_parsed = rdata.parser.parse_file('./TEP_FaultFree_Testing.RData')
train_converted = rdata.conversion.convert(train_parsed)

test_parsed = rdata.parser.parse_file('./TEP_Faulty_Testing.RData')
test_converted = rdata.conversion.convert(test_parsed)

# csv ファイルに変換
# 学習用データ
train_data = pd.DataFrame(train_converted ['fault_free_testing'])
train_data = train_data.iloc [0:960, 3:]
train_data.to_csv('normal_data.csv', index = False)

# パラメータチューニング用データ
tuning_data = pd.DataFrame(train_converted ['fault_free_testing'])
tuning_data = tuning_data.iloc [960:1920, 3:]
tuning_data.to_csv('tuning_data.csv', index = False)

# 異常データ
test_data = pd.DataFrame(test_converted ['faulty_testing'])
# 異常ごとに分割してcsv 形式で出力
for i in range(1, 21):
    idv_data = test_data [test_data ['faultNumber '] == i]
    idv_data = idv_data.iloc [0:960, 3:]
    title_name = 'idv '+ str(i)+ '_data.csv'
    idv_data.to_csv(title_name, index = False)