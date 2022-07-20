import pandas as pd

from func5 import make_dataset

# Cover_type
# 少数クラス: Cottonwood / Willow(class = 4)多数クラス: Ponderosa Pine(class =3)
df1 = pd.read_csv("covertype.csv", sep = ",", header = None)
df1_major = df1.loc[df1.iloc[:,-1] == 3]
df1_minor = df1.loc[df1.iloc[:,-1] == 4]
df1 = pd.merge(df1_major, df1_minor, how = 'outer').reset_index(drop = True)
df1 = make_dataset(df1, 4)
df1.to_csv("cover_type_im.csv", index = None)

# Abalone
# 少数クラス: 9 才, 多数クラス: 18 才
df4 = pd.read_csv("abalone.csv", sep = ",", header = None)
df4_major = df4.loc[df4.iloc[:,-1] == 9]
df4_minor = df4.loc[df4.iloc[:,-1] == 18]
df4 = pd.merge(df4_major, df4_minor, how = 'outer').reset_index(drop = True)
# 項目‘‘ SEX '' がカテゴリカル変数なので，整数値に置換します(Male: 0, Female: 1, Infant: 2)
df4.replace({"M":0, "F":1, "I":2}, inplace = True)
df4 = make_dataset(df4, 18)
df4.to_csv("abalone_im.csv", index = None)

# CTG
# 少数クラス: type_ 3(class = 3)，多数クラス: その他
df3 = pd.read_csv("ctg.csv", sep=",", header = 0)
df3 = make_dataset(df3, 3)
df3.to_csv("CTG_im.csv", index = None)

# Pageblocks
# 少数クラス: graphic(class = 5)，多数クラス: その他
df2 = pd.read_csv("pageblocks.csv", sep = "\s+", header = None)
df2 = make_dataset(df2, 5)
df2.to_csv("pageblocks_im.csv", index = None)