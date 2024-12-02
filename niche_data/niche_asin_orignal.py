import pandas as pd
import niche_data_path as path
from better.conn import sql_engine


def asin_csv(path):
    df = pd.read_csv(path)
    return df


def niche_save_to_sql(df, table, args, niche_conn):
    # if not ("niche_original".__eq__(table) or "niche_size_original".__eq__(table)):
    #     niche_conn.execute("DELETE FROM " + table)
    df.to_sql(table, niche_conn, if_exists=args, index=False)


def asin_clear(df):
    df = df.drop_duplicates(subset=['所属利基'])
    df = df.dropna(subset=['所属利基'], inplace=False)
    df = df.dropna(axis=0, how='all', thresh=None, subset=None, inplace=False)  # 只有全部为空才回被删除

    df_float_1 = df['点击份额_360'].str.strip("%").astype(float) / 100
    df['点击份额_360'] = df_float_1.round(decimals=4)
    df_float_2 = df['平均bsr'].astype(float)
    df['平均bsr'] = df_float_2.round(decimals=2)

    return df


# df_1_1 = asin_csv(path.niche_asin_us_1_1)
# df_1_1.insert(loc=0, column='数据更新时间', value='2023-01-01')
#
# df_1_8 = asin_csv(path.niche_asin_us_1_8)
# df_1_8.insert(loc=0, column='数据更新时间', value='2023-01-08')
#
# df_1_22 = asin_csv(path.niche_asin_us_1_22)
# df_1_22.insert(loc=0, column='数据更新时间', value='2023-01-22')
#
# df_11_13 = asin_csv(path.niche_asin_us_11_13)
# df_11_13.insert(loc=0, column='数据更新时间', value='2022-11-13')
#
# df_11_20 = asin_csv(path.niche_asin_us_11_20)
# df_11_20.insert(loc=0, column='数据更新时间', value='2022-11-20')
#
# df_11_27 = asin_csv(path.niche_asin_us_11_27)
# df_11_27.insert(loc=0, column='数据更新时间', value='2022-11-27')
#
# df_12_4 = asin_csv(path.niche_asin_us_12_4)
# df_12_4.insert(loc=0, column='数据更新时间', value='2022-12-04')
#
# df_12_11 = asin_csv(path.niche_asin_us_12_11)
# df_12_11.insert(loc=0, column='数据更新时间', value='2022-12-11')
#
# df_12_18 = asin_csv(path.niche_asin_us_12_18)
# df_12_18.insert(loc=0, column='数据更新时间', value='2022-12-18')
#
# df_12_25 = asin_csv(path.niche_asin_us_12_25)
# df_12_25.insert(loc=0, column='数据更新时间', value='2022-12-25')
#
# df_asin_1 = pd.concat([df_1_1, df_1_8, df_1_22], ignore_index=True)
# df_asin_1 = asin_clear(df_asin_1)
#
# df_asin_2 = pd.concat([df_11_13, df_11_20, df_11_27], ignore_index=True)
# df_asin_2 = asin_clear(df_asin_2)
#
# df_asin_3 = pd.concat([df_12_4, df_12_11, df_12_18, df_12_25], ignore_index=True)
# df_asin_3 = asin_clear(df_asin_3)

df_asin = asin_csv(path.niche_asin_us)
df_asin.insert(loc=0, column='数据更新时间', value='2023-01-29')
df_asin = asin_clear(df_asin)

# print(df_asin.head())

print("niche_asin")

conn = sql_engine.create_niche_conn()
# niche_save_to_sql(df_asin_1, path.niche_asin_original, "append", conn)
# niche_save_to_sql(df_asin_2, path.niche_asin_original, "append", conn)
# niche_save_to_sql(df_asin_3, path.niche_asin_original, "append", conn)

niche_save_to_sql(df_asin, path.niche_asin_original, "append", conn)
print("asin")
