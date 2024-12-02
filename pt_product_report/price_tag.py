from sklearn.cluster import KMeans
import pandas as pd
import numpy as np


from conn import sql_engine, mysql_config as config
from util import data_cleaning_util
import pt_product_sql as sql


def price_st(df, group_col, price_col):
    df = df[df[price_col] > 0]
    df_std = df[price_col].groupby(df[group_col]).agg(['mean', 'std']).reset_index()
    df_std.columns = [group_col, 'price_mean', 'price_std']
    df_std = df.merge(df_std)
    df_std['price_st'] = df_std['price_mean'] + 2 * df_std['price_std']
    df_st = df_std[df_std[price_col] <= df_std['price_st']]
    df_st = df_st[[group_col, price_col]]
    return df_st


# 定义k值
def price_k(price):
    if len(price) <= 50:
        k = 3
    elif max(price) >= 60:
        k = 7
    else:
        k = 5
    return k


def price_k_means(price):
    k = price_k(price)
    k_model = KMeans(n_clusters=k, random_state=1)
    price_array = price.values.reshape((len(price), 1))
    if len(price_array) <= 5:
        return list()
    k_model.fit(price_array)
    k_model_sort = pd.DataFrame(k_model.cluster_centers_).sort_values(0)
    k_model_price = k_model_sort.rolling(2).mean().iloc[1:]
    k_model_price = k_model_price.round(1)
    price_max = round(price.max() * 100, 2)
    price_list = [0] + list(k_model_price[0]) + [price_max]
    return price_list


def price_tag_rank(price, price_list):
    if len(price_list) < 1:
        return 0
    index = np.digitize([price], price_list)[0]
    return index


# 价格打标
sellersprite_database = 'sellersprite_202410'

df_clue_self = sql_engine.connect_pt_product(config.sellersprite_hostname, config.sellersprite_password,
                                             config.clue_self_database, sql.sql_clue_self)

df_group = sql_engine.connect_pt_product(config.sellersprite_hostname, config.sellersprite_password,
                                         sellersprite_database, sql.sql_group)

df_price = df_group[['sub_category', 'price']]
df_price = df_price.drop_duplicates()
df_price_st = price_st(df_price, 'sub_category', 'price')

df_price_list = df_price_st['price'].groupby(df_price_st['sub_category']).apply(
    lambda x: price_k_means(x)).reset_index()
df_price_list.columns = ['sub_category', 'price_list']

df_price_tag_list = df_price.merge(df_price_list)
df_price_tag_list['price_tag_rank'] = df_price_tag_list.apply(
    lambda row: pd.Series(price_tag_rank(row['price'], row['price_list'])), axis=1).reset_index(drop=True)

df_tag = pd.merge(df_group, df_price_tag_list, how='left', on=['sub_category', 'price'])

df_tag['rank'] = df_tag.groupby(['sub_category', 'price_tag_rank'])['blue_ocean_estimate'].rank(
    ascending=False, method='dense').reset_index(drop=True)

# 将 NaN 替换为 None（MySQL 中的 NULL）
df_tag = df_tag.fillna(value={
    'price_list': '[]',  # 将 NaN 替换为空列表的字符串
    'price_tag_rank': 1,  # 替换为默认值 1
    'rank': 1  # 替换为默认值 1
})

df_tag['duplicate_tag'] = np.where(df_tag['rank'] <= 10, df_tag['rank'], '10+')
df_tag['duplicate_tag'] = np.where(df_tag['sub_category'].str.len() >= 1, df_tag['duplicate_tag'], '1')

# 重复类型打标
df_clue_price_tag_list = df_clue_self.merge(df_price_list, on='sub_category')

df_clue_price_tag_list['price_tag_rank'] = df_clue_price_tag_list.apply(
    lambda row: pd.Series(price_tag_rank(row['price'], row['price_list'])), axis=1).reset_index(drop=True)

data_cleaning_util.convert_str(df_tag, 'price_list')
data_cleaning_util.convert_str(df_clue_price_tag_list, 'price_list')

df_tag = df_tag.merge(df_clue_price_tag_list, how='left', on=['sub_category', 'price_list', 'price_tag_rank'])

df_tag['duplicate_type'] = np.where(df_tag['duplicate_tag'] == "10+", 1, df_tag['duplicate_type'])

df_duplicate = df_tag[['asin', 'price_list', 'price_tag_rank', 'rank', 'duplicate_tag', 'duplicate_type']]

tag_list = ['price_tag_rank', 'rank', 'duplicate_type']
for tag in tag_list:
    data_cleaning_util.convert_type(df_duplicate, tag, 0)

# 将 NaN 替换为 None（MySQL 中的 NULL）
df_duplicate = df_duplicate.fillna(value={
    'duplicate_tag': 1,
    'duplicate_type': 0
})

print('well')

# 构造批量更新数据
update_data = [
    (row['price_list'], row['price_tag_rank'], row['rank'], row['duplicate_tag'], row['duplicate_type'], row['id'])
    for _, row in df_tag.iterrows()
]

# 编写 SQL 更新语句
update_query = """
    # UPDATE pt_product_get_group
    # SET `price_list` = %s, `price_tag_rank` = %s, `rank` = %s, `duplicate_tag` = %s, `duplicate_type` = %s
    # WHERE `id` = %s
"""

batch_size = 10000  # 根据实际情况调整批量大小
for i in range(0, len(update_data), batch_size):
    batch = update_data[i:i + batch_size]
    sql_engine.connect_product_update(
        hostname=config.sellersprite_hostname,
        password=config.sellersprite_password,
        database=sellersprite_database,
        product_sql=update_query,
        update_data=batch
    )

# 使用 executemany 批量更新
sql_engine.connect_product_update(
    hostname=config.sellersprite_hostname,
    password=config.sellersprite_password,
    database=sellersprite_database,
    product_sql=update_query,
    update_data=update_data
)

print('done')
