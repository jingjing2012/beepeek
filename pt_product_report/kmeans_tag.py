import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import data_cleaning_util
import duplicate_util
import pt_product_report_path as path
import pt_product_sql as sql
import db_util
from conn import sql_engine, mysql_config as config


def tag_st(df, group_col, tag_col):
    df = df[df[tag_col] > 0]
    df[tag_col] = df[tag_col].round(0)
    df = df.drop_duplicates()
    df_std = df[tag_col].groupby(df[group_col]).agg(['mean', 'std']).reset_index()
    df_std.columns = [group_col, 'tag_mean', 'tag_std']
    df_std = df.merge(df_std)
    df_std['tag_st'] = df_std['tag_mean'] + 2 * df_std['tag_std']
    df_st = df_std[df_std[tag_col] <= df_std['tag_st']]
    df_st = df_st[[group_col, tag_col]]
    return df_st


# 定义k值
def tag_k(tag_col):
    if len(tag_col) <= 50:
        k = 3
    elif max(tag_col) >= 60:
        k = 7
    else:
        k = 5
    return k


def tag_k_means(tag_col):
    k = tag_k(tag_col)
    k_model = KMeans(n_clusters=k, random_state=1)
    tag_array = tag_col.values.reshape((len(tag_col), 1))
    if len(tag_array) <= 5:
        return list()
    k_model.fit(tag_array)
    k_model_sort = pd.DataFrame(k_model.cluster_centers_).sort_values(0)
    k_model_tag = k_model_sort.rolling(2).mean().iloc[1:]
    k_model_tag = k_model_tag.round(1)
    tag_max = round(tag_col.max() * 100, 2)
    tag_col = [0] + list(k_model_tag[0]) + [tag_max]
    tag_col_unique = sorted(set(tag_col))
    return tag_col_unique


def tag_tag_rank(tag, tag_col):
    if len(tag_col) < 1:
        return 0
    index = np.digitize([tag], tag_col)[0]
    return index


# -------------------------------------beepeek数据去重打标-------------------------------------

sites = ['us', 'uk', 'de', 'fr']

for site in sites:

    sellersprite_database = config.sellersprite_database + '_' + str(site)

    df_get_group = sql_engine.connect_pt_product(
        config.sellersprite_hostname, config.sellersprite_password, sellersprite_database, sql.sql_get_group
    )

    df_get_group_status = sql_engine.connect_pt_product(
        config.sellersprite_hostname, config.sellersprite_password, sellersprite_database, sql.sql_get_group_status
    )

    print(site)

    if df_get_group_status.empty and (not df_get_group.empty):

        # 价格打标
        df_clue_self = sql_engine.connect_pt_product(config.sellersprite_hostname, config.sellersprite_password,
                                                     config.clue_self_database, sql.sql_clue_self)

        df_group = sql_engine.connect_pt_product(config.sellersprite_hostname, config.sellersprite_password,
                                                 sellersprite_database, sql.sql_group)

        # 对价格进行打标
        df_price = df_group[['sub_category', 'price']]
        df_price_st = tag_st(df_price, 'sub_category', 'price')

        df_price_list = df_price_st['price'].groupby(df_price_st['sub_category']).apply(
            lambda x: tag_k_means(x)).reset_index()
        df_price_list.columns = ['sub_category', 'price_list']

        df_price_tag_list = df_price.merge(df_price_list)
        df_price_tag_list['price_tag_rank'] = df_price_tag_list.apply(
            lambda row: pd.Series(tag_tag_rank(row['price'], row['price_list'])), axis=1).reset_index(drop=True)

        # 对FBA费用进行打标
        df_fba_fees = df_group[['sub_category', 'fba_fees']]
        df_fba_fees_st = tag_st(df_fba_fees, 'sub_category', 'fba_fees')

        df_fba_fees_list = df_fba_fees_st['fba_fees'].groupby(df_fba_fees_st['sub_category']).apply(
            lambda x: tag_k_means(x)).reset_index()
        df_fba_fees_list.columns = ['sub_category', 'fba_fees_list']

        df_fba_fees_tag_list = df_fba_fees.merge(df_fba_fees_list)
        df_fba_fees_tag_list['fba_fees_tag_rank'] = df_fba_fees_tag_list.apply(
            lambda row: pd.Series(tag_tag_rank(row['fba_fees'], row['fba_fees_list'])), axis=1).reset_index(drop=True)

        # 重复度计算
        df_tag = df_group.merge(df_price_tag_list, how='left', on=['sub_category', 'price'])
        df_tag = duplicate_util.df_cleaning(df_tag, 'asin')

        df_tag = df_tag.merge(df_fba_fees_tag_list, how='left', on=['sub_category', 'fba_fees'])
        df_tag = duplicate_util.df_cleaning(df_tag, 'asin')

        # 将 NaN 替换为 None（MySQL 中的 NULL）
        df_tag = df_tag.fillna(value={
            'price_list': '[]',  # 将 NaN 替换为空列表的字符串
            'price_tag_rank': 1,  # 替换为默认值 1
            'fba_fees_list': '[]',  # 将 NaN 替换为空列表的字符串
            'fba_fees_tag_rank': 1  # 替换为默认值 1
        })

        tag_kmeans_list = ['price_list', 'fba_fees_list']
        for tag_kmeans in tag_kmeans_list:
            data_cleaning_util.convert_str(df_tag, tag_kmeans)

        df_tag['rank'] = \
            df_tag.groupby(['sub_category', 'price_list', 'price_tag_rank', 'fba_fees_list', 'fba_fees_tag_rank'])[
                'blue_ocean_estimate'].rank(ascending=False, method='dense')
        df_tag['rank'] = df_tag['rank'].round(0)

        df_tag['duplicate_tag'] = np.where(df_tag['rank'] <= 10, df_tag['rank'], '10+')

        duplicate_tag_conditions = (df_tag['sub_category'].str.len() >= 1) & (df_tag['price_list'].str.len() >= 1) & (
                df_tag['fba_fees_list'].str.len() >= 1)

        df_tag['duplicate_tag'] = np.where(duplicate_tag_conditions, df_tag['duplicate_tag'], '0')

        # 重复类型打标
        df_clue_price_tag_list = df_clue_self.merge(df_price_list, on='sub_category').merge(df_fba_fees_list,
                                                                                            on='sub_category')

        df_clue_price_tag_list['price_tag_rank'] = df_clue_price_tag_list.apply(
            lambda row: pd.Series(tag_tag_rank(row['price'], row['price_list'])), axis=1).reset_index(drop=True)

        df_clue_price_tag_list['fba_fees_tag_rank'] = df_clue_price_tag_list.apply(
            lambda row: pd.Series(tag_tag_rank(row['fba_fees'], row['fba_fees_list'])), axis=1).reset_index(drop=True)

        for tag_kmeans in tag_kmeans_list:
            data_cleaning_util.convert_str(df_clue_price_tag_list, tag_kmeans)

        df_tag = df_tag.merge(df_clue_price_tag_list, how='left',
                              on=['sub_category', 'price_list', 'price_tag_rank', 'fba_fees_list', 'fba_fees_tag_rank'])

        df_tag = duplicate_util.df_cleaning(df_tag, 'asin')

        df_tag['duplicate_type'] = np.where(df_tag['duplicate_tag'] == "0", 0, df_tag['duplicate_type'])
        df_tag['duplicate_type'] = np.where(df_tag['duplicate_tag'] == "10+", 1, df_tag['duplicate_type'])

        df_duplicate = df_tag[
            ['asin', 'price_list', 'price_tag_rank', 'fba_fees_list', 'fba_fees_tag_rank', 'rank', 'duplicate_tag',
             'duplicate_type']]

        tag_list = ['price_tag_rank', 'fba_fees_tag_rank', 'rank', 'duplicate_type']
        for tag in tag_list:
            data_cleaning_util.convert_type(df_duplicate, tag, 0)

        # 将 NaN 替换为 None（MySQL 中的 NULL）
        df_duplicate = df_duplicate.fillna(value={
            'duplicate_tag': 1,
            'duplicate_type': 0
        })

        df_duplicate = duplicate_util.df_cleaning(df_duplicate, 'asin')

        print('well')

        # 数据入库
        sql_engine.data_to_sql(
            df_duplicate, path.pt_product_duplicate, 'append', db_util.connet_sellersprite_db_sql(sellersprite_database)
        )

        # 状态更新
        sql_engine.connect_product(
            config.sellersprite_hostname, config.sellersprite_password, sellersprite_database, sql.sql_duplicate_update
        )

        print('done')

    else:
        continue
