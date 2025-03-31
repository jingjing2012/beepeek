import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning

import calculation_util as cal_util
import common_util
import data_cleaning_util as clean_util
import duplicate_util
import profit_cal
import pt_product_report_parameter as para
import pt_product_report_path as path
import pt_product_sql as sql
from conn import mysql_config as config, sql_engine


def price_st(df, group_col, price_col):
    df = df[df[price_col] > 0]
    df_std = df[price_col].groupby(df[group_col]).agg(['mean', 'std']).reset_index()
    df_std.columns = [group_col, 'price_mean', 'price_std']
    df_std = df.merge(df_std)
    df_std['price_st'] = df_std['price_mean'] + 2 * df_std['price_std']
    df_st = df_std[df_std[price_col] <= df_std['price_st']]
    df_st = df_st[[group_col, price_col]]
    return df_st


def price_k_means(price, k):
    k_model = KMeans(n_clusters=k, random_state=1)
    price_array = price.values.reshape((len(price), 1))
    if len(price_array) <= 5:
        return list()
    k_model.fit(price_array)
    k_model_sort = pd.DataFrame(k_model.cluster_centers_).sort_values(0)
    k_model_price = k_model_sort.rolling(2).mean().iloc[1:]
    k_model_price = k_model_price.round(1)
    price_list = [0] + list(k_model_price[0]) + [round(price.max() + 5, 1)] + [round(price.max() * 100, 1)]
    price_list_unique = sorted(set(price_list))
    return price_list_unique


def price_tag(df, price_col, price_list_col):
    labels = []
    price = df[price_col]
    price_list = list(df[price_list_col][:1])[0]
    price_len = len(price_list)
    if price_len < 1:
        df['price_tag'] = df[price_col]
        return df

    for i in range(0, price_len - 1):
        if i == 0:
            labels.append(str("1:小于" + str(price_list[1])))
        elif i == (price_len - 2):
            labels.append(str(str(i + 1) + ":大于" + str(price_list[i])))
        else:
            labels.append(str(i + 1) + ":" + str(price_list[i]) + "-" + str(price_list[i + 1]))
    price_list.pop()
    price_list.append(price.max() * 10)
    df['price_tag'] = pd.cut(x=price, bins=price_list, labels=labels)
    # df[['price_tag_rank', 'price_tag']] = df['price_tag'].str.split(':', expand=True)
    return df


# 开售月数计算
def month_available(df):
    current_date = pd.to_datetime(datetime.now().date())
    df['date_available'] = pd.to_datetime(df['date_available'], errors='coerce')
    df['available_days'] = (current_date - df['date_available']).dt.days
    # df['头程月数'] = np.where((df['seller_type'] == "FBA") & (df['开售天数'] > 15), 0.5, 0)
    # df['开售月数'] = np.fmax(round(df['开售天数'] / 30 - df['头程月数'], 1), 0.1)
    df['available_months'] = np.fmax(round(df['available_days'] / 30), 1)
    return df


# 警告忽略
pd.set_option('future.no_silent_downcasting', True)
pd.options.mode.chained_assignment = None  # 关闭警告

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

df_ai_jude = sql_engine.connect_pt_product(config.sellersprite_hostname, config.sellersprite_password,
                                           config.clue_position_database, sql.sql_position_ai_jude)

if not df_ai_jude.empty:
    print('ai is running')
    sys.exit()

# 数据连接
df_competitior = sql_engine.connect_pt_product(config.sellersprite_hostname, config.sellersprite_password,
                                               config.clue_position_database, sql.sql_position_competitior)
df_ai = sql_engine.connect_pt_product(config.sellersprite_hostname, config.sellersprite_password,
                                      config.clue_position_database, sql.sql_position_ai)
df_clue = sql_engine.connect_pt_product(config.sellersprite_hostname, config.sellersprite_password,
                                        config.clue_position_database, sql.sql_clue_asin)

if df_competitior.empty:
    print('df_competitior.empty')
    sys.exit()

# 数据格式清洗
df_competitiors = df_competitior[
    ['data_id', 'expand_competitors_id', 'asin', 'price', 'sales', 'ratings', 'date_available']]

clean_util.convert_type(df_competitiors, 'price', 2)
clean_util.convert_type(df_competitiors, 'ratings', 0)
clean_util.convert_date(df_competitiors, 'date_available')

clean_util.convert_type(df_ai, 'overall_competitiveness', 2)
clean_util.convert_type(df_ai, 'similarity_score', 0)

clue_product_list = ['length_max', 'length_mid', 'length_min', 'weight', 'price_value']
for i in clue_product_list:
    clean_util.convert_type(df_clue, i, 2)

# 价格打标
df_price = df_competitiors[['data_id', 'price']]
df_price_st = price_st(df_price, 'data_id', 'price')
df_price_list = df_price_st['price'].groupby(df_price_st['data_id']).apply(lambda x: price_k_means(x, 5)).reset_index()
df_price_list.columns = ['data_id', 'price_list']

df_price_list = df_price.merge(df_price_list)
df_price_tag = df_price_list.groupby(df_price_list['data_id']).apply(
    lambda x: price_tag(x, 'price', 'price_list')).reset_index(drop=True)

df_competitiors = pd.merge(df_competitiors, df_price_tag, how='left', on=['data_id', 'price'])

# 竞争力分组
df_ai['competitive_tag'] = common_util.get_cut(df_ai, 'overall_competitiveness', [-99, -0.5, 0.5, 1, 99],
                                               ['软柿子', '实力相当', '慎重选择', '硬茬'])
df_ai['similarity_tag'] = common_util.get_cut(df_ai, 'similarity_score', [-99, 4, 7, 99], ['低相似', '中等相似', '高相似'])

# 星数分级
cal_util.get_mround(df_competitiors, 'ratings', 'ratings_tag', 100)

# 开售月数计算
month_available(df_competitiors)

# 字段整理
df_tag = pd.merge(df_competitiors, df_ai, how='left', on='expand_competitors_id')
df_tag = df_tag[
    ['expand_competitors_id', 'competitive_tag', 'similarity_tag', 'price_tag', 'ratings_tag', 'available_months']]

df_tag = duplicate_util.df_cleaning(df_tag, 'expand_competitors_id')

# 数据入库
sql_engine.data_to_sql(df_tag, path.pt_clue_tag, 'append', config.connet_clue_position_db_sql)

sql_engine.connect_product(
    config.sellersprite_hostname, config.sellersprite_password, config.clue_position_database, sql.update_price_tag_sql)

# --------------------------------------------毛利测算---------------------------------------------------
# 平均价格计算
competitiors_df = df_competitiors.query('sales > 0')
df_tag_price = competitiors_df[['data_id', 'expand_competitors_id', 'price']]
df_product = df_ai.merge(df_tag_price, how='left', on='expand_competitors_id')
df_product_profit = df_product.query('similarity_tag == "高相似"')
df_price_profit = df_product_profit.groupby(['ASIN', 'site'])['price'].mean().reset_index()

# 规格数据整合
df_profit_product = df_price_profit.merge(df_clue, how='left', on='ASIN')
df_profit = df_profit_product.query('price * length_max * length_mid * length_min * weight * price_value > 0')

if df_profit.empty:
    print('df_profit.empty')
else:
    df_profit_us = df_profit.query('site == "us"')
    df_profit_uk = df_profit.query('site == "uk"')
    df_profit_de = df_profit.query('site == "de"')

    if df_profit_us.empty:
        print('df_profit_us.empty')
    else:
        # 规格数据转换
        df_profit_us[['max_length', 'mid_length', 'min_length', 'weight_pound', 'perimeter', 'weight_max_kg',
                      'weight_max_pound', 'product_fee']] = df_profit_us.apply(lambda row: pd.Series(
            profit_cal.value_convert_us(row['length_max'], row['length_mid'], row['length_min'], row['weight'],
                                        row['price_value'], para.exchange_rate_us)), axis=1)
        # 头程计算
        df_profit_us[['freight_fee', 'freight_fee_air']] = df_profit_us.apply(lambda row: pd.Series(
            profit_cal.freight_fee_cal(row['weight_max_kg'], para.freight_fee_us, para.freight_air_fee_us,
                                       para.exchange_rate_us)), axis=1)
        # 规格打标
        df_profit_us['size_tag'] = df_profit_us.apply(lambda row: pd.Series(
            profit_cal.size_tag_us_cal(row['max_length'], row['mid_length'], row['min_length'], row['perimeter'],
                                       row['weight_max_pound'])), axis=1)
        # fba费用计算
        df_profit_us['fba_fee'] = df_profit_us.apply(
            lambda row: profit_cal.fba_fee_us_cal(row['price'], row['size_tag'], row['weight_pound'],
                                                  row['weight_max_pound']), axis=1)
        # 毛利计算
        df_profit_us[
            ['product_fee_rate', 'freight_fee_rate', 'freight_fee_air_rate', 'fba_fee_rate', 'profit_rate', 'profit',
             'profit_air_rate', 'profit_air']] = df_profit_us.apply(lambda row: pd.Series(
            profit_cal.profit_cal(row['price'], row['product_fee'], row['freight_fee'], row['freight_fee_air'],
                                  row['fba_fee'], para.vat_rate_us)), axis=1)
        if df_profit_uk.empty:
            print('df_profit_uk.empty')
        else:
            # 规格数据转换
            df_profit_uk[
                ['perimeter', 'weight_kg', 'weight_volume_kg', 'weight_max_kg', 'product_fee']] = df_profit_uk.apply(
                lambda row: pd.Series(
                    profit_cal.value_convert_ukde(row['length_max'], row['length_mid'], row['length_min'],
                                                  row['weight'], row['price_value'], para.exchange_rate_uk)), axis=1)
            # 头程计算
            df_profit_uk[['freight_fee', 'freight_fee_air']] = df_profit_uk.apply(lambda row: pd.Series(
                profit_cal.freight_fee_cal(row['weight_max_kg'], para.freight_fee_uk, para.freight_air_fee_uk,
                                           para.exchange_rate_uk)), axis=1)
            # 规格打标
            df_profit_uk['size_tag'] = df_profit_uk.apply(lambda row: pd.Series(
                profit_cal.size_tag_ukde_cal(row['length_max'], row['length_mid'], row['length_min'], row['perimeter'],
                                             row['weight_kg'], row['weight_volume_kg'])), axis=1)
            # fba费用计算
            df_profit_uk['fba_fee'] = df_profit_uk.apply(
                lambda row: profit_cal.fba_fee_uk_cal(row['price'], row['size_tag'], row['weight_kg'],
                                                      row['weight_max_kg']), axis=1)
            # 毛利计算
            df_profit_uk[
                ['product_fee_rate', 'freight_fee_rate', 'freight_fee_air_rate', 'fba_fee_rate', 'profit_rate',
                 'profit',
                 'profit_air_rate', 'profit_air']] = df_profit_uk.apply(lambda row: pd.Series(
                profit_cal.profit_cal(row['price'], row['product_fee'], row['freight_fee'], row['freight_fee_air'],
                                      row['fba_fee'], para.vat_rate_uk)), axis=1)

            if df_profit_de.empty:
                print('df_profit_de.empty')
            else:
                # 规格数据转换
                df_profit_de[
                    ['perimeter', 'weight_kg', 'weight_volume_kg', 'weight_max_kg',
                     'product_fee']] = df_profit_de.apply(
                    lambda row: pd.Series(
                        profit_cal.value_convert_ukde(row['length_max'], row['length_mid'], row['length_min'],
                                                      row['weight'], row['price_value'], para.exchange_rate_de)),
                    axis=1)
                # 头程计算
                df_profit_de[['freight_fee', 'freight_fee_air']] = df_profit_de.apply(lambda row: pd.Series(
                    profit_cal.freight_fee_cal(row['weight_max_kg'], para.freight_fee_de, para.freight_air_fee_de,
                                               para.exchange_rate_de)), axis=1)
                # 规格打标
                df_profit_de['size_tag'] = df_profit_de.apply(lambda row: pd.Series(
                    profit_cal.size_tag_ukde_cal(row['length_max'], row['length_mid'], row['length_min'],
                                                 row['perimeter'], row['weight_kg'], row['weight_volume_kg'])), axis=1)
                # fba费用计算
                df_profit_de['fba_fee'] = df_profit_de.apply(
                    lambda row: profit_cal.fba_fee_de_cal(row['price'], row['size_tag'], row['weight_kg'],
                                                          row['weight_max_kg']), axis=1)
                # 毛利计算
                df_profit_de[
                    ['product_fee_rate', 'freight_fee_rate', 'freight_fee_air_rate', 'fba_fee_rate', 'profit_rate',
                     'profit', 'profit_air_rate', 'profit_air']] = df_profit_de.apply(lambda row: pd.Series(
                    profit_cal.profit_cal(row['price'], row['product_fee'], row['freight_fee'], row['freight_fee_air'],
                                          row['fba_fee'], para.vat_rate_de)), axis=1)

    # 字段整理
    df_profit = pd.concat([df_profit_us, df_profit_uk, df_profit_de], ignore_index=True)
    df_profit = df_profit[
        ['ASIN', 'site', 'price', 'max_length', 'mid_length', 'min_length', 'weight_pound', 'perimeter',
         'weight_max_kg', 'weight_max_pound', 'product_fee', 'freight_fee', 'freight_fee_air', 'size_tag', 'fba_fee',
         'product_fee_rate', 'freight_fee_rate', 'freight_fee_air_rate', 'fba_fee_rate', 'profit_rate', 'profit',
         'profit_air_rate', 'profit_air']]

    profit_list_2 = ['price', 'max_length', 'mid_length', 'min_length', 'perimeter', 'weight_max_kg',
                     'weight_max_pound', 'product_fee', 'freight_fee', 'freight_fee_air', 'fba_fee', 'profit',
                     'profit_air']
    for m in profit_list_2:
        clean_util.convert_type(df_profit, m, 2)

    profit_list_4 = ['weight_pound', 'product_fee_rate', 'freight_fee_rate', 'freight_fee_air_rate', 'fba_fee_rate',
                     'profit_rate', 'profit', 'profit_air_rate']
    for n in profit_list_4:
        clean_util.convert_type(df_profit, n, 4)

    # 数据入库
    sql_engine.data_to_sql(df_profit, path.pt_clue_profit, 'append', config.connet_clue_position_db_sql)

# ----------------------------------------------状态更新----------------------------------------------
sql_engine.connect_product(
    config.sellersprite_hostname, config.sellersprite_password, config.clue_position_database, sql.update_position_sql1)
sql_engine.connect_product(
    config.sellersprite_hostname, config.sellersprite_password, config.clue_position_database, sql.update_position_sql2)
sql_engine.connect_product(
    config.sellersprite_hostname, config.sellersprite_password, config.clue_position_database, sql.update_position_sql3)
