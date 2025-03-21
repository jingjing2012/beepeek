import numpy as np
import pandas as pd
from datetime import datetime
import warnings

from conn import mysql_config as config, sql_engine
from util import data_cleaning_util as clean_util, common_util, duplicate_util
import pt_product_sql as sql
import profit_cal
import pt_product_report_path as path
import pt_product_report_parameter as para


# 开售月数计算
def month_available(df):
    current_date = pd.to_datetime(datetime.now().date())
    df['date_available'] = pd.to_datetime(df['date_available'], errors='coerce')
    df['day_available'] = (current_date - df['date_available']).dt.days
    df['first_month'] = np.where((df['seller_type'] == "FBA") & (df['day_available'] > 15), 0.5, 0)
    df['month_available'] = np.fmax(round(df['day_available'] / 30 - df['first_month'], 1), 0.1)
    return df


# hhi计算
def hhi_cal(df, hhi_cal_col):
    df_hhi_cal_data = df[df[hhi_cal_col].notnull()]
    df_hhi_cal = df_hhi_cal_data.groupby(['buybox_seller_id', hhi_cal_col])['monthly_revenue'].sum().reset_index()

    df_hhi_cal['monthly_revenue_proportion'] = df_hhi_cal['monthly_revenue'] / df_hhi_cal.groupby('buybox_seller_id')[
        'monthly_revenue'].transform('sum')
    df_hhi_cal['monthly_revenue_square'] = pow(df_hhi_cal['monthly_revenue_proportion'], 2)

    df_hhi = df_hhi_cal.groupby('buybox_seller_id')['monthly_revenue_square'].sum().reset_index().rename(
        columns={'monthly_revenue_square': hhi_cal_col + '_hhi'})
    return df_hhi


# 临时忽略 SettingWithCopyWarning
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# 状态码更新
sql_engine.connect_product(config.sellersprite_hostname, config.sellersprite_password, config.clue_shop_database,
                           sql.sql_report_brand_status)
sql_engine.connect_product(config.sellersprite_hostname, config.sellersprite_password, config.clue_shop_database,
                           sql.sql_report_seller_status)
sql_engine.connect_product(config.sellersprite_hostname, config.sellersprite_password, config.clue_shop_database,
                           sql.sql_seller_brand_status)
sql_engine.connect_product(config.sellersprite_hostname, config.sellersprite_password, config.clue_shop_database,
                           sql.sql_seller_seller_status)

# 数据连接
df_brand_report = sql_engine.connect_pt_product(config.sellersprite_hostname, config.sellersprite_password,
                                                config.clue_shop_database, sql.sql_brand_report)

df_seller_product = sql_engine.connect_pt_product(config.sellersprite_hostname, config.sellersprite_password,
                                                  config.clue_shop_database, sql.sql_seller_product)

# 数据格式清洗
df_brand_report['task_asin'] = df_brand_report['asin'] + ' | ' + df_brand_report['task_tag']
df_brand_report['buybox_seller_id'] = df_brand_report['buybox_seller'] + ' | ' + df_brand_report['task_tag']

df_brand_report = duplicate_util.df_cleaning(df_brand_report, 'task_asin')

report_list_0 = ['sales', 'monthly_revenue', 'variations', 'ratings']
for i in report_list_0:
    clean_util.convert_type(df_brand_report, i, 0)

report_list_2 = ['price', 'rating', 'fba_fees']
for j in report_list_2:
    clean_util.convert_type(df_brand_report, j, 2)

clean_util.convert_str(df_brand_report, 'category')

df_seller_product['task_asin'] = df_seller_product['asin'] + ' | ' + df_seller_product['task_tag']
df_seller_product['buybox_seller_id'] = df_seller_product['buybox_seller'] + ' | ' + df_seller_product['task_tag']

clean_util.convert_str_lower(df_seller_product, 'weight')
clean_util.convert_str_lower(df_seller_product, 'dimensions')
clean_util.convert_str(df_seller_product, 'seller_type')
clean_util.convert_type(df_seller_product, 'price', 2)

# 新品数计算
month_available(df_brand_report)

df_brand_report_90 = df_brand_report[df_brand_report['month_available'] <= 3]
df_seller_90 = df_brand_report_90.groupby('buybox_seller_id')['task_asin'].count().reset_index().rename(
    columns={'task_asin': 'asin_count_new_90'})

df_brand_report_180 = df_brand_report[df_brand_report['month_available'] <= 6]
df_seller_180 = df_brand_report_180.groupby('buybox_seller_id')['task_asin'].count().reset_index().rename(
    columns={'task_asin': 'asin_count_new_180'})

# 配送占比计算
df_brand_report_revenue = df_brand_report[df_brand_report['monthly_revenue'] > 0]
df_brand_report_revenue_count = df_brand_report_revenue.groupby('buybox_seller_id')[
    'task_asin'].count().reset_index().rename(
    columns={'task_asin': 'asin_count_revenue'})

df_brand_report_revenue_fbm = df_brand_report_revenue[df_brand_report_revenue['seller_type'] == 'FBM']
df_brand_report_revenue_count_fbm = df_brand_report_revenue_fbm.groupby('buybox_seller_id')[
    'task_asin'].count().reset_index().rename(columns={'task_asin': 'asin_count_revenue_fbm'})

df_seller_revenue_fbm = df_brand_report_revenue_count.merge(df_brand_report_revenue_count_fbm, how='left',
                                                            on='buybox_seller_id')
df_seller_revenue_fbm['asin_count_revenue_fbm_rate'] = df_seller_revenue_fbm['asin_count_revenue_fbm'] / \
                                                       df_seller_revenue_fbm['asin_count_revenue']

# 配送数据计算
df_brand_report_fbm = df_brand_report[df_brand_report['seller_type'] == 'FBM']
df_seller_fbm = df_brand_report_fbm.groupby('buybox_seller_id').agg(
    {'monthly_revenue': 'sum', 'task_asin': 'nunique'}).reset_index().rename(
    columns={'monthly_revenue': 'revenue_fbm', 'task_asin': 'asin_count_fbm'})

df_brand_report_revenue_fba = df_brand_report_revenue[df_brand_report_revenue['seller_type'] == 'FBA']
df_brand_report_revenue_count_fba = df_brand_report_revenue_fba.groupby('buybox_seller_id')[
    'task_asin'].count().reset_index().rename(columns={'task_asin': 'asin_count_revenue_fba'})

df_seller_revenue_fba = df_brand_report_revenue_count.merge(df_brand_report_revenue_count_fba, how='left',
                                                            on='buybox_seller_id')
df_seller_revenue_fba['asin_count_revenue_fba_rate'] = df_seller_revenue_fba['asin_count_revenue_fba'] / \
                                                       df_seller_revenue_fba['asin_count_revenue']

# TOP5占比计算
df_brand_report['revenue_rank'] = df_brand_report.groupby('buybox_seller_id')['monthly_revenue'].rank(ascending=False,
                                                                                                      method='first')

df_brand_report_top5 = df_brand_report[df_brand_report['revenue_rank'] <= 5]

df_seller_top5_avg = df_brand_report_top5.groupby('buybox_seller_id')['monthly_revenue'].mean().reset_index().rename(
    columns={'monthly_revenue': 'revenue_top5_avg'})

# 卖家所属地获取
df_seller_location = df_brand_report[df_brand_report['revenue_rank'] <= 1]
df_seller_location = df_seller_location[['buybox_seller_id', 'buybox_location']]

# hhi计算
df_category_hhi = hhi_cal(df_brand_report, 'category')
df_brand_hhi = hhi_cal(df_brand_report, 'brand')

# 数据打标
df_brand_report['revenue_tag'] = common_util.get_cut(df_brand_report, 'monthly_revenue', para.revenue_list,
                                                     para.revenue_tag)
df_brand_report['revenue_tag_rank'] = common_util.get_cut(df_brand_report, 'monthly_revenue', para.revenue_list,
                                                          para.revenue_tag_rank)

df_brand_report['month_available_tag'] = common_util.get_cut(df_brand_report, 'month_available',
                                                             para.month_available_list, para.month_available_tag)
df_brand_report['month_available_tag_rank'] = common_util.get_cut(df_brand_report, 'month_available',
                                                                  para.month_available_list,
                                                                  para.month_available_tag_rank)

df_brand_report['price_tag'] = common_util.get_cut(df_brand_report, 'price', para.price_list,
                                                   para.price_tag)
df_brand_report['price_tag_rank'] = common_util.get_cut(df_brand_report, 'price', para.price_list, para.price_tag_rank)

df_brand_report['rating_tag'] = common_util.get_cut(df_brand_report, 'rating', para.rating_list, para.rating_tag)
df_brand_report['rating_tag_rank'] = common_util.get_cut(df_brand_report, 'rating', para.rating_list,
                                                         para.rating_tag_rank)

df_brand_report['ratings_tag'] = common_util.get_cut(df_brand_report, 'ratings', para.ratings_list, para.ratings_tag)
df_brand_report['ratings_tag_rank'] = common_util.get_cut(df_brand_report, 'ratings', para.ratings_list,
                                                          para.ratings_tag_rank)

# B+等级产品占比
df_brand_report_b = df_brand_report[df_brand_report['monthly_revenue'] >= 6000]
df_brand_report_count_b = df_brand_report_b.groupby('buybox_seller_id')['task_asin'].count().reset_index().rename(
    columns={'task_asin': 'asin_count_b'})
df_brand_report_revenue_b = df_brand_report_b.groupby('buybox_seller_id')['monthly_revenue'].sum().reset_index().rename(
    columns={'monthly_revenue': 'asin_revenue_b'})

df_seller_b = df_brand_report_count_b.merge(df_brand_report_revenue_b, how='left', on='buybox_seller_id')

# C级产品占比
df_brand_report_c = df_brand_report[df_brand_report['revenue_tag'] == 'C']
df_brand_report_count_c = df_brand_report_c.groupby('buybox_seller_id')['task_asin'].count().reset_index().rename(
    columns={'task_asin': 'asin_count_c'})
df_brand_report_revenue_c = df_brand_report_c.groupby('buybox_seller_id')['monthly_revenue'].sum().reset_index().rename(
    columns={'monthly_revenue': 'asin_revenue_c'})

df_seller_c = df_brand_report_count_c.merge(df_brand_report_revenue_c, how='left', on='buybox_seller_id')

# 数据聚合
df_seller_group = df_brand_report.groupby('buybox_seller_id').agg(
    {'monthly_revenue': 'sum', 'task_asin': 'nunique', 'parent': 'nunique', 'price': 'mean', 'fba_fees': 'mean',
     'brand': 'nunique'}).reset_index().rename(
    columns={'monthly_revenue': 'gmv', 'task_asin': 'asin_count', 'parent': 'parent_count', 'price': 'price_avg',
             'fba_fees': 'fba_fees_avg', 'brand': 'brand_count'})

df_seller = df_seller_group.merge(df_seller_location, how='left', on='buybox_seller_id') \
    .merge(df_seller_90, how='left', on='buybox_seller_id') \
    .merge(df_seller_180, how='left', on='buybox_seller_id') \
    .merge(df_seller_revenue_fbm, how='left', on='buybox_seller_id') \
    .merge(df_seller_fbm, how='left', on='buybox_seller_id') \
    .merge(df_seller_revenue_fba, how='left', on='buybox_seller_id') \
    .merge(df_seller_top5_avg, how='left', on='buybox_seller_id') \
    .merge(df_category_hhi, how='left', on='buybox_seller_id') \
    .merge(df_brand_hhi, how='left', on='buybox_seller_id') \
    .merge(df_seller_b, how='left', on='buybox_seller_id') \
    .merge(df_seller_c, how='left', on='buybox_seller_id')

df_seller['asin_new_rate_90'] = df_seller['asin_count_new_90'] / df_seller['asin_count']
df_seller['asin_new_rate_180'] = df_seller['asin_count_new_180'] / df_seller['asin_count']

df_seller['revenue_fbm_rate'] = df_seller['revenue_fbm'] / df_seller['gmv']
df_seller['asin_count_fbm_rate'] = df_seller['asin_count_fbm'] / df_seller['asin_count']

df_seller['asin_b_rate'] = df_seller['asin_count_b'] / df_seller['asin_count']
df_seller['revenue_b_rate'] = df_seller['asin_revenue_b'] / df_seller['gmv']

df_seller['asin_c_rate'] = df_seller['asin_count_c'] / df_seller['asin_count']
df_seller['revenue_c_rate'] = df_seller['asin_revenue_c'] / df_seller['gmv']

# 店铺类型划分
conditions = [(df_seller['gmv'] >= 100000) & (df_seller['brand_hhi'] >= 0.8) & (df_seller['revenue_b_rate'] >= 0.5),
              (df_seller['revenue_b_rate'] >= 0.4) & (df_seller['category_hhi'] >= 0.4),
              (df_seller['revenue_c_rate'] >= 0.3) & (df_seller['asin_count_revenue_fba_rate'] >= 0.4),
              (df_seller['asin_count_fbm_rate'] >= 0.4) | (
                      (df_seller['asin_count_revenue_fbm_rate'] >= 0.2) & (df_seller['category_hhi'] <= 0.4)),
              (df_seller['asin_count_revenue_fba_rate'] >= 0.5) & (df_seller['category_hhi'] <= 0.4),
              (df_seller['brand_hhi'] <= 0.2) & (df_seller['brand_count'] >= 20) & (df_seller['gmv'] <= 4000)]

df_seller['seller_tag'] = np.select(conditions, para.seller_tag, '其他')

# 格式转换
seller_list_2 = ['price_avg', 'fba_fees_avg', 'revenue_top5_avg', 'category_hhi', 'brand_hhi']
for p in seller_list_2:
    clean_util.convert_type(df_seller, p, 2)

seller_list_4 = ['asin_count_revenue_fbm_rate', 'asin_count_revenue_fba_rate', 'asin_new_rate_90', 'asin_new_rate_180',
                 'revenue_fbm_rate', 'asin_b_rate', 'revenue_b_rate', 'asin_c_rate', 'revenue_c_rate']
for q in seller_list_4:
    clean_util.convert_type(df_seller, q, 4)

# 数据整合
df_brand_report_tag = df_brand_report[
    ['task_asin', 'month_available', 'revenue_tag', 'month_available_tag', 'price_tag', 'rating_tag', 'ratings_tag',
     'revenue_tag_rank', 'month_available_tag_rank', 'price_tag_rank', 'rating_tag_rank', 'ratings_tag_rank']]

df_seller = df_seller[
    ['buybox_seller_id', 'buybox_location', 'gmv', 'asin_count', 'parent_count', 'price_avg', 'fba_fees_avg',
     'brand_count', 'asin_count_new_90', 'asin_count_new_180', 'asin_count_revenue_fbm', 'asin_count_revenue_fbm_rate',
     'asin_count_revenue_fba_rate', 'revenue_top5_avg', 'category_hhi', 'brand_hhi', 'asin_count_b', 'asin_revenue_b',
     'asin_count_c', 'asin_revenue_c', 'asin_new_rate_90', 'asin_new_rate_180', 'revenue_fbm_rate', 'asin_b_rate',
     'revenue_b_rate', 'asin_c_rate', 'revenue_c_rate', 'seller_tag']]

# ----------------------------------------------可跟卖链接推荐-----------------------------------------------------

# 重量体积字段清洗
profit_cal.weight_value_clean(df_seller_product)
profit_cal.dimensions_value_clean(df_seller_product)
profit_cal.dimensions_value_sort(df_seller_product, 'dimensions_value_cm_1', 'dimensions_value_cm_2',
                                 'dimensions_value_cm_3')
df_seller_product = df_seller_product[df_seller_product['weight(g)'].notnull()]

# 规格数据转换
df_seller_product[['max_length', 'mid_length', 'min_length', 'weight_pound', 'perimeter', 'weight_max_kg',
                   'weight_max_pound', 'product_fee']] = df_seller_product.apply(lambda row: pd.Series(
    profit_cal.value_convert_us(row['max_length_cm'], row['mid_length_cm'], row['min_length_cm'], row['weight(g)'],
                                row['price'], para.exchange_rate_us)), axis=1)

# 规格打标
df_seller_product['size_tag'] = df_seller_product.apply(lambda row: pd.Series(
    profit_cal.size_tag_us_cal(row['max_length'], row['mid_length'], row['min_length'], row['perimeter'],
                               row['weight_max_pound'])), axis=1)

# fba费用计算
df_seller_product['fba_fee'] = df_seller_product.apply(
    lambda row: profit_cal.fba_fee_us_cal(row['price'], row['size_tag'], row['weight_pound'], row['weight_max_pound']),
    axis=1)

df_seller_product['fba_fee_rate'] = df_seller_product['fba_fee'] / df_seller_product['price']

# 开售月份计算
month_available(df_seller_product)

# 可跟卖产品筛选
report_conditions = (df_seller_product['month_available'] > para.month_available_limit) & \
                    (df_seller_product['price'] > para.price_limit_lower) & \
                    (df_seller_product['price'] < para.price_limit_upper) & \
                    (df_seller_product['weight(g)'] < para.weight_limit) & \
                    (df_seller_product['ratings'] >= para.ratings_limit) & \
                    (df_seller_product['rating'] >= para.rating_limit) & \
                    (df_seller_product['fba_fee'] <= para.fba_fee_limit) & \
                    (df_seller_product['fba_fee_rate'] <= para.fba_fee_rate_limit)
df_seller_report = df_seller_product[report_conditions]

# 可跟卖推荐得分
df_seller_report['weight_score'] = common_util.get_cut(df_seller_report, 'weight(g)', para.follow_weight_list,
                                                       para.follow_weight_label)
df_seller_report['rating_score'] = common_util.get_cut(df_seller_report, 'rating', para.follow_rating_list,
                                                       para.follow_rating_label)
df_seller_report['ratings_score'] = common_util.get_cut(df_seller_report, 'ratings', para.follow_ratings_list,
                                                        para.follow_ratings_label)
df_seller_report['fba_fee_score'] = common_util.get_cut(df_seller_report, 'fba_fee', para.follow_fba_fee_list,
                                                        para.follow_fba_fee_label)
df_seller_report['fba_fee_rate_score'] = common_util.get_cut(df_seller_report, 'fba_fee_rate',
                                                             para.follow_fba_fee_rate_list,
                                                             para.follow_fba_fee_rate_label)

df_seller_report['fbm_score'] = np.where((df_seller_report['seller_type'] == "FBM"), para.follow_fbm_score, 0)

follow_tag_list = ['weight_score', 'rating_score', 'ratings_score', 'fba_fee_score', 'fba_fee_rate_score', 'fbm_score']
for follow_tag in follow_tag_list:
    # df_seller_report[follow_tag] = clean_util.convert_type(df_seller_report, follow_tag, 0)
    df_seller_report[follow_tag] = pd.to_numeric(df_seller_report[follow_tag], errors='coerce').fillna(0)

df_seller_report['follow_score'] = (
        df_seller_report['weight_score'] * para.follow_weight_score +
        df_seller_report['rating_score'] * para.follow_rating_score +
        df_seller_report['ratings_score'] * para.follow_ratings_score +
        df_seller_report['fba_fee_score'] * para.follow_fba_fee_score +
        df_seller_report['fba_fee_rate_score'] * para.follow_fba_fee_rate_score +
        df_seller_report['fbm_score'])

df_follow_report = df_seller_report[
    ['asin', 'task_asin', 'buybox_seller_id', 'max_length_cm', 'mid_length_cm', 'min_length_cm', 'weight(g)',
     'size_tag', 'fba_fee', 'fba_fee_rate', 'month_available', 'weight_score', 'rating_score', 'ratings_score',
     'fba_fee_score', 'fba_fee_rate_score', 'fbm_score', 'follow_score']]

# 数据入库
sql_engine.data_to_sql(df_brand_report_tag, path.pt_brand_report_tag, 'append', config.connet_clue_shop_db_sql)
sql_engine.data_to_sql(df_seller, path.pt_sellers_tag, 'append', config.connet_clue_shop_db_sql)

sql_engine.data_to_sql(df_follow_report, path.pt_sellers_product_follow, 'append', config.connet_clue_shop_db_sql)

# 状态更新
sql_engine.connect_product(config.sellersprite_hostname, config.sellersprite_password, config.clue_shop_database,
                           sql.update_brand_report_sql)

sql_engine.connect_product(config.sellersprite_hostname, config.sellersprite_password, config.clue_shop_database,
                           sql.update_seller_product_sql)

sql_engine.connect_product(config.sellersprite_hostname, config.sellersprite_password, config.clue_shop_database,
                           sql.update_seller_product_variation_sql)

sql_engine.connect_product(config.sellersprite_hostname, config.sellersprite_password, config.clue_shop_database,
                           sql.update_seller_product_status_sql)

sql_engine.connect_product(config.sellersprite_hostname, config.sellersprite_password, config.clue_shop_database,
                           sql.update_seller_product_sale_status_sql)

print('done')
