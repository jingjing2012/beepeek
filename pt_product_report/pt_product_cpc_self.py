import sys
import time
import warnings

import numpy as np
import pandas as pd
from pandas.errors import SettingWithCopyWarning

import calculate_util
import calculation_util
import data_cleaning_util
import duplicate_util
import pt_product_report_parameter as parameter
import pt_product_report_path as path
import pt_product_sql as sql
from conn import sql_engine, mysql_config as config

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 忽略与 Pandas SettingWithCopyWarning 模块相关的警告
warnings.filterwarnings('ignore', category=SettingWithCopyWarning)

# 忽略与 Pandas SQL 模块相关的警告
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas.io.sql")

# 忽略除以零的警告
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in log2")

warnings.filterwarnings("ignore", message="pandas only support SQLAlchemy connectable.*")
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)

start_time = time.time()

# cpc数据处理
sql_engine.connect_product(config.sellersprite_hostname, config.sellersprite_password, path.clue_self,
                           sql.clean_sql_cpc_from_keywords)
sql_engine.connect_product(config.sellersprite_hostname, config.sellersprite_password, path.clue_self,
                           sql.insert_sql_cpc_from_keywords)

# 1.数据连接
df_kw = sql_engine.connect_pt_product(config.sellersprite_hostname, config.sellersprite_password,
                                      config.clue_self_database, sql.sql_kw)
df_cpc = sql_engine.connect_pt_product(config.sellersprite_hostname, config.sellersprite_password,
                                       config.clue_self_database, sql.sql_cpc)

if df_cpc.empty:
    print('df_kw.empty')
    sys.exit()

sql_engine.connect_product(config.oe_hostname, config.oe_password, path.product_database,
                           sql.clear_sql_product_cpc_tag_temporary)

# 2.数据预处理
df_kw['clear_id'] = df_kw['asin'] + df_kw['keyword']
df_kw = duplicate_util.df_cleaning(df_kw, 'clear_id')

data_cleaning_util.convert_type(df_kw, 'search_frequency', 0)
data_cleaning_util.convert_type(df_kw, 'relevance', 1)

kw_list_2 = ['price', 'recommend']
for kw in kw_list_2:
    data_cleaning_util.convert_type(df_kw, kw, 2)

data_cleaning_util.convert_date(df_kw, 'update_time')
data_cleaning_util.convert_date(df_kw, '数据更新时间')
data_cleaning_util.convert_str(df_kw, 'keyword')

cpc_list = ['bid_rangeMedian', 'bid_rangeEnd']
for cpc in cpc_list:
    data_cleaning_util.convert_type(df_cpc, cpc, 2)

# 3.数据计算
df_kw['search_frequency'] = np.fmax(df_kw['search_frequency'], 0)

# 合表
df_cpc = df_cpc[['keyword', 'bid_rangeMedian', 'bid_rangeEnd']]
df_kw_cpc = df_kw.merge(df_cpc, how='left', on='keyword')
df_kw_cpc = df_kw_cpc.query('bid_rangeMedian > 0')

if df_kw_cpc.empty:
    print('df_kw_cpc.empty')
    sys.exit()

df_kw_cpc['搜索排名权重'] = np.fmax(3, np.fmax(np.log10(df_kw_cpc['search_frequency'] + 1), 1))
df_kw_cpc['ASIN_KW相关度'] = round(df_kw_cpc['relevance'] / df_kw_cpc['搜索排名权重'], 2)

# 排名计算
df_kw_cpc = calculate_util.sort_and_rank(df_kw_cpc)

# TOP5关键词提取
df_kw_cpc = df_kw_cpc[df_kw_cpc['rank'] <= 5]

# 加权CPC计算
asin_df = df_kw_cpc.groupby('asin').agg({'country': 'first',
                                         'price': 'first',
                                         'recommend': 'first',
                                         '数据更新时间': 'first',
                                         'keyword': lambda x: ','.join(x.astype(str))}).reset_index()

asin_df = asin_df.drop_duplicates(subset=['asin'], keep="first", inplace=False)

kw_cpc_df = df_kw_cpc.groupby('asin', include_groups=False).apply(lambda x: calculate_util.cpc_avg(x))
kw_cpc_df = data_cleaning_util.convert_col(kw_cpc_df, '加权CPC')

asin_df = asin_df.merge(kw_cpc_df, how='left', on='asin')

# 蓝海度计算
asin_df['预期CR'] = parameter.p_cr_a * (asin_df['price'].pow(parameter.p_cr_b))

data_cleaning_util.convert_type(asin_df, '预期CR', 4)

asin_df['转化净值'] = asin_df['price'] * asin_df['预期CR']

asin_df['预期CPC'] = asin_df['转化净值'] * parameter.product_acos

asin_df['CPC因子'] = np.where(asin_df['预期CPC'] * asin_df['加权CPC'] > 0, asin_df['预期CPC'] / asin_df['加权CPC'],
                            np.nan)

asin_df['市场蓝海度'] = np.where(asin_df['CPC因子'] > 0, parameter.MS_a + parameter.MS_b / (
        1 + (parameter.MS_e ** (- (asin_df['CPC因子'] - parameter.MS_cs * parameter.MS_c)))), np.nan)

df_opportunity = asin_df

# tag表
df_cpc_tag = df_opportunity[['asin', 'country', '加权CPC', '市场蓝海度', '数据更新时间']]

mr_list = ['加权CPC', '市场蓝海度']
for mr in mr_list:
    calculation_util.get_mround(df_cpc_tag, mr, mr + '分布', 0.05)

df_cpc_tag['data_id'] = df_cpc_tag['asin'] + " | " + df_cpc_tag['country'] + " | " + pd.to_datetime(
    df_cpc_tag['数据更新时间']).dt.strftime('%Y-%m-%d')

# 字段整合
df_kw_cpc = df_kw_cpc[
    ['asin', 'country', 'keyword', 'bid_rangeMedian', 'bid_rangeEnd', 'ASIN_KW相关度', 'rank', '数据更新时间']]

df_kw_cpc['clear_id'] = df_kw_cpc['asin'] + df_kw_cpc['keyword']
df_kw_cpc = duplicate_util.df_cleaning(df_kw_cpc, 'clear_id')
df_kw_cpc = df_kw_cpc.drop(['clear_id'], axis=1)

df_kw_cpc.rename(columns={'asin': 'ASIN',
                          'country': 'site',
                          'keyword': '关键词',
                          'bid_rangeMedian': 'AMZ_BID推荐',
                          'bid_rangeEnd': 'AMZ_BID上限',
                          'rank': '相关度排名'}, inplace=True)

df_opportunity = df_opportunity[
    ['asin', 'country', 'price', 'recommend', 'keyword', '加权CPC', '预期CR', '转化净值', '预期CPC', 'CPC因子', '市场蓝海度', '数据更新时间']]
df_opportunity = duplicate_util.df_cleaning(df_opportunity, 'asin')

df_opportunity.rename(columns={'asin': 'ASIN',
                               'country': 'site',
                               'price': '价格',
                               'recommend': '推荐度',
                               'keyword': '关键词'}, inplace=True)

df_cpc_tag = df_cpc_tag[['asin', '加权CPC分布', '市场蓝海度分布', 'data_id']]

df_group = df_opportunity

df_group = df_group[['ASIN', 'site', '价格', '推荐度', '市场蓝海度', '数据更新时间']]

df_group = duplicate_util.df_cleaning(df_group, 'ASIN')

df_group.rename(columns={'ASIN': 'asin',
                         'site': 'country',
                         '价格': 'price',
                         '推荐度': 'recommend',
                         '市场蓝海度': 'blue_ocean_estimate',
                         '数据更新时间': 'update_time'}, inplace=True)

df_keyword_temporary = df_opportunity[['ASIN', '关键词']]
df_keyword_temporary = duplicate_util.df_cleaning(df_keyword_temporary, 'ASIN')

# 存入数据库
sql_engine.data_to_sql(df_cpc_tag, path.product_cpc_tag_temporary, 'append', config.connet_product_db_sql)
sql_engine.connect_product(config.oe_hostname, config.oe_password, path.product_database,
                           sql.update_sql_product_tag_self)

sql_engine.data_to_sql(df_group, path.pt_product_get_group, 'append', config.connet_clue_self_db_sql)
sql_engine.connect_product(config.sellersprite_hostname, config.sellersprite_password, path.clue_self,
                           sql.update_clue_cpc_sql)

sql_engine.data_to_sql(df_kw_cpc, path.product_keywords_self, 'append', config.connet_product_db_sql)
sql_engine.data_to_sql(df_opportunity, path.product_cpc_self, 'append', config.connet_product_db_sql)

print("用时：" + (time.time() - start_time).__str__())
