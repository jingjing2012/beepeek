import time
import warnings
import pandas as pd
import numpy as np
from pandas.errors import SettingWithCopyWarning

from conn import sql_engine, mysql_config as config
from util import data_cleaning_util, calculation_util, calculate_util, duplicate_util
import pt_product_report_path as path
import pt_product_report_parameter as parameter
import pt_product_sql as sql

# 忽略与 Pandas SettingWithCopyWarning 模块相关的警告
warnings.filterwarnings('ignore', category=SettingWithCopyWarning)

# 忽略与 Pandas SQL 模块相关的警告
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas.io.sql")

# 忽略除以零的警告
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in log2")

# Suppress DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)

warnings.filterwarnings("ignore", message="pandas only support SQLAlchemy connectable.*")
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")

pd.set_option('future.no_silent_downcasting', True)

start_time = time.time()

# 循环参数
id_start = 0
id_increment = 10000
id_end = 2000000

update_date = str(config.sellersprite_database)[-6:-2] + "-" + str(config.sellersprite_database)[-2:] + "-01"

# cpc数据写入
# sql_engine.connect_product(config.sellersprite_hostname, config.sellersprite_password, config.sellersprite_database,
#                            sql.insert_sql_cpc_from_keywords)

while id_start < id_end:
    # 1.数据连接
    sql_asin = "select * from " + path.pt_product_get_cpc + " where status=1 and id between " + str(id_start) + " and " \
               + str(id_start + id_increment)
    sql_kw = "select " + path.pt_keywords + ".*,pt_product.price,pt_product.recommend from (" \
             + sql_asin + ") pt_product left join " + path.pt_keywords + " on pt_product.asin = " \
             + path.pt_keywords + ".asin where " + path.pt_keywords + ".id >0"
    sql_cpc = "select DISTINCT " + path.cpc_from_keywords + ".* from (" + sql_kw + ") pt_kw left join " + \
              path.cpc_from_keywords + " on pt_kw.keyword = " + path.cpc_from_keywords + ".keyword"

    df_kw = sql_engine.connect_pt_product(config.sellersprite_hostname, config.sellersprite_password,
                                          config.sellersprite_database, sql_kw)
    df_cpc = sql_engine.connect_pt_product(config.sellersprite_hostname, config.sellersprite_password,
                                           config.sellersprite_database, sql_cpc)

    sql_engine.connect_product(config.oe_hostname, config.oe_password, path.product_database,
                               sql.clear_sql_product_cpc_tag_temporary)

    if df_kw.empty:
        id_start = id_start + id_increment
        continue

    # 2.数据预处理
    df_kw['clear_id'] = df_kw['asin'] + df_kw['keyword']
    df_kw = duplicate_util.df_cleaning(df_kw, 'clear_id')

    data_cleaning_util.convert_type(df_kw, 'search_frequency', 0)
    data_cleaning_util.convert_type(df_kw, 'relevance', 1)

    kw_list_2 = ['price', 'recommend']
    for kw in kw_list_2:
        data_cleaning_util.convert_type(df_kw, kw, 2)

    data_cleaning_util.convert_date(df_kw, 'update_time')

    kw_list = ['keyword']
    for kw_str in kw_list:
        data_cleaning_util.convert_str(df_kw, kw_str)

    cpc_list = ['bid_rangeMedian', 'bid_rangeEnd']
    for cpc in cpc_list:
        data_cleaning_util.convert_type(df_cpc, cpc, 2)

    # 3.数据计算
    df_kw['search_frequency'] = np.fmax(df_kw['search_frequency'], 0)

    # 合表
    df_cpc = df_cpc[['keyword', 'bid_rangeMedian', 'bid_rangeEnd']]
    df_kw_cpc = df_kw.merge(df_cpc, how='left', on='keyword')
    df_kw_cpc = df_kw_cpc.query('bid_rangeMedian > 0')

    df_kw_cpc['搜索排名权重'] = np.fmax(3, np.fmax(np.log10(df_kw_cpc['search_frequency'] + 1), 1))
    df_kw_cpc['ASIN_KW相关度'] = round(df_kw_cpc['relevance'] / df_kw_cpc['搜索排名权重'], 2)

    # 排名计算
    df_kw_cpc = calculate_util.sort_and_rank(df_kw_cpc)

    # TOP5关键词提取
    df_kw_cpc = df_kw_cpc[df_kw_cpc['rank'] <= 5]

    # 加权CPC计算
    asin_df = df_kw_cpc.groupby('asin').agg({'price': 'first',
                                             'recommend': 'first',
                                             'keyword': lambda x: ','.join(x.astype(str))}).reset_index()

    asin_df = asin_df.drop_duplicates(subset=['asin'], keep="first", inplace=False)

    kw_cpc_df = df_kw_cpc.groupby('asin').apply(lambda x: calculate_util.cpc_avg(x))
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

    # 数据更新日期
    df_kw_cpc['数据更新时间'] = update_date
    data_cleaning_util.convert_date(df_kw_cpc, '数据更新时间')

    df_opportunity['数据更新时间'] = update_date
    data_cleaning_util.convert_date(df_opportunity, '数据更新时间')

    # tag表
    df_cpc_tag = df_opportunity[['asin', '加权CPC', '市场蓝海度', '数据更新时间']]

    mr_list = ['加权CPC', '市场蓝海度']
    for mr in mr_list:
        calculation_util.get_mround(df_cpc_tag, mr, mr + '分布', 0.05)

    df_cpc_tag['data_id'] = df_cpc_tag['asin'] + " | " + update_date

    # 字段整合
    df_kw_cpc = df_kw_cpc[
        ['asin', 'keyword', 'bid_rangeMedian', 'bid_rangeEnd', 'ASIN_KW相关度', 'rank', '数据更新时间']]

    df_kw_cpc['clear_id'] = df_kw_cpc['asin'] + df_kw_cpc['keyword']
    df_kw_cpc = duplicate_util.df_cleaning(df_kw_cpc, 'clear_id')
    df_kw_cpc = df_kw_cpc.drop(['clear_id'], axis=1)

    df_kw_cpc.rename(columns={'asin': 'ASIN',
                              'keyword': '关键词',
                              'bid_rangeMedian': 'AMZ_BID推荐',
                              'bid_rangeEnd': 'AMZ_BID上限',
                              'rank': '相关度排名'}, inplace=True)

    df_opportunity = df_opportunity[
        ['asin', 'price', 'recommend', 'keyword', '加权CPC', '预期CR', '转化净值', '预期CPC', 'CPC因子', '市场蓝海度', '数据更新时间']]
    df_opportunity = duplicate_util.df_cleaning(df_opportunity, 'asin')

    df_opportunity.rename(columns={'asin': 'ASIN',
                                   'price': '价格',
                                   'recommend': '推荐度',
                                   'keyword': '关键词'}, inplace=True)

    df_cpc_tag = df_cpc_tag[['asin', '加权CPC分布', '市场蓝海度分布', 'data_id']]

    df_group = df_opportunity.query('市场蓝海度>=2')

    df_group = df_group[['ASIN', '价格', '推荐度', '市场蓝海度']]

    df_group = duplicate_util.df_cleaning(df_group, 'ASIN')

    df_group.rename(columns={'ASIN': 'asin',
                             '价格': 'price',
                             '推荐度': 'recommend',
                             '市场蓝海度': 'blue_ocean_estimate'}, inplace=True)

    df_keyword_temporary = df_opportunity[['ASIN', '关键词']]
    df_keyword_temporary = duplicate_util.df_cleaning(df_keyword_temporary, 'ASIN')

    # 存入数据库
    sql_engine.data_to_sql(df_cpc_tag, path.product_cpc_tag_temporary, 'append', config.connet_product_db_sql)
    sql_engine.connect_product(config.oe_hostname, config.oe_password, path.product_database,
                               sql.update_sql_product_tag)

    # 精铺算法数据源库
    sql_engine.data_to_sql(df_group, path.pt_product_get_group, 'append', config.connet_sellersprite_db_sql)

    sql_engine.data_to_sql(df_kw_cpc, path.product_keywords_history, 'append', config.connet_product_db_sql)
    sql_engine.data_to_sql(df_opportunity, path.product_cpc_history, 'append', config.connet_product_db_sql)

    id_start = id_start + id_increment
    print("id_start：" + id_start.__str__())
    print("用时：" + (time.time() - start_time).__str__())

# 类目去重前置
sql_engine.connect_product(config.sellersprite_hostname, config.sellersprite_password, config.sellersprite_database,
                           sql.update_sql_sub_category)

"""
# 关联流量历史数据复用
sql_engine.connect_product(config.sellersprite_hostname, config.sellersprite_password, config.sellersprite_database,
                           sql.insert_sql_pt_relevance_asins_old)
sql_engine.connect_product(config.sellersprite_hostname, config.sellersprite_password, config.sellersprite_database,
                           sql.insert_sql_pt_relation_traffic_old)
"""
print("用时：" + (time.time() - start_time).__str__())
