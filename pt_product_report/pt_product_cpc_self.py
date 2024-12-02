import re
import sys
import time
import warnings
import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas.errors import SettingWithCopyWarning

from conn import sql_engine, mysql_config as config
from util import data_cleaning_util, calculation_util
import pt_product_report_path as path
import pt_product_report_parameter as parameter
import pt_product_sql as sql


# 检查是否包含中文字符
def contains_chinese(text):
    pattern = re.compile('[\u4e00-\u9fa5]+')
    return bool(pattern.search(text))


# series转DataFrame
def convert_col(series, col):
    df = pd.DataFrame(series)
    df.columns = [col]
    return df


# 加权CPC计算
def cpc_avg(df):
    row_searches = np.array(df['searches'])
    row_related = np.array(df['ASIN_KW相关度'])
    row_bid_sp = np.array(df['bid'])
    row_bid_amz = np.array(df['bid_rangeMedian'])

    bid_sp = sum(row_searches * row_bid_sp * row_related * (row_bid_sp > 0)) / sum(
        row_searches * row_related * (row_bid_sp > 0))
    bid_amz = sum(row_searches * row_bid_amz * row_related * (row_bid_amz > 0)) / sum(
        row_searches * row_related * (row_bid_amz > 0))

    bid_sp_avg = sum((row_related > 0) * (row_bid_sp > 0)) / 5 * 0.2
    bid_amz_avg = sum((row_related > 0) * (row_bid_amz > 0)) / 5 * 0.8

    bid_avg = (bid_sp * bid_sp_avg + bid_amz * bid_amz_avg) / (bid_sp_avg + bid_amz_avg)
    return bid_sp, bid_amz, bid_avg


# 相关度计算
def sp_amz_related(df):
    df['ASIN_KW相关度_1'] = np.where(df['rank_position_page'] > 0,
                                  np.where(df['rank_position_page'] <= 2, 3 - df['rank_position_page'],
                                           1 / df['rank_position_page']), 0)
    df['ASIN_KW相关度_2'] = np.where(df['ad_position_page'] > 0,
                                  np.where(df['ad_position_page'] <= 2, 3 - df['ad_position_page'],
                                           1 / df['ad_position_page']), 0)

    condition_asin_kw_1 = (df['rank_position_page'] > 0) & (df['rank_position_page'] <= 3)
    condition_asin_kw_2 = (df['ad_position_page'] > 0) & (df['ad_position_page'] <= 3)

    df['ASIN_KW相关度_3'] = np.where(condition_asin_kw_1 | condition_asin_kw_2, np.fmin(
        np.where(df['supply_demand_ratio'] > 0, round(df['supply_demand_ratio'] / df['供需比均值'], 1), 1), 3), 0)
    df['ASIN_KW相关度_4'] = np.where(condition_asin_kw_1 | condition_asin_kw_2, np.fmin(
        np.where(df['ad_products'] > 0, round(df['月购买转化率'] / df['ad_products'], 1), 1), 3), 0)

    df['ASIN_KW相关度'] = np.where(df['月购买转化率'] * 1 > 0,
                                df['ASIN_KW相关度_1'] + df['ASIN_KW相关度_2'] * 2 + df['ASIN_KW相关度_3'] + df['ASIN_KW相关度_4'],
                                1)

    calculation_util.get_mround(df, 'ASIN_KW相关度', 'ASIN_KW相关度', 0.5)
    df['ASIN_KW相关度'] = np.where(df['ASIN_KW相关度'] > 0, round(df['ASIN_KW相关度'], 1), 1)

    df['contains_chinese'] = df['keyword'].apply(contains_chinese)
    df['异常'] = np.where(df['contains_chinese'] == False, 0, 1)

    df['ASIN_KW相关度'] = np.where(df['异常'] != 0, 0, df['ASIN_KW相关度'])
    return df


# 排序
def sort_and_rank(df):
    df = df.groupby('asin', group_keys=False).apply(
        lambda x: x.sort_values(by=['ASIN_KW相关度', 'searches'], ascending=[False, False]))
    df['row_rank'] = df.reset_index(drop=True).index
    df['rank'] = df['row_rank'].groupby(df['asin']).rank()
    return df


# df去重
def df_cleaning(df, clear_id):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.drop_duplicates(subset=[clear_id])
    df = df.dropna(subset=[clear_id])
    return df


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
df_kw = df_kw.query('id>0')
df_cpc = df_cpc.query('id>0')

kw_list_1 = ['id', 'searches', 'purchases', 'ad_products', 'rank_position_page', 'ad_position_page']
for kw in kw_list_1:
    data_cleaning_util.convert_type(df_kw, kw, 0)

kw_list_2 = ['bid', 'supply_demand_ratio', 'price', 'recommend']
for kw in kw_list_1:
    data_cleaning_util.convert_type(df_kw, kw, 2)

data_cleaning_util.convert_date(df_kw, 'update_time')
data_cleaning_util.convert_date(df_kw, '数据更新时间')
data_cleaning_util.convert_str(df_kw, 'keyword')

cpc_list = ['bid_rangeMedian', 'bid_rangeEnd']
for cpc in cpc_list:
    data_cleaning_util.convert_type(df_cpc, cpc, 2)

# 3.数据计算
df_kw['月购买转化率'] = np.where(df_kw['searches'] > 0, round(df_kw['purchases'] / df_kw['searches'], 3), np.nan)

df_kw_supply_demand_ratio_mean = df_kw.groupby('asin')['supply_demand_ratio'].mean()
df_kw_supply_demand_ratio_mean = convert_col(df_kw_supply_demand_ratio_mean, '供需比均值')

df_kw_ad_products_mean = df_kw.groupby('asin')['ad_products'].mean()
df_kw_ad_products_mean = convert_col(df_kw_ad_products_mean, '广告竞品数均值')

df_kw = df_kw.merge(df_kw_supply_demand_ratio_mean, how='left', on='asin')
df_kw = df_kw.merge(df_kw_ad_products_mean, how='left', on='asin')

# ASIN_KW相关度计算
sp_amz_related(df_kw)

# 排名计算
df_kw = sort_and_rank(df_kw)

# TOP5关键词提取
df_kw_cpc = df_kw.query('rank<=5')

# 合表
df_cpc = df_cpc[['keyword', 'bid_rangeMedian', 'bid_rangeEnd']]
df_kw_cpc = df_kw_cpc.merge(df_cpc, how='left', on='keyword')

# 加权CPC计算
group_df_list = list(df_kw_cpc.groupby(df_kw_cpc['asin'], as_index=False))

opportunity_frame = []

for tuple_df in group_df_list:
    kw_cpc_df: DataFrame = tuple_df[1].reset_index()
    kw_cpc_df.fillna(0, inplace=True)

    # 使用groupby和agg分组后执行对应列的方法
    asin_df = kw_cpc_df.groupby('asin').agg({'price': 'first',
                                             'recommend': 'first',
                                             '数据更新时间': 'first',
                                             'keyword': lambda x: ','.join(x.astype(str))}).reset_index()

    asin_df = asin_df.drop_duplicates(subset=['asin'], keep="first", inplace=False)

    asin_df['加权SP_CPC'], asin_df['加权AMZ_CPC'], asin_df['加权CPC'] = cpc_avg(kw_cpc_df)
    asin_df['SP_AMZ差异度'] = np.where(asin_df['加权SP_CPC'] * asin_df['加权AMZ_CPC'] > 0,
                                    round(asin_df['加权SP_CPC'] / asin_df['加权AMZ_CPC'] - 1, 4), np.nan)

    asin_df.fillna(0, inplace=True)

    # 蓝海度计算
    asin_df['预期CR'] = parameter.p_cr_a * (asin_df['price'].pow(parameter.p_cr_b))

    data_cleaning_util.convert_type(asin_df, '预期CR', 4)

    asin_df['转化净值'] = asin_df['price'] * asin_df['预期CR']

    asin_df['预期CPC'] = asin_df['转化净值'] * parameter.product_acos

    asin_df['CPC因子'] = np.where(asin_df['预期CPC'] * asin_df['加权CPC'] > 0, asin_df['预期CPC'] / asin_df['加权CPC'],
                                np.nan)

    asin_df['市场蓝海度'] = np.where(asin_df['CPC因子'] > 0, parameter.MS_a + parameter.MS_b / (
            1 + (parameter.MS_e ** (- (asin_df['CPC因子'] - parameter.MS_cs * parameter.MS_c)))), np.nan)

    opportunity_frame.append(asin_df)

df_opportunity = pd.concat(opportunity_frame)

data_cleaning_util.convert_type(df_kw_cpc, '供需比均值', 1)
data_cleaning_util.convert_type(df_kw_cpc, '广告竞品数均值', 1)

# tag表
df_cpc_tag = df_opportunity[['asin', 'SP_AMZ差异度', '加权CPC', '市场蓝海度', '数据更新时间']]

mr_list = ['SP_AMZ差异度', '加权CPC', '市场蓝海度']
for mr in mr_list:
    calculation_util.get_mround(df_cpc_tag, mr, mr + '分布', 0.05)

df_cpc_tag['data_id'] = df_cpc_tag['asin'] + " | " + pd.to_datetime(df_cpc_tag['数据更新时间']).dt.strftime(
    '%Y-%m-%d')

# 字段整合
df_kw_cpc = df_kw_cpc[
    ['asin', 'keyword', 'searches', 'bid', 'purchases', 'products', 'supply_demand_ratio', 'ad_products',
     'rank_position_position', 'rank_position_page', 'ad_position_position', 'ad_position_page', 'bid_rangeMedian',
     'bid_rangeEnd', 'ASIN_KW相关度', '供需比均值', '广告竞品数均值', '月购买转化率', 'rank', '数据更新时间']]

df_kw_cpc['clear_id'] = df_kw_cpc['asin'] + df_kw_cpc['keyword']
df_kw_cpc = df_cleaning(df_kw_cpc, 'clear_id')
df_kw_cpc = df_kw_cpc.drop(['clear_id'], axis=1)

df_kw_cpc.rename(columns={'asin': 'ASIN',
                          'keyword': '关键词',
                          'searches': '月搜索量',
                          'bid': 'BID竞价',
                          'purchases': '月购买量',
                          'products': '商品数',
                          'supply_demand_ratio': '供需比',
                          'ad_products': '广告竞品数',
                          'rank_position_position': 'ASIN自然排名',
                          'rank_position_page': 'ASIN自然排名页',
                          'ad_position_position': 'ASIN广告排名',
                          'ad_position_page': 'ASIN广告排名页',
                          'bid_rangeMedian': 'AMZ_BID推荐',
                          'bid_rangeEnd': 'AMZ_BID上限',
                          'rank': '相关度排名'}, inplace=True)

df_opportunity = df_cleaning(df_opportunity, 'asin')

df_opportunity.rename(columns={'asin': 'ASIN',
                               'price': '价格',
                               'recommend': '推荐度',
                               'keyword': '关键词'}, inplace=True)

df_cpc_tag = df_cpc_tag[['asin', 'SP_AMZ差异度分布', '加权CPC分布', '市场蓝海度分布', 'data_id']]

df_group = df_opportunity

df_group = df_group[['ASIN', '价格', '推荐度', '市场蓝海度', '数据更新时间']]

df_group = df_cleaning(df_group, 'ASIN')

df_group.rename(columns={'ASIN': 'asin',
                         '价格': 'price',
                         '推荐度': 'recommend',
                         '市场蓝海度': 'blue_ocean_estimate',
                         '数据更新时间': 'update_time'}, inplace=True)

df_keyword_temporary = df_opportunity[['ASIN', '关键词']]
df_keyword_temporary = df_cleaning(df_keyword_temporary, 'ASIN')

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
