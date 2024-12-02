import time
import warnings
import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas.errors import SettingWithCopyWarning

from conn import sql_engine, mysql_config as config
from util import data_cleaning_util, calculation_util, duplicate_util
import pt_product_report_path as path
import pt_product_report_parameter as parameter
import pt_product_sql as sql


# 加权CPC计算
def cpc_avg(df):
    row_related = np.array(df['ASIN_KW相关度'])
    row_bid_amz = np.array(df['bid_rangeMedian'])

    bid_amz = sum(row_bid_amz * row_related * (row_bid_amz > 0)) / sum(row_related * (row_bid_amz > 0))
    bid_avg = bid_amz
    return bid_amz, bid_avg


# 排序
def sort_and_rank(df):
    df = df.groupby('asin', group_keys=False).apply(lambda x: x.sort_values(by=['ASIN_KW相关度'], ascending=[False]))
    df['row_rank'] = df.reset_index(drop=True).index
    df['rank'] = df['row_rank'].groupby(df['asin']).rank()
    return df


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 循环参数
id_start = 1
id_increment = 2000
id_end = 2000000

sellersprite_database = 'sellersprite_202409'

update_date = str(sellersprite_database)[-6:-2] + "-" + str(sellersprite_database)[-2:] + "-01"

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

while id_start < id_end:
    # 1.数据连接
    sql_asin = "select * from " + path.pt_product_get_cpc + " where status=2 and id between " + str(id_start) + " and " \
               + str(id_start + id_increment)
    sql_kw = "select " + path.pt_keywords_new + ".*,pt_product.price,pt_product.recommend from (" + sql_asin + \
             ") pt_product left join " + path.pt_keywords_new + " on pt_product.asin = " + path.pt_keywords_new + \
             ".asin where " + path.pt_keywords_new + ".id >0"
    sql_cpc = "select DISTINCT " + path.cpc_from_keywords + ".* from (" + sql_kw + ") pt_kw left join " + \
              path.cpc_from_keywords + " on pt_kw.keyword = " + path.cpc_from_keywords + \
              ".keyword where (cpc_from_keywords.bid_rangeMedian + cpc_from_keywords.bid_rangeEnd)>0"

    df_kw = sql_engine.connect_pt_product(config.sellersprite_hostname, config.sellersprite_password,
                                          sellersprite_database, sql_kw)
    df_cpc = sql_engine.connect_pt_product(config.sellersprite_hostname, config.sellersprite_password,
                                           sellersprite_database, sql_cpc)

    sql_engine.connect_product(config.oe_hostname, config.oe_password, path.product_database,
                               sql.clear_sql_product_cpc_tag_temporary)

    if df_kw.empty:
        id_start = id_start + id_increment
        continue

    # 2.数据预处理
    df_kw['clear_id'] = df_kw['asin'] + df_kw['keyword']
    df_kw = duplicate_util.df_cleaning(df_kw, 'clear_id')

    kw_list_1 = ['id', 'relate']
    for kw in kw_list_1:
        data_cleaning_util.convert_type(df_kw, kw, 0)

    kw_list_2 = ['price', 'recommend']
    for kw in kw_list_1:
        data_cleaning_util.convert_type(df_kw, kw, 2)

    data_cleaning_util.convert_date(df_kw, 'update_time')
    data_cleaning_util.convert_str(df_kw, 'keyword')

    cpc_list = ['bid_rangeMedian', 'bid_rangeEnd']
    for cpc in cpc_list:
        data_cleaning_util.convert_type(df_cpc, cpc, 2)

    # 3.数据计算

    # 合表
    df_cpc = df_cpc[['keyword', 'bid_rangeMedian', 'bid_rangeEnd']]
    df_kw_cpc = df_kw.merge(df_cpc, how='left', on='keyword')
    df_kw_cpc = df_kw_cpc.query('bid_rangeMedian > 0')

    df_kw_cpc['ASIN_KW相关度'] = df_kw_cpc['relate']

    # 排名计算
    df_kw_cpc = sort_and_rank(df_kw_cpc)

    # TOP5关键词提取
    df_kw_cpc = df_kw_cpc[df_kw_cpc['rank'] <= 5]

    # 加权CPC计算
    group_df_list = list(df_kw_cpc.groupby(df_kw_cpc['asin'], as_index=False))

    opportunity_frame = []

    for tuple_df in group_df_list:
        kw_cpc_df: DataFrame = tuple_df[1].reset_index()
        kw_cpc_df = kw_cpc_df.fillna(0)

        # 使用groupby和agg分组后执行对应列的方法
        asin_df = kw_cpc_df.groupby('asin').agg({'price': 'first',
                                                 'recommend': 'first',
                                                 'keyword': lambda x: ','.join(x.astype(str))}).reset_index()

        asin_df = asin_df.drop_duplicates(subset=['asin'], keep="first", inplace=False)

        asin_df['加权AMZ_CPC'], asin_df['加权CPC'] = cpc_avg(kw_cpc_df)

        asin_df = asin_df.fillna(0)

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

    if df_kw.empty:
        id_start = id_start + id_increment
        continue

    df_opportunity = pd.concat(opportunity_frame)

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

    df_opportunity = duplicate_util.df_cleaning(df_opportunity, 'asin')

    df_opportunity.rename(columns={'asin': 'ASIN',
                                   'price': '价格',
                                   'recommend': '推荐度',
                                   'keyword': '关键词'}, inplace=True)

    df_cpc_tag = df_cpc_tag[['asin', '加权CPC分布', '市场蓝海度分布', 'data_id']]

    df_group = df_opportunity.query('市场蓝海度>=2.5')

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
    # sellersprite_database = 'sellersprite_202408'
    sellersprite_hostname = 'rm-8vbodje181md80v052o.mysql.zhangbei.rds.aliyuncs.com'
    sellersprite_port = 3306  # ssh端口
    sellersprite_username = 'betterniche'
    sellersprite_password = "original123#"
    connet_sellersprite_db_sql = 'mysql+pymysql://' + sellersprite_username + ':' + sellersprite_password + '@' + \
                                 sellersprite_hostname + '/' + sellersprite_database + '?charset=utf8mb4'

    sql_engine.data_to_sql(df_group, path.pt_product_get_group, 'append', connet_sellersprite_db_sql)

    sql_engine.data_to_sql(df_kw_cpc, path.product_keywords_history, 'append', config.connet_product_db_sql)
    sql_engine.data_to_sql(df_opportunity, path.product_cpc_history, 'append', config.connet_product_db_sql)

    id_start = id_start + id_increment
    print("id_start：" + id_start.__str__())
    print("用时：" + (time.time() - start_time).__str__())
print("用时：" + (time.time() - start_time).__str__())
