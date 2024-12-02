import time
import warnings
import numpy as np
import pandas as pd
import pymysql
from pandas.errors import SettingWithCopyWarning

import mysql_config as config
import niche_data_parameter as para
import niche_data_path as path
import sql_engine
import niche_sql


# 0.连接数据并清空表格
def connect_mysql_betterin(sql):
    try:
        conn_betterin = pymysql.connect(host=config.betterin_hostname, user=config.betterin_username,
                                        passwd=config.betterin_password, database=config.betterin_database,
                                        charset='utf8')
        with conn_betterin.cursor() as cur:
            cur.execute(sql)
            conn_betterin.commit()
            cur.close()
            conn_betterin.close()
    except pymysql.MySQLError as e:
        print(f"Error connecting to the database: {e}")


# 1.连接数据库并读取数据
def connect_niche_original(sql):
    conn_oe_str = sql_engine.conn_url(config.oe_hostname, config.oe_username, config.oe_password, config.oe_database)
    with sql_engine.create_conn(conn_oe_str) as conn_oe:
        df = pd.read_sql(sql, conn_oe)
    return df


def connect_niche(sql):
    conn_niche_str = sql_engine.conn_url(config.betterin_hostname, config.betterin_username, config.betterin_password,
                                         config.betterin_database)
    with sql_engine.create_conn(conn_niche_str) as conn_niche:
        df = pd.read_sql(sql, conn_niche)
    return df


# 2.数据处理
# 站点修正
def niche_mkid_complete(df):
    df['mkid'] = df['mkid'].replace({'ATVPDKIKX0DER': "US", 'A1F83G8C2ARO7P': "UK", 'A1PA6795UKMFR9': "DE"})
    df = df.iloc[:, ~df.columns.duplicated(keep='first')]
    df['利基站点'] = df['niche_title'] + " | " + df['mkid']
    return df


# 按列去重
def col_duplicate(df):
    duplicate_cols = [col for col in df.columns if col.endswith('_y')]
    df.drop(columns=duplicate_cols, inplace=True)
    df = df.iloc[:, ~df.columns.duplicated(keep='last')]
    return df


# 数据类型修正
def convert_type(df, type_list, d):
    for i in type_list:
        df[i] = df[i].replace('', np.nan)
        df[i] = df[i].astype('float64').round(decimals=d)
    return type_list


# 数值修正
def convert_numeric(df, numeric_list):
    for s in numeric_list:
        df[s] = np.where(df[s] * 1 > 0, df[s], np.nan)
        df[s] = np.where(df[s] > 10000, 10000, df[s])
    return numeric_list


# df去重
def df_clear(df, clear_id):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.drop_duplicates(subset=[clear_id])
    df = df.dropna(subset=[clear_id])
    return df


# 字符串格式转换
def convert_str(df, col):
    df[col] = df[col].astype(str)
    df[col] = df[col].str.strip()
    return df[col]


# series转dataframe
def df_convert_column(series, col, index_col, dec_n):
    if series.empty:
        df = pd.DataFrame()
        df[index_col] = np.nan
        df[col] = np.nan
        return df
    else:
        df = pd.DataFrame(series)
    df.columns = [col]
    df[col] = df[col].round(dec_n)
    return df


# 加权平均计算
def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return d.mean()


# 数据打标签
def df_cut(df, col, col_cut, bins_cut, labels_cut):
    df[col_cut] = pd.cut(df[col], bins_cut, right=False, labels=labels_cut, include_lowest=True)
    return df


# mround函数实现
def df_mround(df, df_str, df_str_mround, mround_n):
    df[df_str_mround] = round(df[df_str] / mround_n) * mround_n
    return df[df_str_mround]


# 类目补全
def niche_category_complete(df, df_category):
    df = df[df['category'].notnull()]
    df = df[['asin', 'category']]
    df = df.drop_duplicates()
    df = df[(df['category'].str.startswith(':') == False)]
    df2 = df[(df['category'].str.contains(':') == True)]
    # 分割 'category' 列
    split_data = df2['category'].str.split(':', n=2, expand=True)

    # 确保结果中有三列
    for i in range(3):
        if i not in split_data.columns:
            split_data[i] = None

    # 将分割数据赋给新列
    df2[['二级类目名称', '三级类目名称', '四级类目名称']] = split_data

    df2['类目id'] = df2['二级类目名称'] + ':' + df2['三级类目名称']
    df2 = df2[['asin', 'category', '类目id']]
    df2 = df2.merge(df_category, on='类目id', how="left")

    df1 = df[(df['category'].str.contains(':') == False)]

    if df1.empty:
        df = df2
    else:
        df1['二级类目名称'] = df1['category']
        df1 = df1.merge(df_category, on='二级类目名称', how="left")
        df = pd.concat([df1, df2], ignore_index=True)
    df = df.drop_duplicates()
    df['category_complete'] = np.where(df['一级类目名称'].isna(), df['category'],
                                       df['一级类目名称'].fillna("") + ":" + df['category'].fillna(""))
    df['一级类目名称'] = np.where(df['一级类目名称'].isna(),
                            df['category'].str.split(':').apply(lambda x: ':'.join(x[:-1]), include_groups=False),
                            df['一级类目名称'])
    df = df[['asin', 'category_complete', '一级类目名称']]
    return df


def niche_top_asin(df, n):
    df = df.query('asin_click_sort==@n')
    df = df[['niche_id', 'asin', 'brand', 'category']]
    df_category = niche_category_complete(df, df_oe_category)
    df = pd.merge(df, df_category, on='asin', how="left")
    df['category_complete'] = df['category_complete'].fillna(df['category'])
    df.rename(columns={'asin': "ASIN" + str(n), 'brand': "ASIN" + str(n) + "品牌",
                       'category_complete': "ASIN" + str(n) + "完整类名", '一级类目名称': "ASIN" + str(n) + "一级类目"}, inplace=True)
    return df


# 3.导入目标库
def oe_save_to_sql(df, table, args, conn_str):
    with sql_engine.create_conn(conn_str) as data_conn:
        df.to_sql(table, con=data_conn, if_exists=args, index=False, chunksize=1000)


# ---------------------------------------------------------------------------------------------------------

start_time = time.time()
# 当清洗US表时清空keywords、asin、trends表，方便替换为最新周数据
niche_table_list = [path.Keyword_sql_name, path.ASINs_sql_name, path.Trends_sql_name, path.niche_price]

for niche_table in niche_table_list:
    if config.oe_database.startswith("oe_us"):
        clear_sql = "TRUNCATE TABLE " + niche_table
        connect_mysql_betterin(clear_sql)

# 循环参数
row_start = 1
row_increment = 1000
row_max = 100000

# 忽略与 Pandas SettingWithCopyWarning 模块相关的警告
warnings.filterwarnings('ignore', category=SettingWithCopyWarning)
warnings.filterwarnings("ignore", message="pandas only support SQLAlchemy connectable.*")
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")

pd.set_option('future.no_silent_downcasting', True)

# 循环写入表格数据
while row_start < row_max:
    # 1.读取原始数据
    niche_key_sql = "select * from " + config.oe_database + "." + path.oe_niche_name + " where pt_niche.id between " + \
                    str(row_start) + " and " + str(row_start + row_increment)

    niche_trends_sql = 'select *,' + config.oe_database + "." + path.oe_trends_name + \
                       '.product_count as "product_count_7" from (' + niche_key_sql + ') pt_niche_title left join ' + \
                       path.oe_trends_name + ' on pt_niche_title.niche_id = ' + path.oe_trends_name + '.niche_id'

    niche_asin_sql_us = "select * from (" + niche_key_sql + ") pt_niche_title left join " + \
                        config.oe_database + "." + path.oe_asin_name + " on pt_niche_title.niche_id =" + \
                        config.oe_database + "." + path.oe_asin_name + ".niche_id left join " + \
                        config.oe_database_asin + "." + path.oe_commodity_name + " on " + \
                        config.oe_database + "." + path.oe_asin_name + ".asin =" + \
                        config.oe_database_asin + "." + path.oe_commodity_name + ".asin"

    niche_asin_sql = "select * from (" + niche_key_sql + ") pt_niche_title left join " + \
                     config.oe_database + "." + path.oe_asin_name + " on pt_niche_title.niche_id =" + \
                     config.oe_database + "." + path.oe_asin_name + ".niche_id left join " + \
                     config.oe_database + "." + path.oe_commodity_name + " on " + \
                     config.oe_database + "." + path.oe_asin_name + ".asin =" + \
                     config.oe_database + "." + path.oe_commodity_name + ".asin"

    niche_keywords_sql = "select pt_niche_title.niche_title,pt_niche_title.mkid," \
                         + config.oe_database + "." + path.oe_keywords_name + ".keyword," \
                         + config.oe_database + "." + path.oe_keywords_name + ".niche_id," \
                         + config.oe_database + "." + path.oe_keywords_name + ".search_term_id," \
                         + config.oe_database + "." + path.oe_keywords_name + ".search_conversion_rate," \
                         + config.oe_database + "." + path.oe_keywords_name + ".search_volume_t_90," \
                         + config.oe_database + "." + path.oe_keywords_name + ".search_volume_qoq," \
                         + config.oe_database + "." + path.oe_keywords_name + ".search_volume_yoy," \
                         + config.oe_database + "." + path.oe_keywords_name + ".search_volume_t_360," \
                         + config.oe_database + "." + path.oe_keywords_name + ".search_volume_growth_t_360_yoy," \
                         + config.oe_database + "." + path.oe_keywords_name + ".search_conversion_rate_t_360," \
                         + config.oe_database + "." + path.oe_keywords_name + ".click_share," \
                         + config.oe_database + "." + path.oe_keywords_name + ".click_share_t_360 from (" + \
                         niche_key_sql + ") pt_niche_title left join " + \
                         config.oe_database + "." + path.oe_keywords_name + " on " + \
                         config.oe_database + ".pt_niche_title.niche_id =" + \
                         config.oe_database + "." + path.oe_keywords_name + ".niche_id"
    niche_keywords_top5_sql = "SELECT * FROM(SELECT *,ROW_NUMBER() OVER (PARTITION by keywords.niche_id " + \
                              "ORDER BY keywords.search_volume_t_360 DESC) AS 'rank' from(" + niche_keywords_sql + \
                              " WHERE " + config.oe_database + "." + path.oe_keywords_name + ".id is NOT NULL) " + \
                              "keywords) keywords_top WHERE keywords_top.rank<=5"
    niche_keywords_cpc_us = "SELECT parent_id,keyword AS 'keyword_amz',bid_rangeMedian,bid_rangeEnd FROM " + \
                            config.oe_database_cpc + "." + path.oe_cpc_name + " WHERE crawler_status=1"
    niche_keywords_cpc = "SELECT parent_id,keyword AS 'keyword_amz',bid_rangeMedian,bid_rangeEnd FROM " + \
                         config.oe_database + "." + path.oe_cpc_name + " WHERE crawler_status=1"
    niche_cpc_us = "SELECT niche_title,mkid,keyword,niche_id,search_volume_t_360,bid_rangeMedian,bid_rangeEnd FROM(" + \
                   niche_keywords_top5_sql + ") keywords_top5 LEFT JOIN (" + niche_keywords_cpc_us + ") keywords_cpc" + \
                   " ON keywords_top5.keyword = keywords_cpc.keyword_amz WHERE keywords_cpc.parent_id IS NOT NULL"
    niche_cpc = "SELECT niche_title,mkid,keyword,niche_id,search_volume_t_360,bid_rangeMedian,bid_rangeEnd FROM(" + \
                niche_keywords_top5_sql + ") keywords_top5 LEFT JOIN (" + niche_keywords_cpc + ") keywords_cpc" + \
                " ON keywords_top5.keyword = keywords_cpc.keyword_amz WHERE keywords_cpc.parent_id IS NOT NULL"

    # 主表读取
    df_niche = connect_niche_original(niche_key_sql)

    # 空数据判断
    if df_niche.empty:
        row_start = row_start + row_increment
        continue

    # asin表读取
    data_date = str(config.oe_database)[-8:]
    if config.oe_database.startswith("oe_us") and data_date < str(20240101):
        print("TRUE")
        df_asin = connect_niche_original(niche_asin_sql_us)
        df_cpc = connect_niche_original(niche_cpc_us)
        if df_cpc.empty:
            df_cpc = connect_niche(niche_sql.niche_cpc)
    elif config.oe_database.startswith("oe_us"):
        print("FLASE")
        df_asin = connect_niche_original(niche_asin_sql)
        df_cpc = connect_niche_original(niche_cpc)
    else:
        df_asin = connect_niche_original(niche_asin_sql_us)
        df_cpc = connect_niche(niche_sql.niche_cpc)

    # 其他表读取
    df_keywords = connect_niche_original(niche_keywords_sql)
    df_trends = connect_niche_original(niche_trends_sql)
    df_oe_category = connect_niche_original(niche_sql.niche_category_sql)
    df_famous_brand = connect_niche(niche_sql.niche_famous_brand)
    df_niche_x = connect_niche(niche_sql.niche_x)
    df_season = connect_niche(niche_sql.niche_season)
    df_translate = connect_niche(niche_sql.niche_translate)

    # 2.数据预处理
    # 站点修正
    df_niche = niche_mkid_complete(df_niche)
    df_keywords = niche_mkid_complete(df_keywords)
    df_trends = niche_mkid_complete(df_trends)
    df_asin = niche_mkid_complete(df_asin)
    df_cpc = niche_mkid_complete(df_cpc)

    # 字段类型修正
    # niche表
    niche_list_1 = ["search_volume_T90", "search_volume_t_360", "minimum_units_sold_t_90", "maximum_units_sold_t_90",
                    "minimum_units_sold_t_360", "maximum_units_sold_t_360", "product_count", "product_count_qoq",
                    "product_count_yoy", "brand_count_t360_count_qoq", "brand_count_t360_count_yoy",
                    "avg_detail_page_quality_currentvalue", "brand_count_t360_currentvalue",
                    "avg_review_count_currentvalue"]
    convert_type(df_niche, niche_list_1, 0)

    niche_list_2 = ["search_volume_growth_t_90", "search_volume_growth_t_360", "search_volume_growth_t_90",
                    "search_volume_growth_t_360", "sponsored_products_percentage_t360_count_qoq",
                    "sponsored_products_percentage_t360_count_yoy", "top5_products_click_share_t360_count_qoq",
                    "top5_products_click_share_t360_count_yoy", "top5_brands_click_share_t360_count_qoq",
                    "top5_brands_click_share_t360_count_yoy", "avg_oosrate_t360_count_qoq",
                    "avg_oosrate_t360_count_yoy"]
    convert_type(df_niche, niche_list_2, 4)

    niche_list_3 = ["avgPriceT360", "avg_brand_age_t360_count_qoq", "avg_brand_age_t360_count_yoy",
                    "new_products_launched_t180_count_qoq", "new_products_launched_t180_count_yoy",
                    "successful_launches_t180_count_qoq", "successful_launches_t180_count_yoy",
                    "avg_review_rating_currentvalue"]
    convert_type(df_niche, niche_list_3, 2)

    # trends表
    df_trends['dataset_date'] = pd.to_datetime(df_trends['dataset_date'], errors='coerce', dayfirst=False,
                                               yearfirst=False, format='%Y-%m-%d')
    df_trends['search_volume_t_7'] = df_trends['search_volume_t_7'].round(0)
    df_trends['product_count_7'] = df_trends['product_count_7'].round(0)
    df_trends['average_price_t_7'] = df_trends['average_price_t_7'].astype('float64').round(decimals=2)
    df_trends['search_conversion_rate_t_7'] = df_trends['search_conversion_rate_t_7'].astype('float64').round(
        decimals=4)

    # asin表
    niche_asin_list_1 = ["asin_click_count_t_360", "total_reviews"]
    convert_type(df_asin, niche_asin_list_1, 0)

    niche_asin_list_2 = ["avg_price", "minimum_price", "maximum_price", "referral_fee", "fba_fee", "avg_bsr",
                         'customer_rating']
    convert_type(df_asin, niche_asin_list_2, 2)

    niche_asin_list_4 = ["asin_click_share_t_360", "length", "breadth", "height", "weight"]
    convert_type(df_asin, niche_asin_list_4, 4)

    # keywords表
    niche_keywords_list_1 = ["click_share", "search_volume_qoq", "click_share_t_360"]
    convert_type(df_keywords, niche_keywords_list_1, 4)

    niche_keywords_list_2 = ["search_volume_t_90", "search_volume_t_360"]
    convert_type(df_keywords, niche_keywords_list_2, 0)

    df_keywords['search_conversion_rate_t_360'] = df_keywords['search_conversion_rate_t_360'].fillna(0)
    df_keywords['search_conversion_rate_t_360'] = df_keywords['search_conversion_rate_t_360'].round(decimals=6)

    # cpc表
    df_cpc['search_volume_t_360'] = df_cpc['search_volume_t_360'].round(0)
    df_cpc['bid_rangeMedian'] = df_cpc['bid_rangeMedian'].astype('float64').round(decimals=2)

    # 2.1 niche表处理
    # 总点击量
    df_niche_click = df_asin.groupby('niche_id')['asin_click_count_t_360'].sum()
    df_niche_click = df_convert_column(df_niche_click, "总点击量", 'niche_id', 0)

    # 总订单量_KW计算
    df_keywords['order_kw_360'] = np.where(df_keywords['click_share_t_360'] * 1 > 0, round(
        df_keywords['search_volume_t_360'] * df_keywords['search_conversion_rate_t_360'], 2), np.nan)

    df_keywords_order = df_keywords.groupby('niche_id')['order_kw_360'].sum()
    df_keywords_order = df_convert_column(df_keywords_order, "总订单量_KW", 'niche_id', 0)

    # CR_KW计算
    df_keywords_CR = df_keywords_order.merge(df_niche_click, how="right", on="niche_id")
    df_keywords_CR['CR_KW'] = np.fmin(1, np.where(df_keywords_CR['总点击量'] * 1 > 0,
                                                  round(df_keywords_CR['总订单量_KW'] / df_keywords_CR['总点击量'], 4), np.nan))

    # 图片URL
    df_asin_url = df_asin[['id', 'niche_id', 'asin', 'asin_image_url']]
    df_niche_url = df_asin_url.drop_duplicates(subset=['niche_id'], keep="first", inplace=False)

    # 总售出件数_90,总售出件数_360
    df_niche['总售出件数_90'] = round((df_niche['minimum_units_sold_t_90'] + df_niche['maximum_units_sold_t_90']) / 2, 0)
    df_niche['总售出件数_360'] = round((df_niche['minimum_units_sold_t_360'] + df_niche['maximum_units_sold_t_360']) / 2, 0)

    # 平均每单件数_KW
    df_niche = df_niche.merge(df_keywords_CR, how="left", on="niche_id")
    df_niche['平均每单件数_KW'] = np.where(df_niche['总订单量_KW'] * 1 > 0, round(df_niche['总售出件数_360'] / df_niche['总订单量_KW'], 2),
                                     np.nan)

    # 最新周搜索量,最新周SCR,数据更新时间
    trend_group_new = df_trends.groupby('niche_id').apply(
        lambda t: t[t.dataset_date == t.dataset_date.max()], include_groups=False)  # 在分组中过滤出Count最大的行

    df_date_new = trend_group_new[['dataset_date', 'search_volume_t_7', 'search_conversion_rate_t_7']]

    # 周SCR_MAX_Date,周SCR_MIN_Date
    trends_group_max_date = df_trends.groupby('niche_id').apply(
        lambda t: t[t.search_conversion_rate_t_7 == t.search_conversion_rate_t_7.max()])
    trends_group_max_date = trends_group_max_date.drop_duplicates(subset=['niche_id'], keep="first", inplace=False)
    df_scr_max_date = trends_group_max_date[['dataset_date']]
    df_scr_max_date.columns = ['周SCR_MAX_Date']

    trends_group_min_date = df_trends.groupby('niche_id').apply(
        lambda s: s[s.search_conversion_rate_t_7 == s.search_conversion_rate_t_7.min()])
    trends_group_min_date = trends_group_min_date.drop_duplicates(subset=['niche_id'], keep="first", inplace=False)
    df_scr_min_date = trends_group_min_date[['dataset_date']]
    df_scr_min_date.columns = ['周SCR_MIN_Date']

    # 趋势覆盖周数
    df_trends_cover = df_trends.groupby('niche_id')['dataset_date'].count()
    df_trends_cover = df_convert_column(df_trends_cover, "趋势覆盖周数", 'niche_id', 0)

    # 周SCR_AVG,周SCR_MAX,周SCR_MIN,avg_review_rating_currentvalue,avg_review_count_currentvalue
    df_scr_avg = df_trends.groupby('niche_id')['search_conversion_rate_t_7'].mean()
    df_scr_avg = df_convert_column(df_scr_avg, "周SCR_AVG", 'niche_id', 4)

    df_scr_max = df_trends.groupby('niche_id')['search_conversion_rate_t_7'].max()
    df_scr_max = df_convert_column(df_scr_max, "周SCR_MAX", 'niche_id', 4)

    df_scr_min = df_trends.groupby('niche_id')['search_conversion_rate_t_7'].min()
    df_scr_min = df_convert_column(df_scr_min, "周SCR_MIN", 'niche_id', 4)

    # 完善表格
    df_cal_niche = df_niche.merge(df_scr_avg, how="left", on="niche_id").merge(df_scr_max, how="left", on="niche_id") \
        .merge(df_scr_min, how="left", on="niche_id").merge(df_date_new, how="left", on="niche_id") \
        .merge(df_scr_max_date, how="left", on="niche_id").merge(df_scr_min_date, how="left", on="niche_id") \
        .merge(df_trends_cover, how="left", on="niche_id").merge(df_niche_url, how="left", on="niche_id")

    # 平均每单件数_trends
    df_cal_niche['平均每单件数_trends'] = np.where(df_cal_niche['search_volume_t_360'] * df_cal_niche['周SCR_AVG'] * 1 > 0,
                                             round(df_cal_niche['总售出件数_360'] / (
                                                     df_cal_niche['search_volume_t_360'] * df_cal_niche['周SCR_AVG']),
                                                   2), np.nan)

    # 平均每单件数
    df_cal_niche['平均每单件数_trends'] = np.where(df_cal_niche['平均每单件数_trends'] * 1 > 0, df_cal_niche['平均每单件数_trends'],
                                             df_cal_niche['平均每单件数_KW'])
    df_cal_niche['平均每单件数_KW'] = np.where(df_cal_niche['平均每单件数_KW'] * 1 > 0, df_cal_niche['平均每单件数_KW'],
                                         df_cal_niche['平均每单件数_trends'])
    df_cal_niche['平均每单件数'] = np.fmax(1, np.where(df_cal_niche['平均每单件数_trends'] * df_cal_niche['平均每单件数_KW'] * 1 > 0,
                                                 round(df_cal_niche['平均每单件数_trends'] * 0.618 + df_cal_niche[
                                                     '平均每单件数_KW'] * 0.382, 2), df_cal_niche['平均每单件数_trends']))
    df_niche['平均每单件数'] = df_cal_niche['平均每单件数']

    # 品牌平均年龄
    df_cal_niche['avg_brand_age_t360_count_qoq'] = round(df_cal_niche['avg_brand_age_t360_count_qoq'] / 12, 2)
    df_cal_niche['avg_brand_age_t360_count_yoy'] = round(df_cal_niche['avg_brand_age_t360_count_yoy'] / 12, 2)

    # 字段整合到表
    df_niche_integrate = df_cal_niche[
        ['利基站点',
         'mkid',
         'niche_title',
         'niche_id',
         'sales_potential_score',
         '趋势覆盖周数',
         'avgPriceT360',
         'product_count',
         'search_volume_T90',
         'search_volume_t_360',
         'search_volume_growth_t_90',
         'search_volume_growth_t_360',
         '总售出件数_90',
         '总售出件数_360',
         '平均每单件数',
         'search_volume_t_7',
         'search_conversion_rate_t_7',
         '周SCR_AVG',
         '周SCR_MAX',
         '周SCR_MIN',
         '周SCR_MAX_Date',
         '周SCR_MIN_Date',
         'avg_review_rating_currentvalue',
         'avg_review_count_currentvalue',
         'asin_image_url',
         'dataset_date',
         'product_count_qoq',
         'product_count_yoy',
         'sponsored_products_percentage_t360_count_qoq',
         'sponsored_products_percentage_t360_count_yoy',
         'top5_products_click_share_t360_count_qoq',
         'top5_products_click_share_t360_count_yoy',
         'top5_brands_click_share_t360_count_qoq',
         'top5_brands_click_share_t360_count_yoy',
         'brand_count_t360_count_qoq',
         'brand_count_t360_count_yoy',
         'avg_brand_age_t360_count_qoq',
         'avg_brand_age_t360_count_yoy',
         'new_products_launched_t180_count_qoq',
         'new_products_launched_t180_count_yoy',
         'successful_launches_t180_count_qoq',
         'successful_launches_t180_count_yoy',
         'avg_oosrate_t360_count_qoq',
         'avg_oosrate_t360_count_yoy',
         'avg_detail_page_quality_currentvalue']]

    df_niche_integrate = df_clear(df_niche_integrate, '利基站点')

    df_niche_integrate.rename(
        columns={
            'mkid': "Station",
            'niche_id': "Niche_ID",
            'niche_title': "Niche",
            'sales_potential_score': "Sales Potential Score",
            '趋势覆盖周数': 'Data Cover Weeks',
            'avgPriceT360': 'Average Price',
            'product_count': 'Top Clicked Products',
            'search_volume_T90': 'Search Volume_90',
            'search_volume_t_360': 'Search Volume_360',
            'search_volume_growth_t_90': 'Search Volume Growth_90',
            'search_volume_growth_t_360': 'Search Volume Growth_360',
            '总售出件数_90': 'Units Sold_90',
            '总售出件数_360': 'Units Sold_360',
            '平均每单件数': 'Products Per Order',
            'search_volume_t_7': 'Newest Week Search Volume',
            'search_conversion_rate_t_7': 'Newest Week Search Conversion Rate',
            '周SCR_AVG': 'Average Week Search Conversion Rate',
            '周SCR_MAX': 'Max Week Search Conversion Rate',
            '周SCR_MIN': 'Min Week Search Conversion Rate',
            '周SCR_MAX_Date': 'Max Week Search Volume Date',
            '周SCR_MIN_Date': 'Min Week Search Volume Date',
            'avg_review_rating_currentvalue': 'Average Review Rating',
            'avg_review_count_currentvalue': 'Average Rating',
            'asin_image_url': 'Image URL',
            'dataset_date': 'Data Update Date',
            'product_count_qoq': 'Number of Products_90',
            'product_count_yoy': 'Number of Products_360',
            'sponsored_products_percentage_t360_count_qoq': 'Products Using Sponsored_90',
            'sponsored_products_percentage_t360_count_yoy': 'Products Using Sponsored_360',
            'top5_products_click_share_t360_count_qoq': 'Top 5 Products Click Share_90',
            'top5_products_click_share_t360_count_yoy': 'Top 5 Products Click Share_360',
            'top5_brands_click_share_t360_count_qoq': 'Top 5 Brands Click Share_90',
            'top5_brands_click_share_t360_count_yoy': 'Top 5 Brands Click Share_360',
            'brand_count_t360_count_qoq': 'Number of Brands_90',
            'brand_count_t360_count_yoy': 'Number of Brands_360',
            'avg_brand_age_t360_count_qoq': 'Average Brand Age in Niche_90',
            'avg_brand_age_t360_count_yoy': 'Average Brand Age in Niche_360',
            'new_products_launched_t180_count_qoq': 'New Products Launched (past 180 days)_90',
            'new_products_launched_t180_count_yoy': 'New Products Launched (past 180 days)_360',
            'successful_launches_t180_count_qoq': 'Successful Launches (past 180 days)_90',
            'successful_launches_t180_count_yoy': 'Successful Launches (past 180 days)_360',
            'avg_oosrate_t360_count_qoq': 'Average Out of Stock Rate_90',
            'avg_oosrate_t360_count_yoy': 'Average Out of Stock Rate_360',
            'avg_detail_page_quality_currentvalue': 'Average Product Listing Quality_90'}, inplace=True)

    # 2.2 niche_size表处理

    # 类目数据处理
    # 使用applymap()函数将字母转换为小写
    # df_oe_category = df_oe_category.applymap(lambda x: x.lower() if isinstance(x, str) else x)
    df_oe_category = df_oe_category.apply(lambda col: col.map(lambda x: x.lower() if isinstance(x, str) else x))
    df_oe_category['类目id'] = df_oe_category['二级类目名称'] + ":" + df_oe_category['三级类目名称']
    df_oe_category = df_oe_category[['类目id', '一级类目名称', '二级类目名称']]

    # ASIN1-3数据
    df_asin['category'] = df_asin['category'].str.replace("/", ":")
    df_asin['asin_click_sort'] = df_asin['asin_click_count_t_360'].groupby(df_asin['niche_id']).rank(ascending=False)

    df_asin1 = niche_top_asin(df_asin, 1)
    df_asin2 = niche_top_asin(df_asin, 2)
    df_asin3 = niche_top_asin(df_asin, 3)

    # 知名品牌识别
    df_asin1 = df_asin1.merge(df_famous_brand, how="left", left_on="ASIN1品牌", right_on="关键词")
    df_asin2 = df_asin2.merge(df_famous_brand, how="left", left_on="ASIN2品牌", right_on="关键词")
    df_asin3 = df_asin3.merge(df_famous_brand, how="left", left_on="ASIN3品牌", right_on="关键词")

    df_top_asin = df_asin1.merge(df_asin2, how="left", on="niche_id")
    df_top_asin = df_top_asin.merge(df_asin3, how="left", on="niche_id")
    df_top_asin['知名品牌'] = df_top_asin['类型'].fillna(df_top_asin['类型_x'])
    df_top_asin['知名品牌'] = df_top_asin['知名品牌'].fillna(df_top_asin['类型_y'])
    df_top_asin['知名品牌'] = df_top_asin['知名品牌'].fillna('')

    # 存疑利基

    # 加权平均价计算
    df_asin_w_avg = df_asin.query('avg_price > 0')
    df_price_w_avg = df_asin_w_avg.groupby('niche_id').apply(wavg, 'avg_price', 'asin_click_share_t_360',
                                                             include_groups=False)
    df_price_w_avg = df_convert_column(df_price_w_avg, "加权平均价", 'niche_id', 2)

    # 规格相关数据计算
    if df_asin_w_avg.empty:
        df_size_w_avg = df_niche[['niche_id']]
        niche_smile_m_list = ['是否探索', 'Monetary', '平均价格', '加权平均价', '体积超重占比', '重量', '体积重', '头程', '实抛偏差率', '规格分布', 'FBA',
                              '头程占比', 'FBA占比', '货值占比', 'FBA货值比', '资金利用效率1', '资金利用效率2', '资金利用效率AVG', '营销前毛利反推', '实抛分布',
                              '重量分布']
        for smile_m in niche_smile_m_list:
            df_size_w_avg[smile_m] = ''
    else:
        df_asin_w_avg['佣金占比_asin'] = np.where(
            (df_asin_w_avg['referral_fee'] * 1 > 0) & (df_asin_w_avg['referral_fee'] < df_asin_w_avg['avg_price']),
            round(df_asin_w_avg['referral_fee'] / df_asin_w_avg['avg_price'], 4), np.nan)
        df_asin_w_avg['体积重量_asin'] = np.where(
            df_asin_w_avg['length'] * df_asin_w_avg['breadth'] * df_asin_w_avg['height'] * 1 > 0,
            round(df_asin_w_avg['length'] * df_asin_w_avg['breadth'] * df_asin_w_avg[
                'height'] / 139, 4), np.nan)
        df_asin_w_avg['计算重量_asin'] = np.where(df_asin_w_avg['体积重量_asin'] * df_asin_w_avg['weight'] * 1 > 0,
                                              df_asin_w_avg[['体积重量_asin', 'weight']].apply(max, axis=1), np.nan)
        df_asin_w_avg['实抛偏差率_asin_a'] = np.where(df_asin_w_avg['体积重量_asin'] * df_asin_w_avg['weight'] * 1 > 0,
                                                 df_asin_w_avg[['体积重量_asin', 'weight']].apply(min, axis=1), np.nan)
        df_asin_w_avg['实抛偏差率_asin'] = np.where(df_asin_w_avg['体积重量_asin'] * df_asin_w_avg['weight'] * 1 > 0,
                                               round(df_asin_w_avg['计算重量_asin'] / df_asin_w_avg['实抛偏差率_asin_a'] - 1, 4),
                                               np.nan)
        df_asin_w_avg['体积超重占比_a'] = np.where(df_asin_w_avg['体积重量_asin'] * df_asin_w_avg['weight'] * 1 > 0,
                                             abs(df_asin_w_avg['weight'] - df_asin_w_avg['体积重量_asin']), np.nan)
        df_asin_w_avg['体积超重占比_asin'] = np.where(df_asin_w_avg['体积重量_asin'] * df_asin_w_avg['weight'] * 1 > 0,
                                                round(df_asin_w_avg['体积超重占比_a'] / df_asin_w_avg['计算重量_asin'], 4),
                                                np.nan)
        df_asin_w_avg['海运运费_asin'] = np.where(df_asin_w_avg['计算重量_asin'] < 5, para.shipping_price_s,
                                              para.shipping_price_f)
        df_asin_w_avg['头程_asin_a'] = round(df_asin_w_avg['计算重量_asin'] * 0.45 * df_asin_w_avg['海运运费_asin'], 2)
        df_asin_w_avg['头程_asin'] = np.where(df_asin_w_avg['头程_asin_a'] < df_asin_w_avg['avg_price'],
                                            df_asin_w_avg['头程_asin_a'],
                                            np.nan)
        df_asin_w_avg['FBA占比_asin'] = np.where(
            (df_asin_w_avg['fba_fee'] * 1 > 0) & (df_asin_w_avg['fba_fee'] < df_asin_w_avg['avg_price']),
            round(df_asin_w_avg['fba_fee'] / df_asin_w_avg['avg_price'], 4), np.nan)
        df_asin_w_avg['头程占比_asin'] = np.where(df_asin_w_avg['头程_asin'] * 1 > 0,
                                              round(df_asin_w_avg['头程_asin'] / df_asin_w_avg['avg_price'], 4), np.nan)

        # 聚合值计算
        df_referral_w_avg = df_asin_w_avg.groupby('niche_id').apply(wavg, '佣金占比_asin', 'asin_click_share_t_360',
                                                                    include_groups=False)
        df_referral_w_avg = df_convert_column(df_referral_w_avg, "佣金占比", 'niche_id', 2)

        df_fba_w_avg = df_asin_w_avg.groupby('niche_id').apply(wavg, 'fba_fee', 'asin_click_share_t_360',
                                                               include_groups=False)
        df_fba_w_avg = df_convert_column(df_fba_w_avg, "FBA", 'niche_id', 2)

        df_shipping_w_avg = df_asin_w_avg.groupby('niche_id').apply(wavg, '头程_asin', 'asin_click_share_t_360',
                                                                    include_groups=False)
        df_shipping_w_avg = df_convert_column(df_shipping_w_avg, "头程", 'niche_id', 2)

        df_weight_w_avg = df_asin_w_avg.groupby('niche_id').apply(wavg, 'weight', 'asin_click_share_t_360',
                                                                  include_groups=False)
        df_weight_w_avg = df_convert_column(df_weight_w_avg, "重量", 'niche_id', 2)

        df_weight_v_w_avg = df_asin_w_avg.groupby('niche_id').apply(wavg, '体积重量_asin', 'asin_click_share_t_360',
                                                                    include_groups=False)
        df_weight_v_w_avg = df_convert_column(df_weight_v_w_avg, "体积重", 'niche_id', 2)

        df_weight_cal_w_avg = df_asin_w_avg.groupby('niche_id').apply(wavg, '计算重量_asin', 'asin_click_share_t_360',
                                                                      include_groups=False)
        df_weight_cal_w_avg = df_convert_column(df_weight_cal_w_avg, "计算重量", 'niche_id', 2)

        df_weight_r_w_avg = df_asin_w_avg.groupby('niche_id').apply(wavg, '实抛偏差率_asin', 'asin_click_share_t_360',
                                                                    include_groups=False)
        df_weight_r_w_avg = df_convert_column(df_weight_r_w_avg, "实抛偏差率", 'niche_id', 2)

        df_weight_v_r_w_avg = df_asin_w_avg.groupby('niche_id').apply(wavg, '体积超重占比_asin', 'asin_click_share_t_360',
                                                                      include_groups=False)
        df_weight_v_r_w_avg = df_convert_column(df_weight_v_r_w_avg, "体积超重占比", 'niche_id', 2)

        df_fba_r_w_avg = df_asin_w_avg.groupby('niche_id').apply(wavg, 'FBA占比_asin', 'asin_click_share_t_360',
                                                                 include_groups=False)
        df_fba_r_w_avg = df_convert_column(df_fba_r_w_avg, "FBA占比", 'niche_id', 4)

        df_shipping_r_w_avg = df_asin_w_avg.groupby('niche_id').apply(wavg, '头程占比_asin', 'asin_click_share_t_360',
                                                                      include_groups=False)
        df_shipping_r_w_avg = df_convert_column(df_shipping_r_w_avg, "头程占比", 'niche_id', 4)

        # 数据分区
        df_cut(df_fba_w_avg, 'FBA', '规格分布', [-99, 0.01, 6.89, 9.73, 158.49, 999], ["", "轻小件", "小件", "大件", "超大件"])
        df_cut(df_weight_cal_w_avg, '计算重量', '重量分布', [-99, 0.01, 0.3, 1, 5, 10, 100], ["", "轻小", "偏轻", "常规", "较重", "超重"])

        df_size_w_avg = df_price_w_avg.merge(df_referral_w_avg, how="left", on="niche_id") \
            .merge(df_fba_w_avg, how="left", on="niche_id").merge(df_shipping_w_avg, how="left", on="niche_id") \
            .merge(df_weight_w_avg, how="left", on="niche_id").merge(df_weight_v_w_avg, how="left", on="niche_id") \
            .merge(df_weight_r_w_avg, how="left", on="niche_id").merge(df_weight_v_r_w_avg, how="left", on="niche_id") \
            .merge(df_fba_r_w_avg, how="left", on="niche_id").merge(df_shipping_r_w_avg, how="left", on="niche_id") \
            .merge(df_weight_cal_w_avg, how="left", on="niche_id")

        size_w_avg_list = ["佣金占比", "FBA", "头程", "重量", "体积重", "计算重量", "实抛偏差率", "体积超重占比", "FBA占比", "头程占比"]
        convert_numeric(df_size_w_avg, size_w_avg_list)

        conditions = [(df_size_w_avg['实抛偏差率'] >= 1) & (df_size_w_avg['体积超重占比'] >= 0.5),
                      (df_size_w_avg['实抛偏差率'] >= 0.2) & (df_size_w_avg['实抛偏差率'] < 1) & (df_size_w_avg['体积超重占比'] >= 0.5),
                      (df_size_w_avg['实抛偏差率'] >= 1) & (df_size_w_avg['体积超重占比'] < 0.5),
                      (df_size_w_avg['实抛偏差率'] >= 0.2) & (df_size_w_avg['实抛偏差率'] < 1) & (df_size_w_avg['体积超重占比'] < 0.5),
                      (df_size_w_avg['实抛偏差率'] < 0.2)]
        labels = ["大抛重", "小抛重", "大实重", "小实重", "-"]
        df_size_w_avg['实抛分布'] = np.select(conditions, labels, default="-")

        # M相关指标计算
        df_size_w_avg['货值占比_pre'] = np.where(df_size_w_avg['佣金占比'] * df_size_w_avg['FBA占比'] * 1 > 0, round(
            1 - df_size_w_avg['佣金占比'] - para.exchange_loss - para.default_profit - df_size_w_avg['FBA占比'] -
            df_size_w_avg['头程占比'], 4), np.nan)
        df_size_w_avg['货值占比'] = df_size_w_avg['货值占比_pre'].clip(para.default_ratio, para.default_ratio_h)

        df_size_w_avg['FBA货值比'] = np.where(df_size_w_avg['货值占比'] * 1 > 0,
                                           round(df_size_w_avg['FBA占比'] / df_size_w_avg['货值占比'], 4),
                                           np.nan)
        df_size_w_avg['资金利用效率1_a'] = np.where(df_size_w_avg['货值占比'] * 1 > 0,
                                              para.default_profit + df_size_w_avg['货值占比'],
                                              para.default_profit)
        df_size_w_avg['资金利用效率1_b'] = df_size_w_avg['头程占比'] + abs(df_size_w_avg['货值占比'])
        df_size_w_avg['资金利用效率1'] = np.where(df_size_w_avg['货值占比'] * 1 > 0,
                                            round(df_size_w_avg['资金利用效率1_a'] / df_size_w_avg['资金利用效率1_b'], 2), np.nan)
        df_size_w_avg['资金利用效率2_a'] = np.where(df_size_w_avg['货值占比'] * 1 > 0,
                                              1 - df_size_w_avg['佣金占比'] - para.exchange_loss - df_size_w_avg['货值占比'] -
                                              df_size_w_avg['FBA占比'] -
                                              df_size_w_avg['头程占比'], np.nan)
        df_size_w_avg['资金利用效率2_b'] = df_size_w_avg['头程占比'] + para.default_ratio
        df_size_w_avg['资金利用效率2'] = np.where(df_size_w_avg['货值占比'] * 1 > 0,
                                            round(df_size_w_avg['资金利用效率2_a'] / df_size_w_avg['资金利用效率2_b'], 2), np.nan)
        df_size_w_avg['资金利用效率AVG'] = round((df_size_w_avg['资金利用效率1'] + df_size_w_avg['资金利用效率2']) / 2, 2)
        df_size_w_avg['营销前毛利反推'] = df_size_w_avg['资金利用效率2_a']

        size_avg_list = ["货值占比", "FBA货值比", "资金利用效率1_a", "资金利用效率1_b", "资金利用效率1", "资金利用效率2_a", "资金利用效率2_b", "资金利用效率2",
                         "资金利用效率AVG", "营销前毛利反推"]
        convert_numeric(df_size_w_avg, size_avg_list)

    # ASIN平均上架年数
    df_asin = df_asin.merge(df_date_new, how="left", on="niche_id")

    # 移除任何包含缺失日期的行
    df_asin_launch_date = df_asin.query('avg_price_t_90 > 0')

    if df_asin_launch_date.empty:
        df_asin_age = df_niche[['niche_id']]
        df_asin_age['ASIN平均上架年数'] = None
    else:

        # 确保 'launch_date' 和 'dataset_date' 列都是日期格式
        df_asin['launch_date'] = pd.to_datetime(df_asin['launch_date'], errors='coerce')
        df_asin['dataset_date'] = pd.to_datetime(df_asin['dataset_date'], errors='coerce')

        # 计算上架天数
        df_asin_launch_date['上架天数'] = (df_asin_launch_date['dataset_date'] - df_asin_launch_date['launch_date']).dt.days

        df_asin_launch_date['上架年数'] = np.where(
            (df_asin_launch_date['上架天数'] / 365 > 0) & (df_asin_launch_date['上架天数'] / 365 < 50),
            round(df_asin_launch_date['上架天数'] / 365, 2), np.nan)

        # 异常年份剔除
        df_asin_age = df_asin_launch_date.groupby('niche_id')['上架年数'].mean()
        df_asin_age = df_convert_column(df_asin_age, "ASIN平均上架年数", 'niche_id', 2)

    # 完善表格
    df_cal_size = df_niche.merge(df_niche_click, how="left", on="niche_id") \
        .merge(df_price_w_avg, how="left", on="niche_id").merge(df_asin_age, how="left", on="niche_id") \
        .merge(df_top_asin, how="left", on="niche_id").merge(df_date_new, how="left", on="niche_id") \
        .merge(df_size_w_avg, how="left", on="niche_id")

    # 类目路径相关字段
    df_cal_size['一级类目'] = df_cal_size['ASIN1一级类目'].fillna(df_cal_size['ASIN2一级类目'].fillna(df_cal_size['ASIN3一级类目']))
    df_cal_size['完整类名'] = df_cal_size['ASIN1完整类名'].fillna(df_cal_size['ASIN2完整类名'].fillna(df_cal_size['ASIN3完整类名']))
    df_cal_size['类目路径'] = df_cal_size['完整类名']

    # 剔除不能做的类目和利基
    # 使用 .loc 方法进行条件筛选和更新
    df_cal_size.loc[df_cal_size['ASIN1完整类名'].str.contains(para.regex_pattern_asin_x, na=False, regex=True) |
                    df_cal_size['ASIN2完整类名'].str.contains(para.regex_pattern_asin_x, na=False, regex=True) |
                    df_cal_size['ASIN3完整类名'].str.contains(para.regex_pattern_asin_x, na=False, regex=True) |
                    df_cal_size['niche_title'].str.contains(para.regex_pattern_niche_x, na=False,
                                                            regex=True), '食品类目'] = 'Y'
    df_cal_size.loc[
        df_cal_size['niche_title'].str.contains(para.regex_pattern_niche, na=False, regex=True), '食品类目'] = np.nan

    df_cal_size = df_cal_size.merge(df_niche_x, how="left", on="利基站点")
    df_cal_size['是否探索'] = np.where(df_cal_size['类目_X'] == "类目_X", "N", '')

    df_cal_size['是否探索'] = np.where(df_cal_size['食品类目'] == "Y", "N", df_cal_size['是否探索'])

    # 字段整合到表
    df_niche_size = df_cal_size[
        ['利基站点', 'mkid', 'niche_id', 'niche_title', '是否探索', '加权平均价_x', '体积重', '体积超重占比', 'FBA', '重量', '体积重', '头程',
         '实抛偏差率', 'ASIN平均上架年数', '总点击量_y', 'CR_KW', 'dataset_date', 'ASIN1', 'ASIN1品牌', 'ASIN1完整类名', 'ASIN2', 'ASIN2品牌',
         'ASIN2完整类名', 'ASIN3', 'ASIN3品牌', 'ASIN3完整类名', '知名品牌', '一级类目', '完整类名', '类目路径', '食品类目']]

    df_niche_size = df_clear(df_niche_size, '利基站点')

    df_niche_size.rename(
        columns={'mkid': "站点",
                 'niche_id': "所属利基ID",
                 'niche_title': "所属利基",
                 '加权平均价_x': "加权平均价",
                 'FBA': "加权平均计算FBA",
                 '重量': "加权平均重量",
                 '体积重': "加权平均体积重",
                 '总点击量_y': "总点击量",
                 'dataset_date': "数据更新时间"}, inplace=True)

    # 2.3 keywords表处理

    # 字段整合到表
    df_niche_keywords = df_keywords[
        ['利基站点', 'mkid', 'keyword', 'niche_title', 'search_volume_t_90', 'search_volume_t_360',
         'search_conversion_rate_t_360', 'click_share', 'search_volume_qoq']]

    df_niche_keywords['clear_id'] = df_niche_keywords['利基站点'] + " | " + df_niche_keywords['keyword']
    df_niche_keywords = df_clear(df_niche_keywords, 'clear_id')
    df_niche_keywords = df_niche_keywords.drop(['clear_id'], axis=1)

    df_niche_keywords.rename(
        columns={'mkid': "站点",
                 'keyword': "关键词",
                 'niche_title': "所属利基",
                 'search_volume_t_90': "KW搜索量_90",
                 'search_volume_t_360': "KW搜索量_360",
                 'search_conversion_rate_t_360': "KW平均SCR_360",
                 'click_share': "点击份额_360",
                 'search_volume_qoq': "搜索量增长(环比)_90"}, inplace=True)

    # 2.4 asin表处理
    if df_asin.empty:
        df_cal_price = df_niche_size[['niche_id', '利基站点', 'niche_title', 'mkid', 'avgPriceT360', '加权平均价']]
        df_cal_price = df_cal_price.merge(df_date_new, how="left", on="niche_id")
        df_cal_price['平均价格'] = df_cal_price['加权平均价'].fillna(df_cal_price['avgPriceT360'])
        cal_price_list = ['原平均价', '修正后平均价', '价格标准差', '价格集中度', '修正后加权平均价']
        for cal_price in cal_price_list:
            df_cal_price[cal_price] = ''
    else:
        # 标准差计算
        df_price_avg = df_asin.groupby('niche_id')['avg_price'].mean()
        df_price_avg = df_convert_column(df_price_avg, "原平均价", 'niche_id', 2)

        df_price_std = df_asin.groupby('niche_id')['avg_price'].std()
        df_price_std = df_convert_column(df_price_std, "原标准差", 'niche_id', 2)

        df_price_xxx = df_price_avg.merge(df_price_std, how='left', on='niche_id')
        df_price_xxx['价格标准'] = round(df_price_xxx['原平均价'] + 3 * df_price_xxx['原标准差'], 2)

        # 异常价格剔除
        df_asin_new = df_asin.merge(df_price_xxx, how='left', on='niche_id')
        df_asin_new['异常价格识别'] = round((df_asin_new['价格标准'] - df_asin_new['avg_price']), 2)

        df_cut(df_asin_new, '异常价格识别', '异常价格', [-5000, 0, 5000], ["Y", "N"])

        df_asin_price_repair = df_asin_new.query('异常价格识别>0')

        # 修正后价格计算
        df_price_repair_avg = df_asin_price_repair.groupby('niche_id')['avg_price'].mean()
        df_price_repair_avg = df_convert_column(df_price_repair_avg, "修正后平均价", 'niche_id', 2)

        df_price_repair_std = df_asin_price_repair.groupby('niche_id')['avg_price'].std()
        df_price_repair_std = df_convert_column(df_price_repair_std, "价格标准差", 'niche_id', 2)

        df_price_repair_w_avg = df_asin_price_repair.groupby('niche_id').apply(wavg, 'avg_price',
                                                                               'asin_click_share_t_360',
                                                                               include_groups=False)
        df_price_repair_w_avg = df_convert_column(df_price_repair_w_avg, "修正后加权平均价", 'niche_id', 2)

        # 完善表格
        df_cal_price = df_niche.merge(df_price_w_avg, how="left", on="niche_id") \
            .merge(df_price_avg, how="left", on="niche_id").merge(df_price_repair_avg, how="left", on="niche_id") \
            .merge(df_price_repair_std, how="left", on="niche_id") \
            .merge(df_price_repair_w_avg, how="left", on="niche_id").merge(df_date_new, how="left", on="niche_id")

        # 平均价格计算
        df_cal_price['价格集中度'] = round((1 - df_cal_price['价格标准差'] / df_cal_price['加权平均价']), 2)
        df_cal_price['平均价格'] = df_cal_price[['加权平均价', '修正后平均价']].min(axis=1)
        df_cal_price['平均价格'] = df_cal_price['平均价格'].fillna(df_cal_price['修正后平均价'])
        df_cal_price['平均价格'] = df_cal_price['平均价格'].fillna(df_cal_price['原平均价'])
        df_cal_price['平均价格'] = df_cal_price['平均价格'].fillna(df_cal_price['avgPriceT360'])

    # 字段整合到表
    df_niche_price = df_cal_price[
        ['利基站点', 'niche_title', 'mkid', 'dataset_date', '原平均价', '加权平均价', '修正后平均价', '平均价格', '价格标准差', '价格集中度',
         '修正后加权平均价']]

    df_niche_price = df_clear(df_niche_price, '利基站点')

    df_niche_price.rename(
        columns={'niche_title': "所属利基",
                 'mkid': "站点",
                 'dataset_date': "数据更新时间"}, inplace=True)

    # asin表字段整合到表
    df_niche_asin = df_asin[
        ['利基站点', 'niche_title', 'asin', 'mkid', 'avg_price', 'minimum_price', 'maximum_price', 'referral_fee',
         'fba_fee', 'asin_click_count_t_360', 'asin_click_share_t_360', 'total_reviews', 'customer_rating', 'brand',
         'launch_date', 'dataset_date']]

    df_niche_asin['clear_id'] = df_niche_asin['利基站点'] + " | " + df_niche_asin['asin']
    df_niche_asin = df_clear(df_niche_asin, 'clear_id')
    df_niche_asin = df_niche_asin.drop(['clear_id'], axis=1)

    df_niche_asin.rename(
        columns={'niche_title': "所属利基",
                 'asin': "ASIN",
                 'mkid': "站点",
                 'avg_price': "平均价_360",
                 'minimum_price': "最低价格",
                 'maximum_price': "最高价格",
                 'referral_fee': "平均销售佣金",
                 'fba_fee': "fba运费",
                 'asin_click_count_t_360': "点击数_360",
                 'asin_click_share_t_360': "点击份额_360",
                 'total_reviews': "总评分",
                 'customer_rating': "平均卖家评分",
                 'brand': "品牌",
                 'launch_date': "发布日期",
                 'dataset_date': "数据更新时间"}, inplace=True)

    # 2.5 trends表处理
    df_trends = df_trends.merge(df_date_new, how="left", on="niche_id")
    df_trends['周GMV'] = df_trends['search_volume_t_7_x'] * df_trends['search_conversion_rate_t_7_x'] * df_trends[
        'average_price_t_7'] * para.search_magnification
    df_trends['周订单量'] = df_trends['search_volume_t_7_x'] * df_trends['search_conversion_rate_t_7_x']
    df_mround(df_trends, '周GMV', '周GMV', 100)

    # 字段整合到表
    df_niche_trends = df_trends[
        ['利基站点', 'niche_title', 'mkid', 'dataset_date_x', 'search_volume_t_7_x', 'search_conversion_rate_t_7_x',
         'average_price_t_7', 'product_count_7', 'niche_id', '周GMV', '周订单量', 'dataset_date_y']]

    # 清洗
    df_niche_trends['dataset_date'] = df_niche_trends['dataset_date_x'].astype(str)
    df_niche_trends['clear_id'] = df_niche_trends['利基站点'] + " | " + df_niche_trends['dataset_date']
    df_niche_trends = df_clear(df_niche_trends, 'clear_id')
    df_niche_trends = df_niche_trends.drop(['clear_id', 'dataset_date'], axis=1)

    df_niche_trends.rename(
        columns={'niche_title': "所属利基",
                 'mkid': "站点",
                 'dataset_date_x': "所属日期",
                 'search_volume_t_7_x': "搜索量_7",
                 'search_conversion_rate_t_7_x': "转化率_7",
                 'average_price_t_7': "平均价格_7",
                 'product_count_7': "商品数量_7",
                 'dataset_date_y': "数据更新时间"}, inplace=True)

    # 3.niche_smile表处理
    df_niche_smile_pre = df_cal_price.merge(df_cal_size, how='left', on='利基站点')
    col_duplicate(df_niche_smile_pre)

    # smile_s表

    # 月均GMV
    df_niche_smile_pre['月均GMV'] = round(
        df_niche_smile_pre['总售出件数_360_x'] * df_niche_smile_pre[
            '平均价格'] / 12 * para.search_magnification / 100) * 100

    # Scale值计算
    conditions_s = [
        (df_niche_smile_pre['search_volume_t_360_x'] >= 10000 * 240) & (df_niche_smile_pre['月均GMV'] >= 10000 * 120),
        (df_niche_smile_pre['search_volume_t_360_x'] >= 10000 * 80) & (df_niche_smile_pre['月均GMV'] >= 10000 * 40),
        (df_niche_smile_pre['search_volume_t_360_x'] >= 10000 * 16) & (df_niche_smile_pre['月均GMV'] >= 10000 * 8),
        (df_niche_smile_pre['search_volume_t_360_x'] >= 10000 * 5) & (df_niche_smile_pre['月均GMV'] >= 10000 * 2.5),
        (df_niche_smile_pre['search_volume_t_360_x'] >= 10000 * 2) & (df_niche_smile_pre['月均GMV'] >= 10000 * 1)]
    labels_s = [5, 4, 3, 2, 1]
    df_niche_smile_pre['Scale'] = np.select(conditions_s, labels_s, default=1)
    df_niche_smile_pre['Scale'] = df_niche_smile_pre['Scale'].clip(1, 5)
    df_niche_smile_pre['Scale'] = df_niche_smile_pre['Scale'].fillna(1)

    df_niche_smile_pre['计算商品数'] = np.fmax(5, df_niche_smile_pre['product_count_x'])

    # 平均单品月销量
    df_niche_smile_pre['平均单品月销量'] = round(
        df_niche_smile_pre['总售出件数_360_x'] / df_niche_smile_pre['product_count_x'] / 12, 2)

    # 利基平均月销售额
    df_niche_smile_pre['利基平均月销售额'] = round(
        df_niche_smile_pre['平均单品月销量'] * df_niche_smile_pre['平均价格'] * para.search_magnification)

    # TOP5平均月销额
    df_niche_smile_pre['TOP5平均月销额'] = np.where(df_niche_smile_pre['top5_products_click_share_t360_count_yoy_x'] * 1 > 0,
                                               round(df_niche_smile_pre['月均GMV'] * df_niche_smile_pre[
                                                   'top5_products_click_share_t360_count_yoy_x'] / np.fmin(
                                                   df_niche_smile_pre['计算商品数'], 5)), df_niche_smile_pre['利基平均月销售额'])

    # 整理到表
    df_niche_s = df_niche_smile_pre[
        ['利基站点', 'niche_title_x', 'mkid_x', '是否探索', 'Scale', '平均价格', 'search_volume_t_360_x', '月均GMV', 'TOP5平均月销额',
         '总售出件数_360_x', 'search_volume_T90_x', '总售出件数_90_x', 'search_volume_t_7_x', 'dataset_date_x']]
    df_niche_s = df_clear(df_niche_s, '利基站点')
    df_niche_s.rename(
        columns={'niche_title_x': "所属利基",
                 'mkid_x': "站点",
                 'search_volume_t_360_x': "搜索量_360",
                 '总售出件数_360_x': "总售出件数_360",
                 'search_volume_T90_x': "搜索量_90",
                 '总售出件数_90_x': "总售出件数_90",
                 'search_volume_t_7_x': "最新周搜索量",
                 'dataset_date_x': "数据更新时间"}, inplace=True)

    # smile_m表

    # Monetary值计算
    conditions_m = [(df_niche_smile_pre['资金利用效率AVG'] >= 1.6) & (df_niche_smile_pre['营销前毛利反推'] >= 0.35),
                    (df_niche_smile_pre['资金利用效率AVG'] >= 1) & (df_niche_smile_pre['营销前毛利反推'] >= 0.3),
                    (df_niche_smile_pre['资金利用效率AVG'] >= 0.8) & (df_niche_smile_pre['营销前毛利反推'] >= 0.25),
                    (df_niche_smile_pre['资金利用效率AVG'] >= 0.6) & (df_niche_smile_pre['营销前毛利反推'] >= 0.2),
                    (df_niche_smile_pre['资金利用效率AVG'] < 0.4) & (df_niche_smile_pre['营销前毛利反推'] >= 0.15)]
    labels_m = [5, 4, 3, 2, 1]
    df_niche_smile_pre['Monetary'] = np.select(conditions_m, labels_m, default=np.nan)

    # 整理到表
    df_niche_m = df_niche_smile_pre[
        ['利基站点', 'niche_title_x', 'mkid_x', '是否探索', 'Monetary', '平均价格', '平均每单件数_x', '体积超重占比', '重量', '体积重', '头程',
         '实抛偏差率', '规格分布', 'FBA', '头程占比', 'FBA占比', '货值占比', 'FBA货值比', '资金利用效率1', '资金利用效率2', '资金利用效率AVG', '营销前毛利反推',
         '实抛分布', '重量分布', 'dataset_date_x']]
    df_niche_m = df_clear(df_niche_m, '利基站点')
    df_niche_m.rename(
        columns={'niche_title_x': "所属利基",
                 'mkid_x': "站点",
                 '平均每单件数_x': "平均每单件数",
                 '重量': "加权平均重量",
                 '体积重': "加权平均体积重",
                 'FBA': "加权平均FBA",
                 'dataset_date_x': "数据更新时间"}, inplace=True)

    # smile_l表

    # 知名品牌
    conditions_brand = [
        (df_niche_smile_pre['top5_brands_click_share_t360_count_yoy_x'] * 1 > 0.9) & (df_niche_smile_pre['Scale'] > 3),
        (df_niche_smile_pre['top5_brands_click_share_t360_count_yoy_x'] * 1 > 0.82) & (df_niche_smile_pre['Scale'] > 2),
        (df_niche_smile_pre['top5_brands_click_share_t360_count_yoy_x'] * 1 > 0.8) & (
                df_niche_smile_pre['知名品牌'] == "Y")]
    labels_brand = ["H", "M", "H"]

    # 知名品牌依赖
    df_niche_smile_pre['知名品牌依赖'] = np.select(conditions_brand, labels_brand)
    df_niche_smile_pre['知名品牌依赖'] = df_niche_smile_pre['知名品牌依赖'].replace('0', np.nan)

    df_niche_smile_pre['smile_l_pre'] = (df_niche_smile_pre['计算商品数'] - 5) * df_niche_smile_pre[
        'top5_products_click_share_t360_count_yoy_x']

    # 非TOP5单品月销量
    df_niche_smile_pre['非TOP5单品月销量'] = np.where(df_niche_smile_pre['smile_l_pre'] * 1 > 0,
                                                round(df_niche_smile_pre['总售出件数_360_x'] * (1 - df_niche_smile_pre[
                                                    'top5_products_click_share_t360_count_yoy_x']) / 12 / (
                                                              df_niche_smile_pre['product_count_x'] - 5)), 0)

    # 非TOP5平均月销额
    df_niche_smile_pre['非TOP5平均月销额'] = round(
        df_niche_smile_pre['非TOP5单品月销量'] * df_niche_smile_pre['平均价格'] * para.search_magnification)

    # 长尾指数
    df_niche_smile_pre['长尾指数'] = np.where(df_niche_smile_pre['利基平均月销售额'] * 1 > 0, round(
        (df_niche_smile_pre['非TOP5平均月销额'] / df_niche_smile_pre['利基平均月销售额']) / 0.05) * 0.05, np.nan)

    # 品牌长尾指数
    df_niche_smile_pre['品牌长尾指数'] = np.where(
        df_niche_smile_pre['brand_count_t360_currentvalue_x'] * df_niche_smile_pre['product_count_x'] > 0,
        round((df_niche_smile_pre['brand_count_t360_currentvalue_x'] / df_niche_smile_pre[
            'product_count_x']) / 0.05) * 0.05, np.nan)

    # Longtail值计算
    conditions_l = [(df_niche_smile_pre['长尾指数'] >= 0.7) & (df_niche_smile_pre['品牌长尾指数'] >= 0.6) & (
            df_niche_smile_pre['product_count_x'] >= 60) & (df_niche_smile_pre['非TOP5平均月销额'] >= 6000),
                    (df_niche_smile_pre['长尾指数'] >= 0.5) & (df_niche_smile_pre['品牌长尾指数'] >= 0.5) & (
                            df_niche_smile_pre['product_count_x'] >= 40),
                    (df_niche_smile_pre['长尾指数'] >= 0.35) & (df_niche_smile_pre['品牌长尾指数'] >= 0.4) & (
                            df_niche_smile_pre['product_count_x'] >= 15),
                    (df_niche_smile_pre['长尾指数'] >= 0.2) & (df_niche_smile_pre['product_count_x'] > 5)]
    labels_l = [5, 4, 3, 2]
    df_niche_smile_pre['Longtail'] = np.select(conditions_l, labels_l, default=1)

    df_niche_smile_pre['Longtail_bonus'] = np.where(df_niche_smile_pre['非TOP5平均月销额'] >= 10000, 0.5, 0) + np.where(
        df_niche_smile_pre['product_count_x'] >= 80, 0.5, 0)

    df_niche_smile_pre['Longtail_bonus'] = df_niche_smile_pre['Longtail_bonus'].clip(-1, 1)

    df_niche_smile_pre['Longtail'] = df_niche_smile_pre['Longtail'] + df_niche_smile_pre['Longtail_bonus']
    df_niche_smile_pre['Longtail'] = df_niche_smile_pre['Longtail'].clip(1, 5)
    df_niche_smile_pre['Longtail'] = df_niche_smile_pre['Longtail'].fillna(1)

    # 整理到表
    df_niche_smile_pre = df_niche_smile_pre.merge(df_niche_url, how='left', left_on='niche_id_x', right_on='niche_id')

    df_niche_a = df_niche_smile_pre[
        ['利基站点', 'niche_title_x', 'mkid_x', '是否探索', 'ASIN1', 'ASIN1品牌', 'ASIN1完整类名', 'ASIN2', 'ASIN2品牌', 'ASIN2完整类名',
         'ASIN3', 'ASIN3品牌', 'ASIN3完整类名', 'ASIN平均上架年数', '知名品牌', 'top5_brands_click_share_t360_count_yoy_x', '知名品牌依赖',
         'asin_image_url', '类目路径', '一级类目', 'dataset_date_x']]
    df_niche_a = df_clear(df_niche_a, '利基站点')
    df_niche_a.rename(
        columns={'niche_title_x': "所属利基",
                 'mkid_x': "站点",
                 'top5_brands_click_share_t360_count_yoy_x': "TOP5品牌点击占比_360",
                 'asin_image_url': "利基图片URL",
                 'dataset_date_x': "数据更新时间"}, inplace=True)

    df_niche_l = df_niche_smile_pre[
        ['利基站点', 'niche_title_x', 'mkid_x', '是否探索', 'Longtail', '平均价格', '长尾指数', 'product_count_x', '品牌长尾指数',
         'brand_count_t360_count_qoq_x', 'brand_count_t360_count_yoy_x', '非TOP5单品月销量', '平均单品月销量', '非TOP5平均月销额',
         '利基平均月销售额', 'top5_products_click_share_t360_count_qoq_x', 'top5_products_click_share_t360_count_yoy_x',
         'top5_brands_click_share_t360_count_qoq_x', 'top5_brands_click_share_t360_count_yoy_x', '知名品牌依赖',
         'dataset_date_x']]
    df_niche_l = df_clear(df_niche_l, '利基站点')
    df_niche_l.rename(
        columns={'niche_title_x': "所属利基",
                 'mkid_x': "站点",
                 'product_count_x': "点击最多的商品数",
                 'brand_count_t360_count_qoq_x': "品牌数量_90",
                 'brand_count_t360_count_yoy_x': "品牌数量_360",
                 'top5_products_click_share_t360_count_qoq_x': "TOP5产品点击占比_90",
                 'top5_products_click_share_t360_count_yoy_x': "TOP5产品点击占比_360",
                 'top5_brands_click_share_t360_count_qoq_x': "TOP5品牌点击占比_90",
                 'top5_brands_click_share_t360_count_yoy_x': "TOP5品牌点击占比_360",
                 'dataset_date_x': "数据更新时间"}, inplace=True)

    # smile_e表

    # 近半年新品成功率
    df_niche_smile_pre['近半年新品成功率'] = np.where(df_niche_smile_pre['new_products_launched_t180_count_yoy_x'] > 3,
                                              df_niche_smile_pre['successful_launches_t180_count_yoy_x'] /
                                              df_niche_smile_pre['new_products_launched_t180_count_yoy_x'], np.nan)

    # 平均缺货率
    df_niche_smile_pre['平均缺货率'] = round((df_niche_smile_pre['avg_oosrate_t360_count_qoq_x'] * 0.618 +
                                         df_niche_smile_pre['avg_oosrate_t360_count_yoy_x'] * 0.382) / 0.01) * 0.02

    # Emerging值计算
    sales_potential_score = df_niche_smile_pre['sales_potential_score_x'].values
    asin_launch_date = df_niche_smile_pre['ASIN平均上架年数'].values

    if any(sales_potential_score) or any(asin_launch_date):
        # 有销量潜力得分和ASIN平均上架年数
        conditions_e_1 = [
            (df_niche_smile_pre['search_volume_growth_t_360_x'] >= 0.4) & (
                    df_niche_smile_pre['search_volume_growth_t_90_x'] >= 0.2) & (
                    df_niche_smile_pre['ASIN平均上架年数'] < 2) & (
                    df_niche_smile_pre['avg_review_count_currentvalue_x'] < 1000) & (
                    df_niche_smile_pre['sales_potential_score_x'] >= 6),
            (df_niche_smile_pre['search_volume_growth_t_360_x'] >= 0.2) & (
                    df_niche_smile_pre['search_volume_growth_t_90_x'] >= -0.1) & (
                    df_niche_smile_pre['ASIN平均上架年数'] < 3.5) & (
                    df_niche_smile_pre['avg_review_count_currentvalue_x'] < 4000) & (
                    df_niche_smile_pre['sales_potential_score_x'] >= 6),
            (df_niche_smile_pre['search_volume_growth_t_360_x'] >= -0.1) & (
                    df_niche_smile_pre['avg_review_count_currentvalue_x'] < 8000) & (
                    df_niche_smile_pre['sales_potential_score_x'] >= 3),
            (df_niche_smile_pre['search_volume_growth_t_360_x'] >= -0.3)]
        labels_e_1 = [5, 4, 3, 2]
        df_niche_smile_pre['Emerging'] = np.select(conditions_e_1, labels_e_1, default=1)

        df_niche_smile_pre['Emerging_bonus'] = np.where(df_niche_smile_pre['avg_review_count_currentvalue_x'] < 800,
                                                        0.5, 0) + np.where(
            (df_niche_smile_pre['sales_potential_score_x'] >= 9) & (df_niche_smile_pre['近半年新品成功率'] >= 0.6), 0.5,
            0) + np.where(df_niche_smile_pre['ASIN平均上架年数'] < 1.5, 0.5, 0) + np.where(
            df_niche_smile_pre['avg_review_rating_currentvalue_x'] < 4.1, 0.5, 0) - np.where(
            df_niche_smile_pre['avg_review_count_currentvalue_x'] >= 8000, 0.5, 0)

        df_niche_smile_pre['Emerging_bonus'] = df_niche_smile_pre['Emerging_bonus'].clip(-1, 1)

        df_niche_smile_pre['Emerging'] = df_niche_smile_pre['Emerging'] + df_niche_smile_pre['Emerging_bonus']
    else:
        # 无销量潜力得分和ASIN平均上架年数
        conditions_e_2 = [
            (df_niche_smile_pre['search_volume_growth_t_360_x'] >= 0.4) & (
                    df_niche_smile_pre['search_volume_growth_t_90_x'] >= 0.2),
            (df_niche_smile_pre['search_volume_growth_t_360_x'] >= 0.2) & (
                    df_niche_smile_pre['search_volume_growth_t_90_x'] >= -0.1),
            (df_niche_smile_pre['search_volume_growth_t_360_x'] >= -0.1),
            (df_niche_smile_pre['search_volume_growth_t_360_x'] >= -0.3)]
        labels_e_2 = [5, 4, 3, 2]
        df_niche_smile_pre['Emerging'] = np.select(conditions_e_2, labels_e_2, default=1)
        df_niche_smile_pre['Emerging_bonus'] = np.where(df_niche_smile_pre['avg_review_count_currentvalue_x'] < 800,
                                                        0.5, 0) + np.where(df_niche_smile_pre['近半年新品成功率'] >= 0.6, 0.5,
                                                                           0) + np.where(
            df_niche_smile_pre['avg_review_rating_currentvalue_x'] < 4.1, 0.5, 0) - np.where(
            df_niche_smile_pre['avg_review_count_currentvalue_x'] >= 8000, 0.5, 0)

        df_niche_smile_pre['Emerging_bonus'] = df_niche_smile_pre['Emerging_bonus'].clip(-1, 1)

        df_niche_smile_pre['Emerging'] = df_niche_smile_pre['Emerging'] + df_niche_smile_pre['Emerging_bonus']

    df_niche_smile_pre['Emerging'] = df_niche_smile_pre['Emerging'].clip(1, 5)
    df_niche_smile_pre['Emerging'] = df_niche_smile_pre['Emerging'].fillna(1)

    # 留评意愿强度
    df_niche_smile_pre['留评意愿强度'] = np.where(
        df_niche_smile_pre['总售出件数_360_x'] * df_niche_smile_pre['ASIN平均上架年数'] * 1 > 0,
        df_niche_smile_pre['avg_review_count_currentvalue_x'] * df_niche_smile_pre['product_count_x'] /
        (df_niche_smile_pre['总售出件数_360_x'] * df_niche_smile_pre['ASIN平均上架年数']), np.nan)

    # 整理到表
    df_niche_smile_pre = df_niche_smile_pre.merge(df_trends_cover, how='left', left_on='niche_id_x',
                                                  right_on='niche_id')

    df_niche_e = df_niche_smile_pre[
        ['利基站点', 'niche_title_x', 'mkid_x', '是否探索', 'Emerging', 'search_volume_growth_t_360_x',
         'search_volume_growth_t_90_x', '趋势覆盖周数', 'ASIN平均上架年数', '留评意愿强度', 'avg_review_count_currentvalue_x',
         'sales_potential_score_x', '近半年新品成功率', 'new_products_launched_t180_count_qoq_x',
         'new_products_launched_t180_count_yoy_x', 'successful_launches_t180_count_qoq_x',
         'successful_launches_t180_count_yoy_x', '平均缺货率', 'avg_oosrate_t360_count_qoq_x',
         'avg_oosrate_t360_count_yoy_x', 'dataset_date_x']]
    df_niche_e = df_clear(df_niche_e, '利基站点')
    df_niche_e.rename(
        columns={'niche_title_x': "所属利基",
                 'mkid_x': "站点",
                 'search_volume_growth_t_360_x': "搜索量增长_360",
                 'search_volume_growth_t_90_x': "搜索量增长_90",
                 '趋势覆盖周数': "数据趋势覆盖周数",
                 'sales_potential_score_x': "销量潜力得分",
                 'new_products_launched_t180_count_qoq_x': "近180天上架新品数_90",
                 'new_products_launched_t180_count_yoy_x': "近180天上架新品数_360",
                 'successful_launches_t180_count_qoq_x': "近180天成功上架新品数_90",
                 'successful_launches_t180_count_yoy_x': "近180天成功上架新品数_360",
                 'avg_oosrate_t360_count_qoq_x': "平均缺货率_90",
                 'avg_oosrate_t360_count_yoy_x': "平均缺货率_360",
                 'dataset_date_x': "数据更新时间"}, inplace=True)

    # smile_i表

    # 搜索点击比
    df_niche_smile_pre['搜索点击比'] = np.where(df_niche_smile_pre['总点击量'] * 1 > 0,
                                           df_niche_smile_pre['总点击量'] / df_niche_smile_pre['search_volume_t_360_x'],
                                           np.nan)

    # CR
    df_niche_smile_pre = df_niche_smile_pre.merge(df_scr_avg, how='left', left_on='niche_id_x', right_on='niche_id')
    df_niche_smile_pre['CR_周SCR'] = np.fmin(1, np.where(df_niche_smile_pre['搜索点击比'] * 1 > 0,
                                                        df_niche_smile_pre['周SCR_AVG'] / df_niche_smile_pre['搜索点击比'],
                                                        df_niche_smile_pre['周SCR_AVG'] * 0.75))
    df_niche_smile_pre['CR'] = np.fmin(1, np.where(df_niche_smile_pre['CR_周SCR'] * 1 > 0,
                                                   df_niche_smile_pre['CR_KW_x'] * 0.618 + df_niche_smile_pre[
                                                       'CR_周SCR'] * 0.382,
                                                   df_niche_smile_pre['CR_KW_x']))

    # 转化净值
    df_niche_smile_pre['计算价格_UK'] = np.fmin(df_niche_smile_pre['平均价格'], para.CR_p_u_uk)
    df_niche_smile_pre['计算价格_OT'] = np.fmin(df_niche_smile_pre['平均价格'], para.CR_p_u_ot)
    df_niche_smile_pre['转化净值'] = np.where(df_niche_smile_pre['mkid_x'] == "UK", df_niche_smile_pre['计算价格_UK'] *
                                          df_niche_smile_pre['CR'] * np.fmin(2,
                                                                             df_niche_smile_pre['平均每单件数_x'].pow(0.9)),
                                          df_niche_smile_pre['计算价格_OT'] *
                                          df_niche_smile_pre['CR'] * np.fmin(2,
                                                                             df_niche_smile_pre['平均每单件数_x'].pow(0.9)))
    # 参数计算
    # f(VR)
    df_niche_smile_pre['f(VR)'] = np.where(df_niche_smile_pre['avg_review_rating_currentvalue_x'] * 1 > 0, np.fmax(
        para.b * np.fmax(((para.bl4 - df_niche_smile_pre['avg_review_rating_currentvalue_x']) / 0.5).pow(para.k4),
                         -para.e), 0), 0)

    # f(NS,SP)
    df_niche_smile_pre['f(NS,SP)'] = np.where(
        (df_niche_smile_pre['mkid_x'] == "US") & (df_niche_smile_pre['sales_potential_score_x'] * 1 > 0),
        para.c * np.where(df_niche_smile_pre['近半年新品成功率'] * 1 > 0, df_niche_smile_pre['近半年新品成功率'] / para.bl5,
                          1) * (df_niche_smile_pre['sales_potential_score_x'] / para.bl6) - 1,
        para.c * np.where(df_niche_smile_pre['近半年新品成功率'] * 1 > 0, df_niche_smile_pre['近半年新品成功率'] / para.bl5,
                          1) - 1)

    # f(CC)
    df_niche_smile_pre['f(CC)'] = np.where(df_niche_smile_pre['product_count_x'] * 1 > 0,
                                           (np.fmin(df_niche_smile_pre['product_count_x'], 150) / para.bl7).pow(
                                               para.k5), 1)
    df_niche_smile_pre['f(CC)'] = df_niche_smile_pre['f(CC)'].fillna(1)

    # f(SR,AR)
    # 搜索广告占比档计算
    df_niche_smile_pre = df_niche_smile_pre.merge(df_season, how='left', on='利基站点')
    df_niche_smile_pre['季节性标签'] = df_niche_smile_pre['季节性标签'].astype('object').fillna("_")

    df_niche_smile_pre['搜索广告占比档_1'] = df_niche_smile_pre['sponsored_products_percentage_t360_count_qoq_x'] * 0.2 + \
                                      df_niche_smile_pre['sponsored_products_percentage_t360_count_yoy_x'] * 0.8

    df_niche_smile_pre['搜索广告占比档_2'] = df_niche_smile_pre['sponsored_products_percentage_t360_count_qoq_x'] * 0.382 + \
                                      df_niche_smile_pre['sponsored_products_percentage_t360_count_yoy_x'] * 0.618

    df_niche_smile_pre['搜索广告占比档_3'] = df_niche_smile_pre['sponsored_products_percentage_t360_count_qoq_x'] * 0.618 + \
                                      df_niche_smile_pre['sponsored_products_percentage_t360_count_yoy_x'] * 0.382

    df_niche_smile_pre['搜索广告占比档'] = np.where(
        (df_niche_smile_pre['季节性标签'] == "大旺季") | (df_niche_smile_pre['季节性标签'] == "大节日"),
        df_niche_smile_pre['搜索广告占比档_1'],
        np.where((df_niche_smile_pre['季节性标签'] == "小旺季") | (df_niche_smile_pre['季节性标签'] == "小节日"),
                 df_niche_smile_pre['搜索广告占比档_2'], df_niche_smile_pre['搜索广告占比档_3']))

    df_niche_smile_pre['搜索广告占比档'] = np.where(df_niche_smile_pre['搜索广告占比档'] * 1 > 0,
                                             round(df_niche_smile_pre['搜索广告占比档'] / 0.02) * 0.02, np.nan)

    # 参数_搜索广告占比档
    df_niche_smile_pre['计算广告占比档'] = (np.fmax(np.where(df_niche_smile_pre['搜索广告占比档'] * 1 > 0,
                                                      df_niche_smile_pre['搜索广告占比档'], para.bl2), para.e) / para.bl2) ** (
                                        np.where(df_niche_smile_pre['搜索广告占比档'] > para.bl2, 1 / para.k2, para.k2))

    # 参数_缺货率
    df_niche_smile_pre['计算缺货率'] = ((1 - np.fmin(
        np.where(df_niche_smile_pre['平均缺货率'] * 1 > 0, df_niche_smile_pre['平均缺货率'], para.bl3),
        (1 - para.e))) / (1 - para.bl3)) ** (para.k3)

    df_niche_smile_pre['f(SR,AR)'] = np.where(df_niche_smile_pre['计算广告占比档'] * df_niche_smile_pre['计算缺货率'] > 0,
                                              df_niche_smile_pre['计算广告占比档'] * df_niche_smile_pre['计算缺货率'], 1)

    # bf(CC)和f(SR,AR)
    df_niche_smile_pre['bf(CC)和f(SR,AR)'] = para.d * (1 - df_niche_smile_pre['f(CC)'] / df_niche_smile_pre['f(SR,AR)'])

    # 转化净值偏差
    df_niche_smile_pre['转化净值偏差_US'] = round(para.CR_c_us * (df_niche_smile_pre['计算价格_OT'].pow(para.CR_k_us)), 2)
    df_niche_smile_pre['转化净值偏差_UK'] = round(para.CR_c_uk * (df_niche_smile_pre['计算价格_UK'].pow(para.CR_k_uk)), 2)
    df_niche_smile_pre['转化净值偏差_OT'] = round(para.CR_c_ot * (df_niche_smile_pre['计算价格_OT'].pow(para.CR_k_ot)), 2)

    df_niche_smile_pre['转化净值偏差'] = np.where(df_niche_smile_pre['mkid_x'] == "US", df_niche_smile_pre['转化净值偏差_US'],
                                            np.where(df_niche_smile_pre['mkid_x'] == "UK",
                                                     df_niche_smile_pre['转化净值偏差_UK'], df_niche_smile_pre['转化净值偏差_OT']))

    # 毛估CPC因子
    df_niche_smile_pre['毛估CPC因子'] = np.fmax(
        para.p1 * df_niche_smile_pre['转化净值偏差'] * df_niche_smile_pre['f(SR,AR)'] + para.p2 * df_niche_smile_pre[
            'f(VR)'] + para.p3 * df_niche_smile_pre['f(NS,SP)'] + para.p4 * df_niche_smile_pre['bf(CC)和f(SR,AR)'], 0)

    # 毛估蓝海度
    df_niche_smile_pre['毛估蓝海度'] = para.MS_a + para.MS_b / (
            1 + (para.MS_e ** (- (df_niche_smile_pre['毛估CPC因子'] - para.MS_cs * para.MS_c))))

    # 获取加权CPC
    if df_cpc.empty:
        df_niche_smile_pre['CPC'] = None
    else:
        df_cpc_avg = df_cpc.groupby('niche_id').apply(
            lambda x: (x['search_volume_t_360'] * x['bid_rangeMedian']).sum() / x['search_volume_t_360'].sum(),
            include_groups=False)
        df_cpc_avg = df_convert_column(df_cpc_avg, 'CPC', 'niche_id', 2)

        # CPC因子
        df_niche_smile_pre = df_niche_smile_pre.merge(df_cpc_avg, how='left', on='niche_id')

        if config.oe_database.startswith("oe_us"):
            df_niche_smile_pre['CPC'] = df_niche_smile_pre['CPC'].astype('float64').round(decimals=2)
        else:
            df_niche_smile_pre['CPC'] = df_niche_smile_pre['CPC'].astype('float64').round(decimals=2)

    df_niche_smile_pre['CPC因子'] = np.where(df_niche_smile_pre['转化净值'] * df_niche_smile_pre['CPC'] * 1 > 0,
                                           para.product_acos * df_niche_smile_pre['转化净值'] / df_niche_smile_pre['CPC'],
                                           np.nan)

    # 广告蓝海度
    df_niche_smile_pre['广告蓝海度'] = np.where(df_niche_smile_pre['CPC'] * 1 > 0, para.MS_a + para.MS_b / (
            1 + (para.MS_e ** (- (df_niche_smile_pre['CPC因子'] - para.MS_cs * para.MS_c)))), np.nan)

    # 蓝海度差异分
    df_niche_smile_pre['蓝海度差异分'] = np.where(df_niche_smile_pre['广告蓝海度'] * 1 > 0,
                                            df_niche_smile_pre['毛估蓝海度'] - df_niche_smile_pre['广告蓝海度'], np.nan)

    # 综合蓝海度
    df_niche_smile_pre['广告权重'] = df_niche_smile_pre['搜索广告占比档'] / para.p5

    df_niche_smile_pre['综合蓝海度'] = np.where(df_niche_smile_pre['CPC'] * 1 > 0,
                                           df_niche_smile_pre['广告权重'] * df_niche_smile_pre['广告蓝海度'] + (
                                                   1 - df_niche_smile_pre['广告权重']) * df_niche_smile_pre['毛估蓝海度'],
                                           df_niche_smile_pre['毛估蓝海度'])

    # 转化净值分级
    df_cut(df_niche_smile_pre, '转化净值偏差', '转化净值分级', [-99, 0.7, 0.9, 1.5, 2, 3, 99], ["很差", "较差", "正常", "良好", "优秀", "极棒"])
    df_niche_smile_pre['转化净值分级'] = df_niche_smile_pre['转化净值分级'].astype('object').fillna("-")

    # Involution值计算
    conditions_i = [
        (df_niche_smile_pre['综合蓝海度'] >= 4.25) & (df_niche_smile_pre['top5_brands_click_share_t360_count_yoy_x'] < 0.75),
        (df_niche_smile_pre['综合蓝海度'] >= 3.5), (df_niche_smile_pre['综合蓝海度'] >= 2.25),
        (df_niche_smile_pre['综合蓝海度'] >= 1.8)]
    labels_i = [5, 4, 3, 2]
    df_niche_smile_pre['Involution'] = np.select(conditions_i, labels_i, default=1)

    df_niche_smile_pre['Involution_bonus'] = np.where(
        (df_niche_smile_pre['avg_review_rating_currentvalue_x'] < 3.9) | (
                df_niche_smile_pre['avg_detail_page_quality_currentvalue_x'] < 85), 0.5,
        0) - np.where(((df_niche_smile_pre['知名品牌依赖'] == 'M') | (df_niche_smile_pre['知名品牌依赖'] == "H") | (
            df_niche_smile_pre['top5_brands_click_share_t360_count_yoy_x'] >= 0.8)) & (
                          (df_niche_smile_pre['ASIN平均上架年数'] >= 2)), 0.5, 0)

    df_niche_smile_pre['Involution_bonus'] = df_niche_smile_pre['Involution_bonus'].clip(-1, 1)

    df_niche_smile_pre['Involution'] = df_niche_smile_pre['Involution'] + df_niche_smile_pre['Involution_bonus']
    df_niche_smile_pre['Involution'] = df_niche_smile_pre['Involution'].clip(1, 5)
    df_niche_smile_pre['Involution'] = df_niche_smile_pre['Involution'].fillna(1)

    # 数据格式调整
    niche_smile_list_1 = ["转化净值", "转化净值偏差", "搜索点击比", "f(SR,AR)", "f(VR)", "f(NS,SP)", "f(CC)", "bf(CC)和f(SR,AR)",
                          "毛估CPC因子", "CPC因子", "广告蓝海度", "蓝海度差异分", "广告权重", "综合蓝海度", "搜索广告占比档"]
    convert_type(df_niche_smile_pre, niche_smile_list_1, 2)

    niche_smile_list_2 = ["CR", "平均缺货率"]
    convert_type(df_niche_smile_pre, niche_smile_list_2, 4)

    # 整理到表
    df_niche_i = df_niche_smile_pre[
        ['利基站点', 'niche_title_x', 'mkid_x', '是否探索', 'Involution', '平均价格', '总点击量', '平均每单件数_x', '转化净值', '转化净值偏差',
         '转化净值分级', '搜索点击比', '毛估蓝海度', 'CR', 'CR_KW_x', 'f(SR,AR)', 'f(VR)', 'f(NS,SP)', 'f(CC)', 'bf(CC)和f(SR,AR)',
         '毛估CPC因子', 'CPC', 'CPC因子', '广告蓝海度', '蓝海度差异分', '广告权重', '综合蓝海度', '搜索广告占比档',
         'sponsored_products_percentage_t360_count_qoq_x', 'sponsored_products_percentage_t360_count_yoy_x', '平均缺货率',
         'avg_oosrate_t360_count_qoq_x', 'avg_oosrate_t360_count_yoy_x', 'top5_brands_click_share_t360_count_yoy_x',
         'ASIN平均上架年数', 'avg_review_rating_currentvalue_x', 'avg_detail_page_quality_currentvalue_x', '周SCR_AVG', '一级类目',
         '季节性标签', 'dataset_date_x']]
    df_niche_i = df_clear(df_niche_i, '利基站点')
    df_niche_i.rename(
        columns={'niche_title_x': "所属利基",
                 'mkid_x': "站点",
                 '平均每单件数_x': "平均每单件数",
                 'CR_KW_x': "CR_KW",
                 'sponsored_products_percentage_t360_count_qoq_x': "搜索广告商品占比_90",
                 'sponsored_products_percentage_t360_count_yoy_x': "搜索广告商品占比_360",
                 'avg_oosrate_t360_count_qoq_x': "平均缺货率_90",
                 'avg_oosrate_t360_count_yoy_x': "平均缺货率_360",
                 'top5_brands_click_share_t360_count_yoy_x': "TOP5品牌点击占比_360",
                 'avg_detail_page_quality_currentvalue_x': "商品listing平均得分_90",
                 'dataset_date_x': "数据更新时间"}, inplace=True)

    # smile_h表

    # SMILE打分
    df_niche_smile_pre['SMILE打分'] = np.where(df_niche_smile_pre['Monetary'] * 1 > 0,
                                             df_niche_smile_pre['Scale'] + df_niche_smile_pre['Monetary'] +
                                             df_niche_smile_pre['Involution'] + df_niche_smile_pre['Longtail'] +
                                             df_niche_smile_pre['Emerging'],
                                             df_niche_smile_pre['Scale'] + df_niche_smile_pre['Involution'] +
                                             df_niche_smile_pre['Longtail'] + df_niche_smile_pre['Emerging'])

    # 业务线
    conditions_h = [
        (df_niche_smile_pre['Scale'] >= 4) & (df_niche_smile_pre['Involution'] >= 2) & (
                df_niche_smile_pre['Longtail'] >= 2) & (df_niche_smile_pre['Emerging'] >= 2),
        (df_niche_smile_pre['Scale'] >= 3) & (df_niche_smile_pre['Involution'] >= 2.5) & (
                df_niche_smile_pre['Longtail'] >= 2.5) & (df_niche_smile_pre['Emerging'] >= 2),
        (df_niche_smile_pre['Scale'] >= 0) & (df_niche_smile_pre['Involution'] >= 2) & (
                df_niche_smile_pre['Longtail'] >= 2) & (df_niche_smile_pre['Emerging'] >= 2) | (
                df_niche_smile_pre['SMILE打分'] >= 11.5)]
    labels_h = ["SA/B/C", "B/C", "C"]
    df_niche_smile_pre['业务线'] = np.select(conditions_h, labels_h, default="-")

    # 获取中文名列
    df_niche_smile_pre = df_niche_smile_pre.merge(df_translate, how='left', on='利基站点')

    # 整理到表
    df_niche_h = df_niche_smile_pre[
        ['利基站点', 'niche_id', 'niche_title_x', 'mkid_x', '是否探索', '中文名', '业务线', 'SMILE打分', 'Scale', '平均价格', '价格集中度',
         'search_volume_t_360_x', '月均GMV', 'TOP5平均月销额', '总售出件数_360_x', 'search_volume_T90_x', '总售出件数_90_x',
         'search_volume_t_7_x', 'Monetary', '平均每单件数_x', '体积超重占比', '重量', '体积重', '头程', '实抛偏差率', '规格分布', 'FBA', '头程占比',
         'FBA占比', '货值占比', 'FBA货值比', '资金利用效率AVG', '营销前毛利反推', '实抛分布', '重量分布', 'Involution', '总点击量', '转化净值', '转化净值偏差',
         '转化净值分级', '搜索点击比', '毛估蓝海度', 'CR', 'CR_KW_x', 'f(SR,AR)', 'f(VR)', 'f(NS,SP)', 'f(CC)', 'bf(CC)和f(SR,AR)',
         '毛估CPC因子', 'CPC', 'CPC因子', '广告蓝海度', '蓝海度差异分', '广告权重', '综合蓝海度', '搜索广告占比档',
         'sponsored_products_percentage_t360_count_qoq_x', 'sponsored_products_percentage_t360_count_yoy_x', '平均缺货率',
         'avg_oosrate_t360_count_qoq_x', 'avg_oosrate_t360_count_yoy_x', 'top5_brands_click_share_t360_count_yoy_x',
         'ASIN平均上架年数', 'avg_review_rating_currentvalue_x', 'avg_detail_page_quality_currentvalue_x', '周SCR_AVG',
         '季节性标签', 'Longtail', '长尾指数', 'product_count_x', '品牌长尾指数', 'brand_count_t360_count_qoq_x',
         'brand_count_t360_count_yoy_x', '非TOP5单品月销量', '平均单品月销量', '非TOP5平均月销额', '利基平均月销售额',
         'top5_products_click_share_t360_count_qoq_x', 'top5_products_click_share_t360_count_yoy_x',
         'top5_brands_click_share_t360_count_qoq_x', '知名品牌依赖', 'Emerging', 'search_volume_growth_t_360_x',
         'search_volume_growth_t_90_x', '趋势覆盖周数', '留评意愿强度', 'avg_review_count_currentvalue_x',
         'sales_potential_score_x', '近半年新品成功率', 'new_products_launched_t180_count_qoq_x',
         'new_products_launched_t180_count_yoy_x', 'successful_launches_t180_count_qoq_x',
         'successful_launches_t180_count_yoy_x', 'ASIN1', 'ASIN1品牌', 'ASIN1完整类名', 'ASIN2', 'ASIN2品牌', 'ASIN2完整类名',
         'ASIN3', 'ASIN3品牌', 'ASIN3完整类名', '知名品牌', 'asin_image_url', '类目路径', '一级类目', 'dataset_date_x']]

    df_niche_h = df_clear(df_niche_h, '利基站点')

    df_niche_h.rename(
        columns={'niche_title_x': "所属利基",
                 'mkid_x': "站点",
                 'search_volume_t_360_x': "搜索量_360",
                 '总售出件数_360_x': "总售出件数_360",
                 'search_volume_T90_x': "搜索量_90",
                 '总售出件数_90_x': "总售出件数_90",
                 'search_volume_t_7_x': "最新周搜索量",
                 '平均每单件数_x': "平均每单件数",
                 '重量': "加权平均重量",
                 '体积重': "加权平均体积重",
                 'FBA': "加权平均FBA",
                 'CR_KW_x': "CR_KW",
                 'sponsored_products_percentage_t360_count_qoq_x': "搜索广告商品占比_90",
                 'sponsored_products_percentage_t360_count_yoy_x': "搜索广告商品占比_360",
                 'avg_oosrate_t360_count_qoq_x': "平均缺货率_90",
                 'avg_oosrate_t360_count_yoy_x': "平均缺货率_360",
                 'top5_brands_click_share_t360_count_yoy_x': "TOP5品牌点击占比_360",
                 'avg_review_rating_currentvalue_x': '平均产品星级',
                 'avg_detail_page_quality_currentvalue_x': "商品listing平均得分_90",
                 'product_count_x': "点击最多的商品数",
                 'brand_count_t360_count_qoq_x': "品牌数量_90",
                 'brand_count_t360_count_yoy_x': "品牌数量_360",
                 'top5_products_click_share_t360_count_qoq_x': "TOP5产品点击占比_90",
                 'top5_products_click_share_t360_count_yoy_x': "TOP5产品点击占比_360",
                 'top5_brands_click_share_t360_count_qoq_x': "TOP5品牌点击占比_90",
                 'search_volume_growth_t_360_x': "搜索量增长_360",
                 'search_volume_growth_t_90_x': "搜索量增长_90",
                 '趋势覆盖周数': "数据趋势覆盖周数",
                 'avg_review_count_currentvalue_x': '平均评论数',
                 'sales_potential_score_x': "销量潜力得分",
                 'new_products_launched_t180_count_qoq_x': "近180天上架新品数_90",
                 'new_products_launched_t180_count_yoy_x': "近180天上架新品数_360",
                 'successful_launches_t180_count_qoq_x': "近180天成功上架新品数_90",
                 'successful_launches_t180_count_yoy_x': "近180天成功上架新品数_360",
                 'asin_image_url': "利基图片URL",
                 'dataset_date_x': "数据更新时间"}, inplace=True)

    # 4.导入目标库
    # 存入niche_original表
    oe_save_to_sql(df_niche_price, path.niche_price_original, "append", config.connet_niche_db_sql)
    oe_save_to_sql(df_niche_asin, path.niche_asin_original, "append", config.connet_niche_db_sql)
    oe_save_to_sql(df_niche_trends, path.niche_trends_original, "append", config.connet_niche_db_sql)
    # 存入niche表
    oe_save_to_sql(df_niche_asin, path.ASINs_sql_name, "append", config.connet_betterin_db_sql)
    oe_save_to_sql(df_niche_price, path.niche_price, "append", config.connet_betterin_db_sql)
    oe_save_to_sql(df_niche_trends, path.Trends_sql_name, "append", config.connet_betterin_db_sql)
    oe_save_to_sql(df_niche_keywords, path.Keyword_sql_name, "append", config.connet_betterin_db_sql)
    oe_save_to_sql(df_niche_h, path.niche_smile_h, "append", config.connet_betterin_db_sql)

    row_start = row_start + row_increment
    print("row_start：" + row_start.__str__())
    print("用时：" + (time.time() - start_time).__str__())

connect_mysql_betterin(niche_sql.clear_smile_sql)
connect_mysql_betterin(niche_sql.insert_smile_sql)
connect_mysql_betterin(niche_sql.update_smile_url_sql)
connect_mysql_betterin(niche_sql.clear_smile_tag_sql)
connect_mysql_betterin(niche_sql.create_smile_tag_sql)
print("用时：" + (time.time() - start_time).__str__())
