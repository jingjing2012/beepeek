from datetime import datetime
import time
import numpy as np
import pandas as pd
import pymysql

from conn import sql_engine, mysql_config as config
import pt_product_report_parameter as parameter
import pt_product_report_path as path


# 连接数据并清空表格
def connect_product(database, sql):
    conn_oe = pymysql.connect(host=config.oe_hostname, user=config.oe_username, passwd=config.oe_password,
                              database=database, charset='utf8')
    cur = conn_oe.cursor()
    cur.execute(sql)
    conn_oe.commit()


# 数据连接
def connect_pt_product(database, sql):
    conn_oe = pymysql.connect(host=config.oe_hostname, user=config.oe_username, passwd=config.oe_password,
                              database=database, charset='utf8')
    df = pd.read_sql(sql, conn_oe)
    return df


# 数据类型修正
def convert_type(df, con_str, d):
    df[con_str] = df[con_str].replace('', np.nan)
    df[con_str] = df[con_str].astype('float64').round(decimals=d)
    return df[con_str]


# 字符串类型修正
def convert_str(df, con_str):
    df[con_str] = df[con_str].astype(str)
    df[con_str] = df[con_str].str.lower()
    df[con_str] = df[con_str].str.strip()
    return df[con_str]


# 日期修正
def convert_date(df, df_str):
    df[df_str] = pd.to_datetime(df[df_str], errors='coerce', format='%Y-%m-%d')
    return df[df_str]


# mround函数实现
def df_mround(df, df_str, df_str_mround, mround_n):
    df[df_str_mround] = round(df[df_str] / mround_n) * mround_n
    return df[df_str_mround]


# 百分比转换
def percent_convert(df, df_str):
    df[df_str] = pd.to_numeric(df[df_str].str.rstrip('%'), errors='coerce') / 100
    return convert_type(df, df_str, 4)


# 数据打标签
def df_cut(df, col, col_cut, bins_cut, labels_cut):
    df[col_cut] = pd.cut(df[col], bins_cut, labels=labels_cut, include_lowest=True)
    return df[col_cut]


# series转DataFrame
def convert_col(series, col):
    df = pd.DataFrame(series)
    df.columns = [col]
    return df


# 直发FBM可能性
def fbm(df):
    df_fbm = df[df['buybox_location'].notnull()]
    df_fbm['按毛利推测FBM可能性'] = np.where(df_fbm['gross_margin'] >= parameter.gross_margin_upper, 2, 1)
    df_fbm['中国卖家FBM可能性'] = np.where(df_fbm['buybox_location'] == "CN", df_fbm['按毛利推测FBM可能性'], 0)

    conditions_fbm_1 = (df_fbm['seller_type'] == "FBM") & (df_fbm['buybox_location'] != "US") & (
            df_fbm['buybox_location'] != "") & (df_fbm['gross_margin'] >= parameter.gross_margin_lower)
    conditions_fbm_2 = (df_fbm['fba_fees'] > 0) | (df_traffic['重量(g)'] <= 2500)
    conditions_fbm_3 = (df_fbm['fba_fees'] <= parameter.fba_fees_upper) | (
            df_fbm['gross_margin'] >= parameter.gross_margin_upper)
    df_fbm['直发FBM可能性'] = np.where(conditions_fbm_1 & conditions_fbm_2 & conditions_fbm_3, 1 + df_fbm['中国卖家FBM可能性'], 0)
    df_fbm = df_fbm[['id', '直发FBM可能性']]
    return df_fbm


def match_holidays(row):
    matched_holidays = [keyword for keyword in df_holiday['节日关键词'] if keyword in row['combined_kw']]
    holidays_count = len(matched_holidays)
    holidays_str = ", ".join(matched_holidays) if matched_holidays else ""
    return holidays_count, holidays_str


def match_custom_kw(row):
    match_custom_kw = [keyword for keyword in custom_kw if keyword in row['title']]
    custom_kw_count = len(match_custom_kw)
    return custom_kw_count


# 开售月数计算
def month_available(df):
    current_date = datetime.now().date()
    df['date_available'] = pd.to_datetime(df['date_available'], errors='coerce').dt.date
    df['开售天数'] = (current_date - df['date_available']).dt.days
    df['头程月数'] = np.where((df['seller_type'] == "FBA") & (df['开售天数'] > 15), 0.5, 0)
    df['开售月数'] = np.fmax(round(df['开售天数'] / 30 - df['头程月数'], 1), 0.1)
    return df


# 销额级数计算
def revenue(df):
    df['monthly_revenue_increase'] = df['monthly_revenue_increase'].fillna(0)
    df['近两月销额'] = np.where(df['monthly_revenue_increase'] <= (-1), np.nan,
                           df['monthly_revenue'] + (df['monthly_revenue'] / (1 + df['monthly_revenue_increase'])))
    df['月均销额'] = np.where(df['近两月销额'] * 1 > 0, df['近两月销额'] / np.fmax(np.fmin(df['开售月数'] - 1, 1), 0.5), np.nan)
    df['销额级数'] = np.where(df['月均销额'] * 1 > 0, np.log2(df['月均销额'] / 2 / (parameter.monthly_revenue_C / 2)), np.nan)
    return df


# 排序
def sort_and_rank(df):
    df['ac_tag'] = np.where(df['ac'] == "Y", 1, 0)
    df = df.groupby('parent', group_keys=False).apply(
        lambda x: x.sort_values(by=['update_time', '开售月数', 'ac_tag'], ascending=[False, False, False]))
    df['row_rank'] = df.reset_index(drop=True).index
    df['rank'] = df['row_rank'].groupby(df['parent']).rank()
    return df['rank']


# 综合竞品推荐度
def recommend_avg(df):
    row_relevance = np.array(df['relevance'])
    row_recommend = np.array(df['推荐度'])
    recommend_avg = sum(row_relevance * row_recommend) / sum(row_relevance)
    return recommend_avg


def product_count(df, col_str):
    asins_count = df['asin'].groupby(df['related_asin']).count()
    return convert_col(asins_count, col_str)


def df_clear(df, clear_id):
    df.replace(to_replace=[None], value='', inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.drop_duplicates(subset=[clear_id])
    df = df.dropna(subset=[clear_id])
    return df


# 导入数据库
def save_to_sql(df, table, conn, args):
    df.to_sql(table, conn, if_exists=args, index=False)


# -------------------------------------------------------------------------------------------------------------------


start_time = time.time()

clear_sql_product_main_temporary = "TRUNCATE TABLE " + path.product_main_temporary

create_sql_pt_relation_main = "CREATE TABLE `pt_relation_main` (" \
                              "`id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键'," \
                              "`parent` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci " \
                              "DEFAULT NULL COMMENT '父体'," \
                              "`variations` int DEFAULT NULL COMMENT '变体数'," \
                              "`asin_count` int DEFAULT NULL COMMENT 'ASIN数'," \
                              "`asin` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci " \
                              "DEFAULT NULL COMMENT '主推款ASIN'," \
                              "`traffic_id` bigint DEFAULT NULL COMMENT '主推款ASIN关联id'," \
                              "EFAULT NULL COMMENT '关联ASIN'," \
                              "PRIMARY KEY (`id`) USING BTREE," \
                              "KEY `id_index` (`id`)," \
                              "KEY `traffic_id_index` (`traffic_id`)," \
                              "KEY `asin_index` (`asin`)," \
                              "KEY `parent_index` (`parent`)" \
                              ") ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci " \
                              "ROW_FORMAT=DYNAMIC COMMENT='主推款关联表'"

insert_sql_pt_relation_main = "INSERT INTO `pt_relation_main`(`parent`, variations, asin_count)" \
                              "(SELECT * FROM(SELECT parent,AVG(variations) as variations,count(*) as asin_count " \
                              "FROM pt_relation_traffic GROUP BY parent) traffic_parent ORDER BY " + \
                              "traffic_parent.asin_count DESC)"

update_sql_pt_relation_main = "UPDATE " + config.pt_product_database + "." + path.pt_relation_main + " INNER JOIN " + \
                              path.product_database + "." + path.product_main_temporary + " ON " + \
                              config.pt_product_database + "." + path.pt_relation_main + ".parent = " + \
                              path.product_database + "." + path.product_main_temporary + ".parent SET " + \
                              config.pt_product_database + "." + path.pt_relation_main + ".asin = " + \
                              path.product_database + "." + path.product_main_temporary + ".asin," + \
                              config.pt_product_database + "." + path.pt_relation_main + ".traffic_id = " + \
                              path.product_database + "." + path.product_main_temporary + ".relation_traffic_id"

# connect_product(config.pt_product_database, create_sql_pt_relation_main)
# connect_product(config.pt_product_database, insert_sql_pt_relation_main)

# 循环参数
row_start = 814500
row_increment = 1000
row_max_main = 10000000
row_max = 1100000

# 主推款确定
while row_start < row_max_main:
    # 1.数据连接
    sql_parent = "select * from " + path.pt_relation_main + " where asin_count>1 and id between " + str(
        row_start) + " and " + str(row_start + row_increment)
    sql_related_traffic = 'SELECT ' + path.pt_relation_traffic + '.* FROM (' + sql_parent + ') pt_parent LEFT JOIN ' + \
                          path.pt_relation_traffic + ' ON pt_parent.parent=' + path.pt_relation_traffic + '.parent'
    sql_traffic_parent = 'SELECT pt_traffic_parent.*,' + path.pt_relevance_asins + '.relevance FROM (' + \
                         sql_related_traffic + ') pt_traffic_parent LEFT JOIN ' + path.pt_relevance_asins + \
                         ' ON pt_traffic_parent.id=' + path.pt_relevance_asins + '.relation_traffic_id'

    df_traffic_parent = connect_pt_product(config.pt_product_database, sql_traffic_parent)

    connect_product(path.product_database, clear_sql_product_main_temporary)

    if df_traffic_parent.empty:
        row_start = row_start + row_increment
        continue

    # 2.获取主推变体款
    df_traffic_parent = df_traffic_parent[df_traffic_parent['relevance'] >= 0.5]

    # 2.1开售月数
    month_available(df_traffic_parent)

    # 2.2排序
    # 销额处理
    df_traffic_parent['monthly_revenue'].fillna(0, inplace=True)
    df_traffic_avg = df_traffic_parent['monthly_revenue'].groupby(df_traffic_parent['parent']).mean()
    df_traffic_avg = convert_col(df_traffic_avg, 'monthly_revenue_avg')
    df_traffic_parent = df_traffic_parent.merge(df_traffic_avg, how='left', on='parent')
    # 竞品销额均为空的处理
    df_related_parent_0 = df_traffic_parent.query('monthly_revenue_avg==0')
    df_related_parent_0['rank'] = sort_and_rank(df_related_parent_0)
    df_main_0 = df_related_parent_0.query('rank==1')
    # 竞品销额含非空的处理
    df_related_parent_1 = df_traffic_parent.query('monthly_revenue_avg>0')
    df_related_parent_1 = df_related_parent_1.query('monthly_revenue>0')
    df_related_parent_1['rank'] = sort_and_rank(df_related_parent_1)
    df_main_1 = df_related_parent_1.query('rank==1')
    df_main = pd.concat([df_main_0, df_main_1])

    # 3.字段整合
    df_main = df_main[['parent', 'asin', 'id']]
    df_clear(df_main, 'parent')
    df_main.rename(columns={'id': 'relation_traffic_id'}, inplace=True)

    # 4.存入数据库
    pt_product_conn = sql_engine.create_conn(config.connet_product_db_sql)
    save_to_sql(df_main, path.product_main_temporary, pt_product_conn, 'append')

    pt_product_cpc_conn = sql_engine.create_conn(config.connet_product_pt_db_sql)
    connect_product(config.pt_product_database, update_sql_pt_relation_main)

    row_start = row_start + row_increment
    print("row_start：" + row_start.__str__())
    print("用时：" + (time.time() - start_time).__str__())

print("用时：" + (time.time() - start_time).__str__())

"""
# 关联ASIN聚合
while row_start < row_max:
    # 1.数据连接
    sql_asin = "select * from " + path.pt_product_get_group + " where id between " + str(row_start) + " and " + str(
        row_start + row_increment)
    sql_relevance = 'SELECT pt_product.*,count(distinct ' + path.pt_relevance_asins + \
                    '.relation_traffic_id) as "相关竞品款数" FROM (' + sql_asin + ') pt_product LEFT JOIN ' + \
                    path.pt_relevance_asins + ' ON pt_product.asin = ' + path.pt_relevance_asins + \
                    '.asin GROUP BY pt_product.id'
    sql_parent = 'SELECT ' + path.pt_relevance_asins + '.* FROM (' + sql_asin + ') pt_product LEFT JOIN ' + \
                 path.pt_relevance_asins + ' ON pt_product.asin = ' + path.pt_relevance_asins + '.asin WHERE ' + \
                 path.pt_relevance_asins + '.relevance is not null'
    sql_related = 'SELECT pt_parent.asin as asin_index,pt_parent.relevance,' + path.pt_relation_main + '.* FROM (' + \
                  sql_parent + ') pt_parent LEFT JOIN ' + path.pt_relation_main + ' ON pt_parent.relation_traffic_id = ' \
                  + path.pt_relation_main + '.traffic_id WHERE ' + path.pt_relation_main + '.id is not null'
    sql_traffic = 'SELECT ' + path.pt_relation_traffic + '.*,SUBSTRING_INDEX(category_path,":",2) as "二级类目",' \
                  + 'pt_relation.related_asin,pt_relation.relevance FROM( ' + sql_related + ' ) pt_relation LEFT JOIN ' \
                  + path.pt_relation_traffic + ' ON pt_relation.traffic_id = ' + path.pt_relation_traffic + '.id'

    holiday_sql = 'select 节日关键词 from ' + path.product_database + '.' + path.product_holiday
    famous_brand_sql = 'select brand,预估影响力 as "疑似知名品牌" from ' + path.product_database + '.' + path.product_famous_brand

    df_parent = connect_pt_product(config.pt_product_database, sql_parent)
    df_related = connect_pt_product(config.pt_product_database, sql_related)
    df_relevance = connect_pt_product(config.pt_product_database, sql_relevance)
    df_traffic = connect_pt_product(config.pt_product_database, sql_traffic)

    df_famous_brand = connect_pt_product(path.product_database, famous_brand_sql)
    df_holiday = connect_pt_product(path.product_database, holiday_sql)

    if df_traffic.empty:
        row_start = row_start + row_increment
        continue

    # 2.数据预处理
    conditions_traffic = (df_traffic['date_available'].notnull()) & (df_traffic['id'].notnull()) & (
        df_traffic['monthly_revenue'].notnull()) & (df_traffic['relevance'] >= 0.5)

    df_traffic = df_traffic[conditions_traffic]

    df_traffic['ASIN'] = df_traffic['asin']
    df_clear(df_traffic, 'ASIN')
    product_con_list_1 = ['category_bsr_growth', 'sales_growth', 'price', 'gross_margin', 'fba_fees']
    for con_i in product_con_list_1:
        convert_type(df_traffic, con_i, 2)

    product_con_list_2 = ['sales', 'qa', 'ratings', 'variations']
    for con_j in product_con_list_2:
        convert_type(df_traffic, con_j, 0)

    convert_type(df_traffic, 'reviews_rate', 4)
    convert_type(df_traffic, 'rating', 1)

    product_con_list_3 = ['brand', 'title', 'category_path', 'category', 'sub_category', 'ac_keyword', 'weight', '二级类目']
    for con_l in product_con_list_3:
        convert_str(df_traffic, con_l)

    percent_convert(df_traffic, 'monthly_revenue_increase')
    convert_date(df_traffic, 'date_available')

    convert_str(df_famous_brand, 'brand')
    convert_type(df_famous_brand, '疑似知名品牌', 0)
    convert_str(df_holiday, '节日关键词')

    # 3.M相关指标计算
    for error_unit, replacement in parameter.replace_error_dict.items():
        df_traffic['weight'] = df_traffic['weight'].str.replace(error_unit, replacement)

    df_traffic['重量值'] = df_traffic['weight'].str.split(" ", expand=True)[0]
    df_traffic['单位'] = df_traffic['weight'].str.split(" ", expand=True)[1]

    df_traffic['单位'] = pd.Categorical(df_traffic['单位'])

    df_traffic['换算'] = df_traffic['单位'].replace(parameter.replace_dict)
    try:
        df_traffic['换算'] = df_traffic['换算'].astype('float64')
    except ValueError as e:
        print("Error:", df_traffic['换算'])
        print("Error:", e)

    convert_type(df_traffic, '重量值', 4)
    df_traffic['重量(g)'] = round(df_traffic['重量值'] * df_traffic['换算'], 4)

    # 直发FBM可能性
    df_traffic_fbm = fbm(df_traffic)

    df_traffic = df_traffic.merge(df_traffic_fbm, how='left', on='id')
    df_traffic_fbm['直发FBM可能性'].fillna(0)

    df_traffic['预估FBA占比'] = np.where(df_traffic['fba_fees'] > 0, df_traffic['fba_fees'] / df_traffic['price'],
                                     parameter.fba_fees_rate)
    df_traffic['预估头程占比'] = np.where(df_traffic['预估FBA占比'] > 0, df_traffic['预估FBA占比'] / 2.5, parameter.pre_fees_rate)
    df_traffic['预估货值占比'] = df_cut(df_traffic, 'price', '预估货值占比', [0, 6, 10, 15, 30, 50, 100, 200, 9999],
                                  [0.08, 0.1, 0.15, 0.2, 0.25, 0.27, 0.3, 0.35])
    convert_type(df_traffic, '预估货值占比', 2)

    df_traffic['预估毛利率_FBM'] = df_traffic['gross_margin'] - df_traffic['预估头程占比'] * 2 - parameter.product_fees_rate
    df_traffic['预估毛利率_FBA'] = df_traffic['gross_margin'] - df_traffic['预估头程占比'] - parameter.product_fees_rate
    df_traffic['预估毛利率_反推'] = np.where(
        (df_traffic['直发FBM可能性'] >= 1) & (df_traffic['gross_margin'] >= parameter.gross_margin_upper),
        df_traffic['预估毛利率_FBM'], df_traffic['预估毛利率_FBA'])

    df_traffic['预估毛利率'] = np.where(abs(df_traffic['gross_margin']) > 0, df_traffic['预估毛利率_反推'],
                                   1 - df_traffic['预估FBA占比'] - df_traffic['预估头程占比'] - parameter.referral_fees_rate -
                                   df_traffic['预估货值占比'])
    df_traffic['毛利率级别_上限'] = np.fmin(df_mround(df_traffic, '预估毛利率', '毛利率级别_上限', 0.05),
                                     parameter.gross_rate_upper)
    df_traffic['毛利率级别_下限'] = np.fmax(df_mround(df_traffic, '预估毛利率', '毛利率级别_下限', -0.05),
                                     parameter.gross_rate_lower)
    df_traffic['毛利率级别'] = np.where(df_traffic['预估毛利率'] >= 0, df_traffic['毛利率级别_上限'], df_traffic['毛利率级别_下限'])

    df_traffic['毛估资金利用率'] = df_traffic['预估毛利率'] / (df_traffic['预估头程占比'] + parameter.product_fees_rate)

    # 4.推荐度相关指标计算
    # M相关指标
    df_traffic['高资金利用率'] = np.where(abs(df_traffic['毛估资金利用率']) > 0,
                                    df_traffic['毛估资金利用率'] / parameter.product_revenue_std - 1,
                                    0)

    # I相关指标
    month_available(df_traffic)

    # S相关指标
    revenue(df_traffic)

    conditions_lqs_1 = (df_traffic['预估毛利率'] >= -0.05) & (df_traffic['lqs'] > 0) & (df_traffic['lqs'] <= 8)
    conditions_lqs_2 = (df_traffic['开售月数'] >= 24) & (df_traffic['rating'] >= 4) & (df_traffic['ratings'] >= 10) & (
            df_traffic['预估毛利率'] >= -0.15) & (df_traffic['lqs'] > 0) & (df_traffic['lqs'] <= 8)
    df_traffic['高销低LQS_pre'] = np.fmin(5, 0.5 + df_traffic['销额级数'] * parameter.lqs_std / df_traffic['lqs'])
    df_traffic['高销低LQS'] = np.where(conditions_lqs_1 | conditions_lqs_2, df_traffic['高销低LQS_pre'], 0)

    df_traffic['开售月数_QA'] = np.fmin(df_traffic['开售月数'], 24)
    df_traffic['月均QA数'] = np.where(df_traffic['qa'] > 0, round(df_traffic['qa'] / df_traffic['开售月数_QA'], 1), 0)

    conditions_available = (df_traffic['开售月数'] >= parameter.available_std) & (
            df_traffic['monthly_revenue'] >= parameter.monthly_revenue_C)

    df_traffic['长期上架少Q&A'] = np.where(conditions_available,
                                      np.fmax(-1, parameter.qa_std - df_traffic['月均QA数'] / parameter.qa_std), 0)

    df_traffic['长期上架无A+'] = np.where(conditions_available & (df_traffic['ebc_available'] != "Y"), 1, 0)

    df_traffic['长期上架无视频'] = np.where(conditions_available & (df_traffic['video_available'] != "Y"), 1, 0)

    df_traffic['类轻小直发FBM'] = np.where(df_traffic['直发FBM可能性'] > 0,
                                      np.fmax(0, 1 + df_traffic['销额级数'] * np.fmin(1, df_traffic['直发FBM可能性'] / 2)), 0)

    df_traffic['差评好卖'] = np.where((df_traffic['开售月数'] >= parameter.available_std) & (
            df_traffic['monthly_revenue'] >= parameter.monthly_revenue_C / 2) & (df_traffic['ratings'] >= 10) & (
                                          df_traffic['rating'] >= 3) & (df_traffic['rating'] < 4) & (
                                          abs(df_traffic['预估毛利率']) > 0) & (df_traffic['category_bsr_growth'] >= -0.5),
                                  0.5 + df_traffic['销额级数'] * (4.5 - df_traffic['rating']), 0)

    df_traffic = df_traffic.merge(df_famous_brand, how='left', on='brand')

    df_traffic['combined_kw'] = df_traffic['title'] + "" + df_traffic['sub_category'] + "" + df_traffic['ac_keyword']

    df_traffic[['疑似节日性', '节日名']] = df_traffic.apply(lambda row: pd.Series(match_holidays(row)), axis=1)

    df_traffic['知名品牌'] = np.where(df_traffic['疑似知名品牌'] > 0,
                                  -df_traffic['疑似知名品牌'] / np.where(df_traffic['疑似节日性'] > 0, 2, 1),
                                  0)
    df_traffic['知名品牌'].fillna(0)

    df_traffic['疑似节日性'] = np.where(df_traffic['疑似节日性'] > 3, "3+", df_traffic['疑似节日性'])

    custom_kw = ['custom', 'personalize', 'personalized', 'custom-made', 'customized', 'made-to-order']
    df_traffic['custom_kw'] = df_traffic.apply(lambda row: pd.Series(match_custom_kw(row)), axis=1)
    df_traffic['是否个人定制'] = np.where(df_traffic['custom_kw'] > 0, 1, 0)

    # L相关指标
    df_traffic['平均变体月销额等级'] = np.where(df_traffic['monthly_revenue'] > 0, np.log2(
        df_traffic['monthly_revenue'] / df_traffic['variations'] / (parameter.monthly_revenue_C / 2)), 0)
    df_traffic['变体等级'] = np.log2(df_traffic['variations'])
    df_traffic['少变体'] = np.where(df_traffic['variations'] == 1, 0,
                                 np.fmax(-10, np.fmin(0, df_traffic['平均变体月销额等级'] - 0.5 * df_traffic['变体等级'] + 0.5)))

    # E相关指标
    conditions_new_product = df_traffic['开售月数'] < parameter.available_std

    df_traffic['新品爬坡快'] = np.where(conditions_new_product & (df_traffic['monthly_revenue'] >= 0), np.log2(
        df_traffic['monthly_revenue'] / df_traffic['开售月数'] / (parameter.monthly_revenue_C / parameter.revenue_month_C)),
                                   0)

    df_traffic['新品增评好'] = np.where(conditions_new_product & (df_traffic['rating'] >= 4) & (df_traffic['ratings'] >= 10),
                                   np.fmax(0, np.fmin(2, np.log(
                                       df_traffic['ratings'] / df_traffic['开售月数'] / df_traffic['variations'] / np.log(
                                           5)))), 0)

    df_traffic['新品NSR'] = np.where(conditions_new_product & (df_traffic['new_release'] == "Y"), 1, 0)

    df_traffic['新品AC标'] = np.where(conditions_new_product & (df_traffic['ac'] == "Y"), 1, 0)

    # 推荐度计算
    df_recommend = df_traffic[
        ['id', '销额级数', '高资金利用率', '高销低LQS', '长期上架少Q&A', '长期上架无A+', '长期上架无视频', '类轻小直发FBM', '差评好卖', '知名品牌', '少变体', '新品爬坡快',
         '新品增评好', '新品NSR', '新品AC标']]
    recommend_weights = np.array([0, 0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 1.5, 1, 1, 1, 0.5, 0.5, 0.5, 0.5])

    df_recommend = df_recommend.astype(float)
    recommend_weights = recommend_weights.astype(float)

    df_recommend['推荐度'] = df_recommend.dot(recommend_weights)
    df_recommend['推荐度'] = df_recommend['推荐度'].fillna(0)
    convert_type(df_recommend, '推荐度', 1)

    df_recommend = df_recommend[['id', '推荐度']]
    df_traffic = df_traffic.merge(df_recommend, how='left', on='id')

    # 数据更新日期
    df_traffic['数据更新时间'] = str(config.pt_product_database)[-6:] + "01"
    convert_date(df_traffic, '数据更新时间')

    # 类目清洗
    df_traffic['category'] = df_traffic['category'].replace(parameter.replace_category_dict)
    df_traffic['二级类目'] = df_traffic['二级类目'].replace(parameter.replace_category_dict_2)

    # 剔除类目
    df_traffic.loc[
        df_traffic['category_path'].str.contains(parameter.regex_pattern_kw, na=False, regex=True), '剔除类目'] = 1
    df_traffic['剔除类目'] = np.where(df_traffic['剔除类目'] == 1, 1, 0)

    # 数据格式整理
    product_con_list = ['预估FBA占比', '预估头程占比', '预估毛利率', '毛估资金利用率', '销额级数', '高资金利用率', '高销低LQS', '类轻小直发FBM', '平均变体月销额等级',
                        '新品爬坡快']
    for con_k in product_con_list:
        convert_type(df_traffic, con_k, 4)

    # 5.添加数据标签
    df_traffic_tag = df_traffic[
        ['ASIN', 'price', 'category', 'seller_type', '直发FBM可能性', '预估FBA占比', '预估毛利率', '毛估资金利用率', '开售月数', '月均QA数',
         '疑似知名品牌', '疑似节日性', '节日名', '是否个人定制', '推荐度', '销额级数', 'lqs', 'qa', 'ebc_available', 'video_available', 'rating',
         '知名品牌', 'variations', 'ratings', 'new_release', 'ac', '二级类目', '剔除类目', '数据更新时间']]

    df_traffic_tag['data_id'] = df_traffic_tag['ASIN'] + " | " + str(df_traffic_tag['数据更新时间'])
    df_traffic_tag['价格分布'] = round(df_traffic_tag['price'])
    df_traffic_tag['预估FBA占比分布'] = df_mround(df_traffic_tag, '预估FBA占比', '预估FBA占比分布', 0.05)
    df_traffic_tag['预估毛利率分布'] = df_mround(df_traffic_tag, '预估毛利率', '预估毛利率分布', 0.05)
    df_traffic_tag['毛估资金利用率分布'] = df_mround(df_traffic_tag, '毛估资金利用率', '毛估资金利用率分布', 0.05)

    df_traffic_tag['开售月数分布'] = df_mround(df_traffic_tag, '开售月数', '开售月数分布', 3)
    df_traffic_tag['推荐度分布'] = df_mround(df_traffic_tag, '推荐度', '推荐度分布', 0.5)
    df_traffic_tag['销额级数分布'] = df_mround(df_traffic_tag, '销额级数', '销额级数分布', 0.1)
    df_traffic_tag['评分数分布'] = df_mround(df_traffic_tag, 'ratings', '评分数分布', 100)

    # 6.聚合字段计算
    group_traffic_list = list(df_traffic.groupby(df_traffic['related_asin'], as_index=False))

    traffic_group_frame = []
    traffic_recommend_frame = []

    for tuple_df in group_traffic_list:
        traffic_tuple_df: DataFrame = tuple_df[1].reset_index()

        # 综合竞品推荐度
        traffic_tuple_df['综合竞品推荐度'] = recommend_avg(traffic_tuple_df)

        # 有销额竞品款数
        traffic_count = product_count(traffic_tuple_df, '有销额竞品款数')

        traffic_recommend = traffic_tuple_df.query('推荐度>=3')

        # 有销额推荐达标款数
        traffic_recommend_count = product_count(traffic_recommend, '有销额推荐达标款数')

        # 聚合表生成
        traffic_df = traffic_tuple_df.groupby('related_asin').agg({'综合竞品推荐度': 'first',
                                                                   # 'asin': lambda x: ','.join(x.astype(str)),
                                                                   '数据更新时间': 'first'}).reset_index()

        traffic_df = traffic_df.merge(traffic_count, how='left', on='related_asin')
        traffic_df = traffic_df.merge(traffic_recommend_count, how='left', on='related_asin')

        traffic_group_frame.append(traffic_df)
        traffic_recommend_frame.append(traffic_recommend)

    if df_traffic.empty:
        row_start = row_start + row_increment
        continue

    df_traffic_group = pd.concat(traffic_group_frame)
    df_traffic_recommend = pd.concat(traffic_recommend_frame)

    # 聚合表字段计算
    df_traffic_group = df_traffic_group.merge(df_relevance, how='left', left_on='related_asin', right_on='asin')

    df_traffic_group['达标推荐度占比'] = np.where(df_traffic_group['有销额推荐达标款数'] * 1 > 0,
                                           df_traffic_group['有销额推荐达标款数'] / df_traffic_group['有销额竞品款数'], np.nan)

    condition_group_pre = df_traffic_group['综合竞品推荐度'] * df_traffic_group['达标推荐度占比'] > 0
    condition_group = [
        (df_traffic_group['综合竞品推荐度'] >= 5) & (df_traffic_group['达标推荐度占比'] >= 0.6) & (df_traffic_group['有销额竞品款数'] >= 5),
        (df_traffic_group['综合竞品推荐度'] >= 3) & (df_traffic_group['达标推荐度占比'] >= 0.4) & (df_traffic_group['有销额竞品款数'] >= 3),
        (df_traffic_group['综合竞品推荐度'] < 2) | (df_traffic_group['达标推荐度占比'] < 0.25)]
    labels_group = ['推荐', '轻度推荐', '不推荐']

    df_traffic_group['推荐级别+PMI'] = np.select(condition_group, labels_group, default="待定")
    df_traffic_group['推荐级别+PMI'] = np.where(condition_group_pre, df_traffic_group['推荐级别+PMI'], np.nan)

    convert_type(df_traffic_group, '综合竞品推荐度', 2)
    convert_type(df_traffic_group, '达标推荐度占比', 3)

    # 7.字段整合
    df_traffic_table = df_traffic_recommend[['related_asin',
                                             'relevance',
                                             'ASIN',
                                             'sku',
                                             'brand',
                                             'title',
                                             'image',
                                             'parent',
                                             'category_path',
                                             'category',
                                             'category_bsr',
                                             'category_bsr_growth',
                                             'sales',
                                             'monthly_revenue',
                                             'price',
                                             'qa',
                                             'gross_margin',
                                             'fba_fees',
                                             'ratings',
                                             'reviews_rate',
                                             'rating',
                                             'monthly_rating_increase',
                                             'date_available',
                                             'seller_type',
                                             'lqs',
                                             'variations',
                                             'sellers',
                                             'buybox_seller',
                                             'buybox_location',
                                             'buybox_type',
                                             'best_seller',
                                             'ac',
                                             'new_release',
                                             'ebc_available',
                                             'video_available',
                                             'ac_keyword',
                                             'weight',
                                             '重量(g)',
                                             'dimensions',
                                             'update_time',
                                             '直发FBM可能性',
                                             '预估FBA占比',
                                             '预估头程占比',
                                             '预估货值占比',
                                             '预估毛利率',
                                             '毛利率级别',
                                             '毛估资金利用率',
                                             '开售月数',
                                             '月均QA数',
                                             '疑似知名品牌',
                                             '疑似节日性',
                                             '节日名',
                                             # '疑似红海类目',
                                             '销额级数',
                                             '高资金利用率',
                                             '高销低LQS',
                                             '长期上架少Q&A',
                                             '长期上架无A+',
                                             '长期上架无视频',
                                             '类轻小直发FBM',
                                             '差评好卖',
                                             '知名品牌',
                                             '平均变体月销额等级',
                                             '变体等级',
                                             '少变体',
                                             '新品爬坡快',
                                             '新品增评好',
                                             '新品NSR',
                                             '新品AC标',
                                             '是否个人定制',
                                             '推荐度',
                                             '二级类目',
                                             '剔除类目',
                                             '数据更新时间']]

    df_traffic_table['clear_id'] = df_traffic_table['related_asin'] + df_traffic_table['ASIN']
    df_traffic_table = df_clear(df_traffic_table, 'clear_id')
    df_traffic_table = df_traffic_table.drop(['clear_id'], axis=1)

    df_traffic_table.rename(columns={'related_asin': '原ASIN',
                                     'relevance': '关联度',
                                     'sku': 'SKU',
                                     'brand': '品牌',
                                     'title': '标题',
                                     'image': '主图',
                                     'parent': '父体',
                                     'category_path': '类目路径',
                                     'category': '大类目',
                                     'category_bsr': '大类BSR',
                                     'category_bsr_growth': '大类BSR增长率',
                                     'sales': '月销量',
                                     'monthly_revenue': '月销售额',
                                     'price': '价格',
                                     'qa': 'Q&A',
                                     'gross_margin': '毛利率',
                                     'fba_fees': 'FBA运费',
                                     'ratings': '评分数',
                                     'reviews_rate': '留评率',
                                     'rating': '评分',
                                     'monthly_rating_increase': '月新增评分数',
                                     'date_available': '上架时间',
                                     'seller_type': '配送方式',
                                     'lqs': 'LQS',
                                     'variations': '变体数',
                                     'sellers': '卖家数',
                                     'buybox_seller': 'BuyBox卖家',
                                     'buybox_location': '卖家所属地',
                                     'buybox_type': 'BuyBox类型',
                                     'best_seller': 'Best Seller标识',
                                     'ac': 'Amazon s Choice',
                                     'new_release': 'New Release标识',
                                     'ebc_available': 'A+页面',
                                     'video_available': '视频介绍',
                                     'ac_keyword': 'AC关键词',
                                     'weight': '重量',
                                     'dimensions': '体积',
                                     'update_time': '引流时间'}, inplace=True)

    df_traffic_tag = df_traffic_tag[
        ['ASIN', '价格分布', 'category', 'seller_type', '直发FBM可能性', '预估FBA占比分布', '预估毛利率分布', '毛估资金利用率分布', '开售月数分布',
         '月均QA数', '疑似知名品牌', '疑似节日性', '节日名', '推荐度分布', '销额级数分布', 'lqs', 'qa', 'ebc_available', 'video_available',
         'rating', '知名品牌', 'variations', '评分数分布', 'new_release', 'ac', '是否个人定制', '剔除类目', '数据更新时间', 'data_id']]

    df_traffic_tag = df_clear(df_traffic_tag, 'ASIN')

    df_traffic_tag.rename(columns={'category': '一级类目',
                                   'seller_type': '配送方式分布',
                                   '月均QA数': '月均QA分布',
                                   'lqs': 'LQS分布',
                                   'qa': 'QA分布',
                                   'ebc_available': 'A+页面',
                                   'video_available': '视频介绍',
                                   'rating': '评分分布',
                                   'variations': '变体分布',
                                   'new_release': 'New Release标识',
                                   'ac': 'Amazon s Choice'}, inplace=True)

    df_traffic_group = df_traffic_group[
        ['ASIN', 'price', '推荐度', '相关竞品款数', '有销额竞品款数', '有销额推荐达标款数', '原ASIN推荐度', '综合竞品推荐度', '达标推荐度占比', '推荐级别+PMI']]
    df_traffic_group = df_clear(df_traffic_group)

    df_traffic_group.rename(columns={'ASIN': 'asin', '推荐度': 'recommend'}, inplace=True)

    # 8.存入数据库
    pt_product_conn = sql_engine.create_conn(config.connet_product_db_sql)

    # save_to_sql(df_traffic_table, path.product_table, pt_product_conn, 'append')
    # save_to_sql(df_traffic_tag, path.product_tag, pt_product_conn, 'append')

    # save_to_sql(df_traffic_table, path.product_table_history, pt_product_conn, 'append')
    # save_to_sql(df_traffic_tag, path.product_tag_history, pt_product_conn, 'append')

    # pt_product_cpc_conn = sql_engine.create_product_cpc_conn()
    # save_to_sql(df_traffic_cpc, path.pt_product_get_cpc, pt_product_cpc_conn, 'append')

    save_to_sql(df_traffic_recommend_modify, path.product_recommend_modify, pt_product_conn, 'append')
    connect_product(path.product_database, update_sql_product_recommend)
    connect_product(path.product_database, update_sql_product_tag)

    row_start = row_start + row_increment
    print("row_start：" + row_start.__str__())
    print("用时：" + (time.time() - start_time).__str__())

# connect_product(config.pt_product_database, update_sql_product_get_cpc)
print("用时：" + (time.time() - start_time).__str__())
"""
