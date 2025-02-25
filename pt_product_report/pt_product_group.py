import sys
import time
import numpy as np
import pandas as pd
import warnings
from pandas.errors import PerformanceWarning, SettingWithCopyWarning

from conn import sql_engine, mysql_config as config
import pt_product_report_parameter as para
import pt_product_report_path as path
import pt_product_sql as sql
from util import duplicate_util, data_cleaning_util, calculate_util, calculation_util, common_util, db_util

# -------------------------------------------------------------------------------------------------------------------

# 忽略与 Pandas SettingWithCopyWarning 模块相关的警告
warnings.filterwarnings('ignore', category=SettingWithCopyWarning)

# 忽略与 Pandas SQL 模块相关的警告
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas.io.sql")

# 忽略除以零的警告
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in log2")

warnings.filterwarnings("ignore", message="pandas only support SQLAlchemy connectable.*")
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented.", category=PerformanceWarning)

# 设置选项以选择未来行为
pd.set_option('future.no_silent_downcasting', True)

"""
sql_engine.connect_product(config.sellersprite_hostname, config.sellersprite_password, config.sellersprite_database,
                sql.create_index_sql_supplement)
sql_engine.connect_product(config.sellersprite_hostname, config.sellersprite_password, config.sellersprite_database,
                sql.duplicate_sql_supplement)

# 总数据量
total_rows = 800000
# 每页查询的数据量
page_size = 3000
# 需要查询的总页数
total_pages = total_rows // page_size + 1
"""

df_relevance_asins = sql_engine.connect_pt_product(config.sellersprite_hostname, config.sellersprite_password,
                                                   config.sellersprite_database, sql.sql_relevance_asins)
df_group = sql_engine.connect_pt_product(config.oe_hostname, config.oe_password, config.product_database,
                                         sql.sampling_group_sql)

if df_group.empty and (not df_relevance_asins.empty):

    start_time = time.time()

    # 循环参数
    id_start = 0
    id_increment = 3000
    id_end = 800000

    # 更新日期
    update_date = str(config.sellersprite_database)[-6:-2] + "-" + str(config.sellersprite_database)[-2:] + "-01"

    df_famous_brand = sql_engine.connect_pt_product(config.oe_hostname, config.oe_password, path.product_database,
                                                    sql.famous_brand_sql)
    df_holiday = sql_engine.connect_pt_product(config.oe_hostname, config.oe_password, path.product_database,
                                               sql.holiday_sql)

    """
    for page in range(total_pages):
        # 计算查询的起始位置
        start_index = page * page_size
        end_index = min((page + 1) * page_size, total_pages)
        # 执行查询操作"""

    while id_start < id_end:
        # 1.数据连接
        start_time = time.time()

        df_product = sql_engine.connect_pt_product_page(config.sellersprite_hostname, config.sellersprite_password,
                                                        config.sellersprite_database,
                                                        db_util.group_asin_sql(id_start, id_increment))

        if df_product.empty:
            id_start = id_start + id_increment
            continue

        df_duplicate = sql_engine.connect_pt_product_page(config.sellersprite_hostname, config.sellersprite_password,
                                                          config.sellersprite_database,
                                                          db_util.group_duplicate_sql(id_start, id_increment,
                                                                                      path.pt_product_duplicate))

        df_relation = sql_engine.connect_pt_product_page(config.sellersprite_hostname, config.sellersprite_password,
                                                         config.sellersprite_database,
                                                         db_util.group_traffic_sql(id_start, id_increment,
                                                                                   path.pt_relevance_asins,
                                                                                   path.pt_relation_traffic))

        df_relation_old = sql_engine.connect_pt_product_page(config.sellersprite_hostname, config.sellersprite_password,
                                                             config.sellersprite_database,
                                                             db_util.group_traffic_sql(id_start, id_increment,
                                                                                       path.pt_relevance_asins_old,
                                                                                       path.pt_relation_traffic_old))

        df_relation_older = sql_engine.connect_pt_product_page(config.sellersprite_hostname,
                                                               config.sellersprite_password,
                                                               config.sellersprite_database,
                                                               db_util.group_traffic_sql(id_start, id_increment,
                                                                                         path.pt_relevance_asins_older,
                                                                                         path.pt_relation_traffic_older))

        # df_relation_add = sql_engine.connect_pt_product_page(config.sellersprite_hostname, config.sellersprite_password,
        #                                      config.sellersprite_database, sql_traffic_add)

        print("数据连接用时：" + (time.time() - start_time).__str__())

        # 2.数据预处理
        start_time = time.time()
        df_relation = pd.concat([df_relation, df_relation_old, df_relation_older])
        # 字段整理
        df_relation['ASIN'] = df_relation['asin']
        df_relation = df_relation[(df_relation['relevance'] >= 0.5) & (df_relation['category_relevance'] >= 0)]
        # df_relation_add['asin'] = df_relation_add['ASIN']
        # df_relation_add = df_relation_add[df_relation_add['relevance'] > 0]

        # relation_add_con_list = ['category_bsr_growth', 'sales_growth', 'reviews_rate', 'gross_margin']
        # for add_con in relation_add_con_list:
        #     data_cleaning_util.percent_convert(df_relation_add, add_con)
        #
        # df_related_traffic = pd.concat([df_relation, df_relation_add], axis=0, ignore_index=True)

        df_related_traffic = df_relation

        if df_related_traffic.empty:
            continue

        # 数据类型转换
        product_con_list_1 = ['category_bsr_growth', 'sales_growth', 'price', 'monthly_revenue', 'gross_margin',
                              'fba_fees']
        for con_i in product_con_list_1:
            df_related_traffic[con_i] = data_cleaning_util.convert_type(df_related_traffic, con_i, 2)

        product_con_list_2 = ['sales', 'qa', 'ratings', 'variations']
        for con_j in product_con_list_2:
            df_related_traffic[con_j] = data_cleaning_util.convert_type(df_related_traffic, con_j, 0)

        df_related_traffic['reviews_rate'] = data_cleaning_util.convert_type(df_related_traffic, 'reviews_rate', 4)
        df_related_traffic['rating'] = data_cleaning_util.convert_type(df_related_traffic, 'rating', 1)

        product_con_list_3 = ['brand', 'title', 'category_path', 'category', 'sub_category', 'parent', 'seller_type',
                              'buybox_seller', 'ac_keyword', 'weight', '二级类目']
        for con_l in product_con_list_3:
            df_related_traffic[con_l] = data_cleaning_util.convert_str(df_related_traffic, con_l)

        product_con_list_5 = ['title', 'category_path', 'category', 'sub_category', 'ac_keyword', 'weight', '二级类目']
        for con_lower in product_con_list_5:
            df_related_traffic[con_lower] = df_related_traffic[con_lower].str.lower()

        product_con_list_4 = ['date_available', 'update_time']
        for con_h in product_con_list_4:
            df_related_traffic[con_h] = data_cleaning_util.convert_date(df_related_traffic, con_h)

        # data_cleaning_util.percent_convert(df_related_traffic, 'monthly_revenue_increase')
        df_related_traffic['sales_growth'] = data_cleaning_util.convert_type(df_related_traffic, 'sales_growth', 4)
        df_related_traffic['monthly_revenue_increase'] = df_related_traffic['sales_growth']

        df_famous_brand['brand'] = data_cleaning_util.convert_str(df_famous_brand, 'brand')
        df_famous_brand['疑似知名品牌'] = data_cleaning_util.convert_type(df_famous_brand, '疑似知名品牌', 0)
        df_holiday['节日关键词'] = data_cleaning_util.convert_str(df_holiday, '节日关键词')
        df_holiday['节日关键词'] = df_holiday['节日关键词'].str.lower()

        for error_u, replace_m in para.replace_related_type_dict.items():
            df_related_traffic.loc[:, 'related_type'] = df_related_traffic['related_type'].str.replace(error_u,
                                                                                                       replace_m,
                                                                                                       regex=False)
        df_related_traffic.loc[:, 'related_type'] = df_related_traffic['related_type'].fillna("以图搜图")

        print("数据预处理用时：" + (time.time() - start_time).__str__())

        # 3.获取主推变体款
        start_time = time.time()

        # 3.1开售月数
        calculate_util.month_available(df_related_traffic)

        # 3.2排序
        # 销额处理
        df_related_traffic['monthly_revenue'] = df_related_traffic['monthly_revenue'].fillna(0)
        # df_related_traffic['monthly_revenue_avg'] = df_related_traffic.groupby(df_related_traffic['parent'])[
        #     'monthly_revenue'].mean().reset_index(drop=True)

        df_related_traffic_monthly_revenue_avg = df_related_traffic.groupby(df_related_traffic['parent'])[
            'monthly_revenue'].mean()
        df_related_traffic_monthly_revenue_avg = data_cleaning_util.convert_col(df_related_traffic_monthly_revenue_avg,
                                                                                'monthly_revenue_avg')
        df_related_traffic = df_related_traffic.merge(df_related_traffic_monthly_revenue_avg, how='left', on='parent')

        # 竞品销额均为空的处理
        df_parent_0 = df_related_traffic.loc[df_related_traffic['monthly_revenue_avg'] == 0]
        df_parent_0.loc[:, 'rank'] = calculate_util.sort_and_rank_group(df_parent_0)

        # 竞品销额含非空的处理
        df_parent_1 = df_related_traffic.loc[
            (df_related_traffic['monthly_revenue_avg'] > 0) & (df_related_traffic['monthly_revenue'] > 0)]
        df_parent_1.loc[:, 'rank'] = calculate_util.sort_and_rank_group(df_parent_1)

        df_main = pd.concat([df_parent_0[df_parent_0['rank'] == 1], df_parent_1[df_parent_1['rank'] == 1]])

        print("获取主推变体款用时：" + (time.time() - start_time).__str__())

        # 4.M相关指标计算
        start_time = time.time()

        # 去除空值行
        df_main_weight = df_main[df_main['weight'].notnull()]

        if not df_main_weight.empty:
            # 替换错误单位
            for error_unit, replacement in para.replace_weight_error_dict.items():
                df_main_weight['weight'] = df_main_weight['weight'].str.replace(error_unit, replacement, regex=False)

            # 一次性分割并创建新列
            weight_split = df_main_weight['weight'].str.split(" ", expand=True)
            df_main_weight['重量值'] = weight_split[0]
            df_main_weight['单位'] = weight_split[1]

            # 去除不合法单位和重量值
            df_main_weight.loc[~df_main_weight['单位'].isin(para.replace_weight_unit_list), '单位'] = np.nan
            df_main_weight['重量值判断'] = df_main_weight['重量值'].str.replace(".", "")
            df_main_weight.loc[~df_main_weight['重量值判断'].str.isdecimal(), '重量值'] = "-1"
            df_main_weight['重量值'] = np.where(df_main_weight['重量值判断'] == "-1", np.nan, df_main_weight['重量值'])

            # 计算换算值
            df_main_weight['换算'] = df_main_weight['单位'].replace(para.replace_weight_dict, regex=False)

            # 计算重量
            df_main_weight['重量值'] = df_main_weight['重量值'].astype(float)
            df_main_weight['换算'] = df_main_weight['换算'].astype(float)
            df_main_weight['重量(g)'] = np.where(df_main_weight['重量值'].astype(float) * 1 > 0, round(
                df_main_weight['重量值'].astype(float) * df_main_weight['换算'].astype(float), 4), np.nan)
        else:
            # 如果DataFrame为空，创建空的DataFrame并设置重量列为NaN
            df_main_weight = pd.DataFrame(columns=df_main.columns)
            df_main_weight['重量(g)'] = np.nan

        # 直发FBM可能性
        df_traffic_fbm = calculate_util.get_fbm(df_main_weight)

        df_traffic = df_main.merge(df_traffic_fbm, how='left', on='id')
        df_traffic['直发FBM可能性'] = df_traffic['直发FBM可能性'].fillna(0)

        df_traffic['预估FBA占比'] = np.where(df_traffic['fba_fees'] * 1 > 0,
                                         np.fmin(1, df_traffic['fba_fees'] / df_traffic['price']), para.fba_fees_rate)
        df_traffic['预估头程占比'] = np.where(df_traffic['预估FBA占比'] * 1 > 0, np.fmin(1, df_traffic['预估FBA占比'] / 2.5),
                                        para.pre_fees_rate)
        df_traffic['预估货值占比'] = common_util.get_cut(df_traffic, 'price', [0, 6, 10, 15, 30, 50, 100, 200, 9999],
                                                   [0.08, 0.1, 0.15, 0.2, 0.25, 0.27, 0.3, 0.35])
        df_traffic['预估货值占比'] = data_cleaning_util.convert_type(df_traffic, '预估货值占比', 2)

        df_traffic['预估毛利率_FBM'] = df_traffic['gross_margin'] - df_traffic['预估头程占比'] * 2 - para.product_fees_rate
        df_traffic['预估毛利率_FBA'] = df_traffic['gross_margin'] - df_traffic['预估头程占比'] - para.product_fees_rate
        df_traffic['预估毛利率_反推'] = np.where(
            (df_traffic['直发FBM可能性'] >= 1) & (df_traffic['gross_margin'] >= para.gross_margin_upper),
            df_traffic['预估毛利率_FBM'], df_traffic['预估毛利率_FBA'])

        df_traffic['预估毛利率'] = np.where(abs(df_traffic['gross_margin'] * 1) > 0,
                                       np.fmax(-1, np.fmin(1, df_traffic['预估毛利率_反推'])),
                                       np.fmax(-1, np.fmin(1, 1 - df_traffic['预估FBA占比'] - df_traffic['预估头程占比'] -
                                                           para.referral_fees_rate - df_traffic['预估货值占比'])))
        df_traffic['毛利率级别_上限'] = np.fmin(calculation_util.get_mround(df_traffic, '预估毛利率', '毛利率级别_上限', 0.05),
                                         para.gross_rate_upper)
        df_traffic['毛利率级别_下限'] = np.fmax(calculation_util.get_mround(df_traffic, '预估毛利率', '毛利率级别_下限', -0.05),
                                         para.gross_rate_lower)
        df_traffic['毛利率级别'] = np.where(df_traffic['预估毛利率'] >= 0, df_traffic['毛利率级别_上限'], df_traffic['毛利率级别_下限'])

        df_traffic['毛估资金利用率'] = df_traffic['预估毛利率'] / (df_traffic['预估头程占比'] + para.product_fees_rate)

        print("M相关指标计算用时：" + (time.time() - start_time).__str__())

        # 5.推荐度相关指标计算
        start_time = time.time()
        # M相关指标
        df_traffic['高资金利用率'] = np.where(abs(df_traffic['毛估资金利用率']) > 0,
                                        df_traffic['毛估资金利用率'] / para.product_revenue_std - 1, 0)
        # I相关指标
        # calculate_util.month_available(df_traffic)

        # S相关指标
        calculate_util.get_revenue(df_traffic)

        conditions_lqs_1 = (df_traffic['预估毛利率'] >= -0.05) & (df_traffic['lqs'] > 0) & (df_traffic['lqs'] <= 8)
        conditions_lqs_2 = (df_traffic['开售月数'] >= 24) & (df_traffic['rating'] >= 4) & (df_traffic['ratings'] >= 10) & (
                df_traffic['预估毛利率'] >= -0.15) & (df_traffic['lqs'] > 0) & (df_traffic['lqs'] <= 8)
        df_traffic['高销低LQS_pre'] = np.fmax(0, np.fmin(3, 0.5 + df_traffic['销额级数'] * para.lqs_std / df_traffic['lqs']))
        df_traffic['高销低LQS'] = np.where(conditions_lqs_1 | conditions_lqs_2, df_traffic['高销低LQS_pre'], 0)

        df_traffic['开售月数_QA'] = np.fmin(df_traffic['开售月数'], 24)
        df_traffic['月均QA数'] = np.where(df_traffic['qa'] > 0, round(df_traffic['qa'] / df_traffic['开售月数_QA'], 1), 0)

        conditions_available = (df_traffic['开售月数'] >= para.available_std) & (
                df_traffic['monthly_revenue'] >= para.monthly_revenue_C)

        df_traffic['长期上架少Q&A'] = np.where(conditions_available,
                                          np.fmax(-1, para.qa_std - df_traffic['月均QA数'] / para.qa_std), 0)

        df_traffic['长期上架无A+'] = np.where(conditions_available & (df_traffic['ebc_available'] != "Y"), 1, 0)

        df_traffic['长期上架无视频'] = np.where(conditions_available & (df_traffic['video_available'] != "Y"), 1, 0)

        df_traffic['类轻小直发FBM'] = np.where(df_traffic['直发FBM可能性'] > 0,
                                          np.fmax(0, 1 + df_traffic['销额级数'] * np.fmin(1, df_traffic['直发FBM可能性'] / 2)),
                                          0)

        df_traffic['差评好卖'] = np.where((df_traffic['开售月数'] >= para.available_std) & (
                df_traffic['monthly_revenue'] >= para.monthly_revenue_C / 2) & (df_traffic['ratings'] >= 10) & (
                                              df_traffic['rating'] >= 3) & (df_traffic['rating'] < 4) & (
                                              abs(df_traffic['预估毛利率']) > 0) & (
                                              df_traffic['category_bsr_growth'] >= -0.5),
                                      0.5 + df_traffic['销额级数'] * (4.5 - df_traffic['rating']), 0)

        # 知名品牌
        df_traffic_brand = df_traffic[df_traffic['brand'].notnull()]
        if df_traffic_brand.empty:
            df_traffic['疑似知名品牌'] = 0
        else:
            df_traffic = df_traffic.merge(df_famous_brand, how='left', on='brand')

        df_traffic['combined_kw'] = df_traffic['title'] + "" + df_traffic['sub_category'] + "" + df_traffic[
            'ac_keyword']

        # df_traffic.apply(lambda row: print(calculate_util.match_holidays(row,df_holiday['节日关键词'])), axis=1)
        df_traffic[['疑似节日性', '节日名']] = df_traffic.apply(
            lambda row: pd.Series(calculate_util.match_holidays(row, df_holiday['节日关键词'])), axis=1)

        df_traffic['知名品牌_1'] = np.where(df_traffic['疑似知名品牌'] * 1 > 0,
                                        -df_traffic['疑似知名品牌'] / np.where(df_traffic['疑似节日性'] * 1 > 0, 2, 1), 0)

        conditions_brand = (df_traffic['seller_type'] == 'AMZ') | (df_traffic['buybox_seller'] == 'Amazon') | (
                df_traffic['buybox_seller'] == 'Amazon.com')
        df_traffic['知名品牌_2'] = np.where(conditions_brand, -3, 0)

        df_traffic['知名品牌'] = np.where(df_traffic['brand'] == 'Generic', 1, df_traffic['知名品牌_1'] + df_traffic['知名品牌_2'])

        df_traffic['知名品牌'] = df_traffic['知名品牌'].fillna(0)

        # 疑似节日性
        df_traffic['疑似节日性'] = np.where(df_traffic['疑似节日性'] * 1 > 3, "3+", df_traffic['疑似节日性'])

        print("知名品牌用时：" + (time.time() - start_time).__str__())
        start_time = time.time()

        # 是否个人定制
        custom_kw = ['custom', 'personalize', 'personalized', 'custom-made', 'customized', 'made-to-order']
        df_traffic['custom_kw'] = df_traffic.apply(
            lambda row: pd.Series(calculate_util.match_custom_kw(row, custom_kw)),
            axis=1)
        df_traffic['是否个人定制'] = np.where(df_traffic['custom_kw'] * 1 > 0, 1, 0)

        # 是否翻新
        df_traffic.loc[df_traffic['title'].astype(str).str.contains('renewed', na=False, regex=False), '是否翻新'] = 1
        df_traffic['是否翻新'] = np.where(df_traffic['是否翻新'] == 1, 1, 0)

        # L相关指标
        df_traffic['平均变体月销额等级'] = np.where(df_traffic['monthly_revenue'] * 1 > 0, np.log2(
            df_traffic['monthly_revenue'] / df_traffic['variations'] / (para.monthly_revenue_C / 2)), 0)
        df_traffic['变体等级'] = np.log2(df_traffic['variations'])
        df_traffic['少变体'] = np.where(df_traffic['variations'] <= 2, 0,
                                     np.fmax(-10, np.fmin(0, df_traffic['平均变体月销额等级'] - 0.5 * df_traffic['变体等级'] + 0.5)))

        # E相关指标
        conditions_new_product = (df_traffic['开售月数'] < para.available_std)

        df_traffic['新品爬坡快'] = np.where(conditions_new_product & (df_traffic['monthly_revenue'] * 1 >= 0),
                                       np.fmax(0, np.log2(df_traffic['monthly_revenue'] / df_traffic['开售月数'] / (
                                               para.monthly_revenue_C / para.revenue_month_C))), 0)

        df_traffic['新品增评好'] = np.where(
            conditions_new_product & (df_traffic['rating'] * 1 >= 4) & (df_traffic['ratings'] * 1 >= 10),
            np.fmax(0, np.fmin(2, np.log(
                df_traffic['ratings'] / df_traffic['开售月数'] / df_traffic['variations'] / np.log(
                    5)))), 0)

        df_traffic['新品NSR'] = np.where(conditions_new_product & (df_traffic['new_release'] == "Y"), 1, 0)

        df_traffic['新品AC标'] = np.where(conditions_new_product & (df_traffic['ac'] == "Y"), 1, 0)

        df_traffic['销级星数比'] = np.where(df_traffic['rating'] * 1 > 0,
                                       df_traffic['销额级数'] / round(2 + df_traffic['rating'] / 100), 0)

        df_traffic['少评好卖'] = np.where(df_traffic['销额级数'] * 1 > 0, np.fmax(0, round(df_traffic['销级星数比'] - 1, 2)), 0)

        # 推荐度计算
        df_recommend = df_traffic[
            ['id', '销额级数', '高资金利用率', '高销低LQS', '长期上架少Q&A', '长期上架无A+', '长期上架无视频', '类轻小直发FBM', '差评好卖', '知名品牌', '少变体',
             '新品爬坡快',
             '新品增评好', '新品NSR', '新品AC标']]
        recommend_weights = np.array([0, 0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 1.5, 1, 1, 1, 0.5, 0.5, 0.5, 0.5])

        df_recommend = df_recommend.astype(float)
        recommend_weights = recommend_weights.astype(float)

        df_recommend['推荐度'] = df_recommend.dot(recommend_weights)
        df_recommend['推荐度'] = df_recommend['推荐度'].fillna(0)
        df_recommend['推荐度'] = data_cleaning_util.convert_type(df_recommend, '推荐度', 1)

        df_recommend = df_recommend[['id', '推荐度']]
        df_traffic = df_traffic.merge(df_recommend, how='left', on='id')

        # 数据更新日期
        df_traffic['数据更新时间'] = update_date
        data_cleaning_util.convert_date(df_traffic, '数据更新时间')

        # 类目清洗
        df_traffic['category'] = df_traffic['category'].replace(para.replace_category_dict, regex=False)
        df_traffic['二级类目'] = df_traffic['二级类目'].replace(para.replace_category_dict_2, regex=False)

        # 剔除类目
        df_traffic.loc[
            df_traffic['category_path'].str.contains(para.regex_pattern_kw, na=False, regex=False), '剔除类目'] = 1
        df_traffic['剔除类目'] = np.where(df_traffic['剔除类目'] == 1, 1, 0)

        # 数据格式整理
        product_con_list = ['预估FBA占比', '预估头程占比', '预估毛利率', '毛估资金利用率', '销额级数', '高资金利用率', '高销低LQS', '类轻小直发FBM',
                            '平均变体月销额等级',
                            '新品爬坡快', '长期上架少Q&A']
        for con_k in product_con_list:
            df_traffic[con_k] = data_cleaning_util.convert_type(df_traffic, con_k, 4)

        print("推荐度计算用时：" + (time.time() - start_time).__str__())

        # 6.聚合字段计算
        start_time = time.time()

        # 获取有销额产品
        traffic_revenue_pass = df_traffic.query('monthly_revenue > 0')

        # 获取销额TOP5关联竞品
        traffic_revenue_pass['revenue_rank'] = traffic_revenue_pass.groupby(df_traffic['related_asin'])[
            'monthly_revenue'].rank(ascending=False, method='first')
        traffic_top5_df = traffic_revenue_pass.query('revenue_rank <= 5')
        traffic_top25_df = traffic_revenue_pass.query('revenue_rank <= 25')

        # 聚合表生成
        traffic_df = traffic_top5_df.groupby('related_asin').agg({'relevance': 'first', 'asin': lambda x: ','.join(
            x.astype(str)), '数据更新时间': 'first'}).reset_index()

        # 相关竞品款数
        traffic_count = calculation_util.product_count(df_traffic, 'related_asin', 'asin', '相关竞品款数')

        # 有销额竞品款数
        traffic_revenue_count = calculation_util.product_count(traffic_revenue_pass, 'related_asin', 'asin', '有销额竞品款数')
        traffic_revenue_pass = traffic_revenue_pass.merge(traffic_revenue_count, how='left', on='related_asin')

        # 有销额推荐达标款数
        traffic_recommend = traffic_revenue_pass.query('推荐度>=3')
        traffic_recommend_count = calculation_util.product_count(traffic_recommend, 'related_asin', 'asin', '有销额推荐达标款数')

        traffic_df = traffic_df.merge(traffic_count, how='left', on='related_asin') \
            .merge(traffic_revenue_count, how='left', on='related_asin') \
            .merge(traffic_recommend_count, how='left', on='related_asin')

        # 综合竞品推荐度
        traffic_df_recommend_all = calculate_util.product_recommend(df_traffic, '推荐度', 'relevance', '综合竞品推荐度_all')
        traffic_df_recommend_revenue = calculate_util.product_recommend(traffic_revenue_pass, '推荐度', 'relevance',
                                                                        '综合竞品推荐度_有销额')

        traffic_df = traffic_df.merge(traffic_df_recommend_all, how='left', on='related_asin') \
            .merge(traffic_df_recommend_revenue, how='left', on='related_asin')

        traffic_df['综合竞品推荐度'] = np.where(abs(traffic_df['综合竞品推荐度_有销额'] * 1) > 0, traffic_df['综合竞品推荐度_有销额'],
                                         traffic_df['综合竞品推荐度_all'])

        # 推荐级别v1
        traffic_df['达标推荐度占比'] = np.where(traffic_df['有销额推荐达标款数'] * 1 > 0,
                                         traffic_df['有销额推荐达标款数'] / traffic_df['有销额竞品款数'], np.nan)

        condition_recommend_pre = abs(traffic_df['综合竞品推荐度'] * traffic_df['达标推荐度占比'] * 1) >= 0
        condition_recommend = [
            (traffic_df['综合竞品推荐度'] >= 3) & (traffic_df['达标推荐度占比'] >= 0.3) & (
                    traffic_df['综合竞品推荐度'] * traffic_df['达标推荐度占比'] >= 4 * 0.4) & (traffic_df['有销额竞品款数'] >= 10),
            (traffic_df['综合竞品推荐度'] >= 2.5) & (traffic_df['达标推荐度占比'] >= 0.25) & (
                    traffic_df['综合竞品推荐度'] * traffic_df['达标推荐度占比'] >= 3 * 0.3) & (traffic_df['有销额竞品款数'] >= 10),
            (traffic_df['综合竞品推荐度'] >= 1.5) & (traffic_df['达标推荐度占比'] >= 0.2) & (
                    traffic_df['综合竞品推荐度'] * traffic_df['达标推荐度占比'] >= 2 * 0.25),
            (traffic_df['综合竞品推荐度'] >= 1) & (traffic_df['达标推荐度占比'] >= 0.15) & (
                    traffic_df['综合竞品推荐度'] * traffic_df['达标推荐度占比'] >= 1.5 * 0.2)]
        labels_group = [2, 1, 0, -1]

        traffic_df['推荐级别v1'] = np.select(condition_recommend, labels_group, default=-2)
        traffic_df['推荐级别v1'] = np.where(condition_recommend_pre, traffic_df['推荐级别v1'], 0)

        # 获取推荐主推款竞品
        traffic_recommend_tag = traffic_df[['related_asin', '综合竞品推荐度', '推荐级别v1']]
        # df_traffic_recommend = traffic_top5_df.merge(traffic_recommend_tag, how='left', on='related_asin')
        df_traffic_recommend = traffic_top25_df.merge(traffic_recommend_tag, how='left', on='related_asin')

        # -------------------------------V2版本-------------------------34个维度聚合------------------------------------

        # S规模分析
        # TOP5月销额
        traffic_revenue_top5 = calculation_util.product_sum(traffic_top5_df, 'related_asin', 'monthly_revenue',
                                                            'TOP5月销额')

        traffic_df = traffic_df.merge(traffic_revenue_top5, how='left', on='related_asin')

        # TOP5月均销额
        traffic_df['TOP5月均销额'] = np.where(traffic_df['有销额竞品款数'] > 0,
                                          traffic_df['TOP5月销额'] / np.fmin(5, traffic_df['有销额竞品款数']), np.nan)

        traffic_df['TOP5月均销额_score'] = common_util.get_cut(traffic_df, 'TOP5月均销额', para.s_sales_bins,
                                                           para.s_sales_labels)
        traffic_df['TOP5月均销额_tag'] = common_util.get_cut(traffic_df, 'TOP5月均销额', para.s_sales_bins, para.s_sales_tags)

        # 利基月GMV
        traffic_revenue = calculation_util.product_sum(traffic_revenue_pass, 'related_asin', 'monthly_revenue',
                                                       '利基月GMV')

        traffic_df = traffic_df.merge(traffic_revenue, how='left', on='related_asin')

        # 有销额竞品款数占比
        traffic_df['有销额竞品款数占比'] = np.where(traffic_df['相关竞品款数'] > 0, traffic_df['有销额竞品款数'] / traffic_df['相关竞品款数'],
                                           np.nan)

        # M资金回报分析
        # 价格中位数
        traffic_price_median = calculation_util.product_median(df_traffic, 'related_asin', 'price', '价格中位数')

        traffic_df = traffic_df.merge(traffic_price_median, how='left', on='related_asin')

        # 价格标准差
        traffic_price_std = calculation_util.product_std(df_traffic, 'related_asin', 'price', '价格标准差')

        traffic_df = traffic_df.merge(traffic_price_std, how='left', on='related_asin')

        # 价格集中度
        traffic_df['价格集中度'] = np.where(traffic_df['价格中位数'] > 0, 1 - traffic_df['价格标准差'] / traffic_df['价格中位数'], np.nan)

        traffic_df['价格集中度_score'] = common_util.get_cut(traffic_df, '价格集中度', para.m_price_bins, para.m_price_labels)
        traffic_df['价格集中度_tag'] = common_util.get_cut(traffic_df, '价格集中度', para.m_price_bins, para.m_price_tags)

        # 预估平均毛利率
        traffic_gross_margin = calculation_util.product_mean(traffic_revenue_pass, 'related_asin', '预估毛利率', '预估平均毛利率')

        traffic_df = traffic_df.merge(traffic_gross_margin, how='left', on='related_asin')

        traffic_df['预估平均毛利率_score'] = common_util.get_cut(traffic_df, '预估平均毛利率', para.m_gross_bins, para.m_gross_labels)
        traffic_df['预估平均毛利率_tag'] = common_util.get_cut(traffic_df, '预估平均毛利率', para.m_gross_bins, para.m_gross_tags)

        # 预估平均资金利用率
        traffic_revenue_reward = calculation_util.product_mean(traffic_revenue_pass, 'related_asin', '毛估资金利用率',
                                                               '预估平均资金利用率')

        traffic_df = traffic_df.merge(traffic_revenue_reward, how='left', on='related_asin')

        traffic_df['预估平均资金利用率_score'] = common_util.get_cut(traffic_df, '预估平均资金利用率', para.m_revenue_bins,
                                                            para.m_revenue_labels)
        traffic_df['预估平均资金利用率_tag'] = common_util.get_cut(traffic_df, '预估平均资金利用率', para.m_revenue_bins,
                                                          para.m_revenue_tags)

        # 获取FBA运费不为空行
        traffic_fba_fees = df_traffic.query('fba_fees > 0')

        # 加权FBA运费
        traffic_fba_fees_mean = calculation_util.product_mean(traffic_fba_fees, 'related_asin', 'fba_fees', '加权FBA运费')

        traffic_df = traffic_df.merge(traffic_fba_fees_mean, how='left', on='related_asin')

        traffic_df['加权FBA运费_score'] = common_util.get_cut(traffic_df, '加权FBA运费', para.m_fba_bins, para.m_fba_labels)
        traffic_df['加权FBA运费_tag'] = common_util.get_cut(traffic_df, '加权FBA运费', para.m_fba_bins, para.m_fba_tags)

        # 获取FBM配送产品
        traffic_fbm_df = traffic_revenue_pass.query('seller_type == "FBM"')

        # FBM配送产品数
        traffic_fbm_count = calculation_util.product_count(traffic_fbm_df, 'related_asin', 'asin', 'FBM配送产品数')

        traffic_df = traffic_df.merge(traffic_fbm_count, how='left', on='related_asin')

        # FBM配送占比
        traffic_df['FBM配送占比'] = np.where(traffic_df['有销额竞品款数'] > 0, traffic_df['FBM配送产品数'] / traffic_df['有销额竞品款数'],
                                         np.nan)

        traffic_df['FBM配送占比_score'] = common_util.get_cut(traffic_df, 'FBM配送占比', para.m_fbm_bins, para.m_fbm_labels)
        traffic_df['FBM配送占比_tag'] = common_util.get_cut(traffic_df, 'FBM配送占比', para.m_fbm_bins, para.m_fbm_tags)

        # 获取直发FBM产品
        traffic_fbm_cal_df = traffic_revenue_pass.query('直发FBM可能性 >= 1')

        # 直发FBM产品数
        traffic_fbm_cal_count = calculation_util.product_count(traffic_fbm_cal_df, 'related_asin', 'asin', '直发FBM产品数')

        traffic_df = traffic_df.merge(traffic_fbm_cal_count, how='left', on='related_asin')

        # 直发FBM产品占比
        traffic_df['直发FBM产品占比'] = np.where(traffic_df['有销额竞品款数'] > 0, traffic_df['直发FBM产品数'] / traffic_df['有销额竞品款数'],
                                           np.nan)

        # 直发FBM销额
        traffic_fbm_cal_revenue = calculation_util.product_sum(traffic_fbm_cal_df, 'related_asin', 'monthly_revenue',
                                                               '直发FBM销额')

        traffic_df = traffic_df.merge(traffic_fbm_cal_revenue, how='left', on='related_asin')

        # 直发FBM销额占比
        traffic_df['直发FBM销额占比'] = np.where(traffic_df['利基月GMV'] > 0, traffic_df['直发FBM销额'] / traffic_df['利基月GMV'],
                                           np.nan)

        conditions_fbm_rate = (traffic_df['直发FBM产品占比'] >= 0.15) | (traffic_df['直发FBM销额占比'] >= 0.2)

        traffic_df['直发FBM产品占比_score'] = np.where(conditions_fbm_rate, 2, np.nan)
        traffic_df['直发FBM产品占比_tag'] = np.where(conditions_fbm_rate, "直发FBM多", np.nan)

        # 直发FBM月均销额
        traffic_fbm_cal_revenue_mean = calculation_util.product_mean(traffic_fbm_cal_df, 'related_asin',
                                                                     'monthly_revenue',
                                                                     '直发FBM月均销额')

        traffic_df = traffic_df.merge(traffic_fbm_cal_revenue_mean, how='left', on='related_asin')

        traffic_df['直发FBM月均销额_score'] = common_util.get_cut(traffic_df, '直发FBM月均销额', para.m_fbm_cal_sales_bins,
                                                            para.m_fbm_cal_sales_labels)
        traffic_df['直发FBM月均销额_tag'] = common_util.get_cut(traffic_df, '直发FBM月均销额', para.m_fbm_cal_sales_bins,
                                                          para.m_fbm_cal_sales_tags)

        traffic_df['直发FBM月均销额_score'] = np.where(traffic_df['直发FBM产品数'] >= 3, traffic_df['直发FBM月均销额_score'], np.nan)
        traffic_df['直发FBM月均销额_tag'] = np.where(traffic_df['直发FBM产品数'] >= 3, traffic_df['直发FBM月均销额_tag'], np.nan)

        # I内卷分析
        # 广告蓝海度

        # 获取AMZ直营产品
        traffic_amz_df = traffic_revenue_pass.query('seller_type == "AMZ"')

        # AMZ直营销额
        traffic_amz_revenue = calculation_util.product_sum(traffic_amz_df, 'related_asin', 'monthly_revenue', 'AMZ直营销额')

        traffic_df = traffic_df.merge(traffic_amz_revenue, how='left', on='related_asin')

        # AMZ直营销额占比
        traffic_df['AMZ直营销额占比'] = np.where(traffic_df['利基月GMV'] > 0, traffic_df['AMZ直营销额'] / traffic_df['利基月GMV'],
                                           np.nan)

        traffic_df['AMZ直营销额占比_score'] = common_util.get_cut(traffic_df, 'AMZ直营销额占比', para.i_amz_bins, para.i_amz_labels)
        traffic_df['AMZ直营销额占比_tag'] = common_util.get_cut(traffic_df, 'AMZ直营销额占比', para.i_amz_bins, para.i_amz_tags)

        # 获取大牌商标产品
        traffic_famous_df = traffic_revenue_pass.query('疑似知名品牌 >= 4')
        # 大牌商标销额
        traffic_famous_revenue = calculation_util.product_sum(traffic_famous_df, 'related_asin', 'monthly_revenue',
                                                              '大牌商标销额')

        traffic_df = traffic_df.merge(traffic_famous_revenue, how='left', on='related_asin')

        # 大牌商标销额占比
        traffic_df['大牌商标销额占比'] = np.where(traffic_df['利基月GMV'] > 0, traffic_df['大牌商标销额'] / traffic_df['利基月GMV'],
                                          np.nan)

        traffic_df['大牌商标销额占比_score'] = common_util.get_cut(traffic_df, '大牌商标销额占比', para.i_famous_bins,
                                                           para.i_famous_labels)
        traffic_df['大牌商标销额占比_tag'] = common_util.get_cut(traffic_df, '大牌商标销额占比', para.i_famous_bins, para.i_famous_tags)

        # 获取中国卖家产品
        traffic_cn_df = df_traffic.query('buybox_location == "CN"')
        # 中国卖家产品数
        traffic_cn_count = calculation_util.product_count(traffic_cn_df, 'related_asin', 'asin', '中国卖家产品数')

        traffic_df = traffic_df.merge(traffic_cn_count, how='left', on='related_asin')

        # 中国卖家占比
        traffic_df['中国卖家占比'] = np.where(traffic_df['相关竞品款数'] > 0, traffic_df['中国卖家产品数'] / traffic_df['相关竞品款数'],
                                        np.nan)

        traffic_df['中国卖家占比_score'] = common_util.get_cut(traffic_df, '中国卖家占比', para.i_cn_bins, para.i_cn_labels)
        traffic_df['中国卖家占比_tag'] = common_util.get_cut(traffic_df, '中国卖家占比', para.i_cn_bins, para.i_cn_tags)

        # TOP5平均LQS
        traffic_top5_lqs = calculation_util.product_mean(traffic_top5_df, 'related_asin', 'lqs', 'TOP5平均LQS')

        traffic_df = traffic_df.merge(traffic_top5_lqs, how='left', on='related_asin')

        traffic_df['TOP5平均LQS_score'] = common_util.get_cut(traffic_df, 'TOP5平均LQS', para.i_lqs_top5_bins,
                                                            para.i_lqs_top5_labels)
        traffic_df['TOP5平均LQS_tag'] = common_util.get_cut(traffic_df, 'TOP5平均LQS', para.i_lqs_top5_bins,
                                                          para.i_lqs_top5_tags)

        # 获取冒出产品
        traffic_good_df = traffic_revenue_pass.query('monthly_revenue >= 600')

        # 冒出品产品数
        traffic_good_count = calculation_util.product_count(traffic_good_df, 'related_asin', 'asin', '冒出品产品数')

        traffic_df = traffic_df.merge(traffic_good_count, how='left', on='related_asin')

        # 冒出品平均LQS
        traffic_good_lqs = calculation_util.product_mean(traffic_good_df, 'related_asin', 'lqs', '冒出品平均LQS')

        traffic_df = traffic_df.merge(traffic_good_lqs, how='left', on='related_asin')

        traffic_df['冒出品平均LQS_score'] = common_util.get_cut(traffic_df, '冒出品平均LQS', para.i_lqs_bins, para.i_lqs_labels)
        traffic_df['冒出品平均LQS_tag'] = common_util.get_cut(traffic_df, '冒出品平均LQS', para.i_lqs_bins, para.i_lqs_tags)

        # 获取冒出品有A+产品
        traffic_good_ebc_df = traffic_good_df.query('ebc_available == "Y"')

        # 冒出品有A+产品数
        traffic_good_ebc_count = calculation_util.product_count(traffic_good_ebc_df, 'related_asin', 'asin',
                                                                '冒出品有A+产品数')

        traffic_df = traffic_df.merge(traffic_good_ebc_count, how='left', on='related_asin')

        # 冒出品A+占比
        traffic_df['冒出品A+占比'] = np.where(traffic_df['冒出品产品数'] >= 5, traffic_df['冒出品有A+产品数'] / traffic_df['冒出品产品数'],
                                         np.nan)

        traffic_df['冒出品A+占比_score'] = common_util.get_cut(traffic_df, '冒出品A+占比', para.i_ebc_bins, para.i_ebc_labels)
        traffic_df['冒出品A+占比_tag'] = common_util.get_cut(traffic_df, '冒出品A+占比', para.i_ebc_bins, para.i_ebc_tags)

        # 获取冒出品有视频产品
        traffic_good_video_df = traffic_good_df.query('video_available == "Y"')

        # 冒出品有视频产品数
        traffic_good_video_count = calculation_util.product_count(traffic_good_video_df, 'related_asin', 'asin',
                                                                  '冒出品有视频产品数')

        traffic_df = traffic_df.merge(traffic_good_video_count, how='left', on='related_asin')

        # 冒出品视频占比
        traffic_df['冒出品视频占比'] = np.where(traffic_df['冒出品产品数'] >= 5, traffic_df['冒出品有视频产品数'] / traffic_df['冒出品产品数'],
                                         np.nan)

        traffic_df['冒出品视频占比_score'] = common_util.get_cut(traffic_df, '冒出品视频占比', para.i_video_bins, para.i_video_labels)
        traffic_df['冒出品视频占比_tag'] = common_util.get_cut(traffic_df, '冒出品视频占比', para.i_video_bins, para.i_video_tags)

        # 获取冒出品有QA产品
        traffic_good_qa_df = traffic_good_df.query('qa >= 1')

        # 冒出品有QA产品数
        traffic_good_qa_count = calculation_util.product_count(traffic_good_qa_df, 'related_asin', 'asin', '冒出品有QA产品数')

        traffic_df = traffic_df.merge(traffic_good_qa_count, how='left', on='related_asin')

        # 冒出品QA占比
        traffic_df['冒出品QA占比'] = np.where(traffic_df['冒出品产品数'] >= 5, traffic_df['冒出品有QA产品数'] / traffic_df['冒出品产品数'],
                                         np.nan)

        traffic_df['冒出品QA占比_score'] = common_util.get_cut(traffic_df, '冒出品QA占比', para.i_qa_bins, para.i_qa_labels)
        traffic_df['冒出品QA占比_tag'] = common_util.get_cut(traffic_df, '冒出品QA占比', para.i_qa_bins, para.i_qa_tags)

        # 获取动销产品
        traffic_available_df = traffic_revenue_pass.query('monthly_revenue >= 100')

        # 动销品平均星级
        traffic_available_rating = calculation_util.product_mean(traffic_available_df, 'related_asin', 'rating',
                                                                 '动销品平均星级')

        traffic_df = traffic_df.merge(traffic_available_rating, how='left', on='related_asin')

        traffic_df['动销品平均星级_score'] = common_util.get_cut(traffic_df, '动销品平均星级', para.i_rating_bins,
                                                          para.i_rating_labels)
        traffic_df['动销品平均星级_tag'] = common_util.get_cut(traffic_df, '动销品平均星级', para.i_rating_bins, para.i_rating_tags)

        # 获取冒出品低星产品
        traffic_good_rating_lower = traffic_good_df.query('rating < 3.9')

        # 冒出品低星产品数(冒出品低星款数)
        traffic_good_rating_lower_count = calculation_util.product_count(traffic_good_rating_lower, 'related_asin',
                                                                         'asin',
                                                                         '冒出品低星款数')

        traffic_df = traffic_df.merge(traffic_good_rating_lower_count, how='left', on='related_asin')

        # 冒出品低星占比
        traffic_df['冒出品低星占比'] = np.where(traffic_df['冒出品产品数'] >= 5, traffic_df['冒出品低星款数'] / traffic_df['冒出品产品数'],
                                         np.nan)

        conditions_rating = [
            (traffic_df['冒出品低星占比'] >= 0.25) & (traffic_df['冒出品低星占比'] < 0.4) & (traffic_df['预估平均毛利率'] >= 0.2) | (
                    traffic_df['预估平均资金利用率'] >= 0.8) & (traffic_df['动销品平均星级'] < 4.2),
            (traffic_df['冒出品低星占比'] >= 0.4) & (traffic_df['预估平均毛利率'] >= 0.2) | (traffic_df['预估平均资金利用率'] >= 0.8) & (
                    traffic_df['动销品平均星级'] < 4.2)]
        traffic_df['冒出品低星占比_score'] = np.select(conditions_rating, para.i_rating_rate_labels, np.nan)
        traffic_df['冒出品低星占比_tag'] = np.select(conditions_rating, para.i_rating_rate_tags, np.nan)

        # L长尾分析
        # TOP5销额占比
        traffic_df['TOP5销额占比'] = np.where(traffic_df['利基月GMV'] > 0, traffic_df['TOP5月销额'] / traffic_df['利基月GMV'],
                                          np.nan)

        traffic_df['TOP5销额占比_score'] = common_util.get_cut(traffic_df, 'TOP5销额占比', para.l_sales_top5_bins,
                                                           para.l_sales_top5_labels)
        traffic_df['TOP5销额占比_tag'] = common_util.get_cut(traffic_df, 'TOP5销额占比', para.l_sales_top5_bins,
                                                         para.l_sales_top5_tags)

        # 非TOP5销额占比
        traffic_df['非TOP5销额占比'] = np.where(traffic_df['利基月GMV'] > 0, 1 - traffic_df['TOP5销额占比'], np.nan)

        traffic_df['非TOP5销额占比_score'] = common_util.get_cut(traffic_df, '非TOP5销额占比', para.l_sales_rate_bins,
                                                            para.l_sales_rate_labels)
        traffic_df['非TOP5销额占比_tag'] = common_util.get_cut(traffic_df, '非TOP5销额占比', para.l_sales_rate_bins,
                                                          para.l_sales_rate_tags)

        # 非TOP5月均销额
        traffic_df['非TOP5月均销额'] = np.where(traffic_df['有销额竞品款数'] > 5,
                                           (traffic_df['利基月GMV'] - traffic_df['TOP5月销额']) / (traffic_df['有销额竞品款数'] - 5),
                                           np.nan)

        traffic_df['非TOP5月均销额_score'] = common_util.get_cut(traffic_df, '非TOP5月均销额', para.l_sales_bins,
                                                            para.l_sales_labels)
        traffic_df['非TOP5月均销额_tag'] = common_util.get_cut(traffic_df, '非TOP5月均销额', para.l_sales_bins, para.l_sales_tags)

        # 动销品变体中位数
        traffic_available_variations = calculation_util.product_median(traffic_available_df, 'related_asin',
                                                                       'variations',
                                                                       '动销品变体中位数')

        traffic_df = traffic_df.merge(traffic_available_variations, how='left', on='related_asin')

        traffic_df['动销品变体中位数_score'] = common_util.get_cut(traffic_df, '动销品变体中位数', para.l_variations_bins,
                                                           para.l_variations_labels)
        traffic_df['动销品变体中位数_tag'] = common_util.get_cut(traffic_df, '动销品变体中位数', para.l_variations_bins,
                                                         para.l_variations_tags)

        # E新品冒出
        # 平均开售月数
        traffic_revenue_month = calculation_util.product_mean(traffic_revenue_pass, 'related_asin', '开售月数', '平均开售月数')

        traffic_df = traffic_df.merge(traffic_revenue_month, how='left', on='related_asin')

        traffic_df['平均开售月数_score'] = common_util.get_cut(traffic_df, '平均开售月数', para.e_month_bins, para.e_month_labels)
        traffic_df['平均开售月数_tag'] = common_util.get_cut(traffic_df, '平均开售月数', para.e_month_bins, para.e_month_tags)

        # 获取冒出品新品产品
        traffic_good_new = traffic_good_df.query('开售月数 <= 9')

        # 冒出品新品产品数
        traffic_good_new_count = calculation_util.product_count(traffic_good_new, 'related_asin', 'asin', '冒出品新品产品数')

        traffic_df = traffic_df.merge(traffic_good_new_count, how='left', on='related_asin')

        # 冒出品新品占比
        traffic_df['冒出品新品占比'] = np.where(traffic_df['冒出品产品数'] >= 5, traffic_df['冒出品新品产品数'] / traffic_df['冒出品产品数'],
                                         np.nan)

        traffic_df['冒出品新品占比_score'] = common_util.get_cut(traffic_df, '冒出品新品占比', para.e_new_bins, para.e_new_labels)
        traffic_df['冒出品新品占比_tag'] = common_util.get_cut(traffic_df, '冒出品新品占比', para.e_new_bins, para.e_new_tags)

        # 获取新品产品
        traffic_revenue_new = traffic_revenue_pass.query('有销额竞品款数 >= 5 and 开售月数 <= 9')

        # 新品平均月销额
        traffic_revenue_new_revenue = calculation_util.product_mean(traffic_revenue_new, 'related_asin',
                                                                    'monthly_revenue',
                                                                    '新品平均月销额')

        traffic_df = traffic_df.merge(traffic_revenue_new_revenue, how='left', on='related_asin')

        traffic_df['新品平均月销额_score'] = common_util.get_cut(traffic_df, '新品平均月销额', para.e_new_sales_bins,
                                                          para.e_new_sales_labels)
        traffic_df['新品平均月销额_tag'] = common_util.get_cut(traffic_df, '新品平均月销额', para.e_new_sales_bins,
                                                        para.e_new_sales_tags)

        # 平均星数
        traffic_available_ratings = calculation_util.product_mean(traffic_revenue_pass, 'related_asin', 'ratings',
                                                                  '平均星数')

        traffic_df = traffic_df.merge(traffic_available_ratings, how='left', on='related_asin')

        traffic_df['平均星数_score'] = common_util.get_cut(traffic_df, '平均星数', para.e_ratings_bins, para.e_ratings_labels)
        traffic_df['平均星数_tag'] = common_util.get_cut(traffic_df, '平均星数', para.e_ratings_bins, para.e_ratings_tags)

        # 销级星数比
        traffic_sales_mean = calculation_util.product_mean(df_traffic, 'related_asin', '销级星数比', '销级星数比')

        traffic_df = traffic_df.merge(traffic_sales_mean, how='left', on='related_asin')

        # 获取三标新品产品
        conditions_lables = ((traffic_revenue_pass['best_seller'] == 'Y') | (traffic_revenue_pass['ac'] == 'Y') | (
                traffic_revenue_pass['new_release'] == 'Y'))
        traffic_lables_df = traffic_revenue_pass[conditions_lables]
        traffic_new_lables_df = traffic_revenue_pass[(traffic_revenue_pass['开售月数'] <= 9) & conditions_lables]

        # 三标产品数
        traffic_lables_count = calculation_util.product_count(traffic_lables_df, 'related_asin', 'asin', '三标产品数')

        traffic_df = traffic_df.merge(traffic_lables_count, how='left', on='related_asin')

        # 三标新品产品数
        traffic_new_lables_count = calculation_util.product_count(traffic_new_lables_df, 'related_asin', 'asin',
                                                                  '三标新品产品数')

        traffic_df = traffic_df.merge(traffic_new_lables_count, how='left', on='related_asin')

        # 三标新品占比
        traffic_df['三标新品占比'] = np.where(traffic_df['三标产品数'] >= 5, traffic_df['三标新品产品数'] / traffic_df['三标产品数'],
                                        np.nan)

        traffic_df['三标新品占比_score'] = common_util.get_cut(traffic_df, '三标新品占比', para.e_new_lables_bins,
                                                         para.e_new_lables_labels)
        traffic_df['三标新品占比_tag'] = common_util.get_cut(traffic_df, '三标新品占比', para.e_new_lables_bins,
                                                       para.e_new_lables_tags)

        # 月销增长率不为空产品数
        traffic_sales_increase = traffic_revenue_pass.loc[traffic_revenue_pass['monthly_revenue_increase'].notnull()]

        traffic_sales_increase_count = calculation_util.product_count(traffic_sales_increase, 'related_asin', 'asin',
                                                                      '月销增长率不为空产品数')

        traffic_sales_increase = traffic_sales_increase.merge(traffic_sales_increase_count, how='left',
                                                              on='related_asin')

        # 月销增长率
        traffic_sales_increase_revenue = traffic_sales_increase.query('月销增长率不为空产品数 >= 10')

        traffic_sales_increase_rate = calculate_util.product_recommend(traffic_sales_increase_revenue,
                                                                       'monthly_revenue_increase',
                                                                       'monthly_revenue', '月销增长率')

        traffic_df = traffic_df.merge(traffic_sales_increase_rate, how='left', on='related_asin')

        traffic_df['月销增长率_score'] = common_util.get_cut(traffic_df, '月销增长率', para.e_sales_bins, para.e_sales_labels)
        traffic_df['月销增长率_tag'] = common_util.get_cut(traffic_df, '月销增长率', para.e_sales_bins, para.e_sales_tags)

        # 平均留评率
        traffic_ratings_rate_mean = calculation_util.product_mean(traffic_revenue_pass, 'related_asin', 'reviews_rate',
                                                                  '平均留评率')

        traffic_revenue_pass = traffic_revenue_pass.merge(traffic_ratings_rate_mean, how='left', on='related_asin')

        # 留评率标准差
        traffic_ratings_rate_std = calculation_util.product_std(traffic_revenue_pass, 'related_asin', 'reviews_rate',
                                                                '留评率标准差')

        traffic_revenue_pass = traffic_revenue_pass.merge(traffic_ratings_rate_std, how='left', on='related_asin')

        # 筛选留评率计算数据
        traffic_revenue_pass['留评率筛选标准'] = traffic_revenue_pass['平均留评率'] + 2 * traffic_revenue_pass['留评率标准差']

        traffic_ratings_rate_pass = traffic_revenue_pass.loc[
            traffic_revenue_pass['reviews_rate'] < traffic_revenue_pass['留评率筛选标准']]

        # 加权留评率
        traffic_ratings_avg = traffic_ratings_rate_pass.loc[traffic_ratings_rate_pass['reviews_rate'] <= 0.1]

        traffic_ratings_rate = calculate_util.product_recommend(traffic_revenue_pass, 'reviews_rate', 'monthly_revenue',
                                                                '加权留评率')
        traffic_df = traffic_df.merge(traffic_ratings_rate, how='left', on='related_asin')

        # PMI计算
        traffic_df = traffic_df.apply(lambda row: calculate_util.pmi_score(row), axis=1)

        traffic_df['PMI得分'] = traffic_df['P得分'] + traffic_df['M得分']

        condition_pmi = traffic_df['有销额竞品款数'] * 1 >= 3
        pmi_pre_list = ['PMI得分', 'P得分', 'M得分', 'P标签', 'M标签', 'I标签']

        for pmi_col in pmi_pre_list:
            traffic_df[pmi_col] = np.where(condition_pmi, traffic_df[pmi_col], np.nan)

        traffic_df['PMI得分'] = traffic_df['PMI得分'].fillna(0)

        # -------------------------------V4版本-------------------------去重打标---------------------------------------

        # 获取有效类目节点
        df_traffic_category = df_traffic.loc[
            (df_traffic['sub_category'].notnull()) & (df_traffic['sub_category'] != "none")]

        # 获取代表节点
        df_category = calculation_util.product_mode(df_traffic_category, 'related_asin', 'sub_category', '代表节点')

        df_traffic = df_traffic.merge(df_category, how='left', on='related_asin')
        traffic_df = traffic_df.merge(df_category, how='left', on='related_asin')

        # 同代表节点竞品数
        df_category_main = df_traffic.loc[df_traffic['sub_category'] == df_traffic['代表节点']]

        traffic_duplicate = calculation_util.product_count(df_category_main, 'related_asin', 'asin', '同代表节点竞品数')

        traffic_df = traffic_df.merge(traffic_duplicate, how='left', on='related_asin')

        # 代表度计算
        traffic_df['代表度'] = np.where(traffic_df['相关竞品款数'] * 1 > 0,
                                     round(traffic_df['同代表节点竞品数'] / traffic_df['相关竞品款数'], 2), np.nan)

        traffic_df['重复利基'] = np.where(traffic_df['代表度'] * 1 < 0.4, 0, 1)

        # -------------------------------V3版本-------------------------推荐级别重算------------------------------------

        # 推荐级别v2
        traffic_df['推荐级别v2'] = common_util.get_cut(traffic_df, 'PMI得分', para.pmi_bins, para.pmi_labels)

        # 综合投票分
        traffic_df['推荐级别v1'] = traffic_df['推荐级别v1'].astype(int)
        traffic_df['推荐级别v2'] = traffic_df['推荐级别v2'].astype(int)
        traffic_df['综合投票分'] = np.where(abs(traffic_df['推荐级别v1'] * traffic_df['推荐级别v2']) > 0,
                                       np.where(traffic_df['PMI得分'] > -8, round(
                                           abs((traffic_df['推荐级别v1'] + traffic_df['推荐级别v2']) / 2)) * np.where(
                                           (traffic_df['推荐级别v1'] + traffic_df['推荐级别v2']) / 2 >= 0, 1, -1), -2), 0)

        traffic_df['综合推荐级别'] = common_util.get_cut(traffic_df, '推荐级别v2', para.recommend_bins, para.recommend_labels)

        traffic_df['综合推荐级别'] = np.where(traffic_df['有销额竞品款数'] >= 5, traffic_df['综合推荐级别'], '<数据不足>')

        # -------------------------------V3版本-------------------------推荐级别重算------------------------------------

        # 格式转换
        traffic_list_0 = ['TOP5月均销额', 'TOP5月销额', '利基月GMV', '直发FBM月均销额', '冒出品低星款数', '非TOP5月均销额', '动销品变体中位数',
                          '新品平均月销额', '平均星数', '重复利基']
        for traffic_i in traffic_list_0:
            traffic_df[traffic_i] = data_cleaning_util.convert_type(traffic_df, traffic_i, 0)

        traffic_list_1 = ['综合竞品推荐度', 'TOP5平均LQS', '冒出品平均LQS', '平均开售月数']
        for traffic_j in traffic_list_1:
            traffic_df[traffic_j] = data_cleaning_util.convert_type(traffic_df, traffic_j, 1)

        traffic_list_2 = ['综合竞品推荐度', '有销额竞品款数占比', '价格中位数', '预估平均资金利用率', '加权FBA运费', '动销品平均星级', 'TOP5销额占比', '非TOP5销额占比',
                          '代表度', 'P得分', 'M得分', 'PMI得分']
        for traffic_k in traffic_list_2:
            traffic_df[traffic_k] = data_cleaning_util.convert_type(traffic_df, traffic_k, 2)

        traffic_list_3 = ['达标推荐度占比', '价格集中度', '预估平均毛利率', 'FBM配送占比', '直发FBM产品占比', '直发FBM销额占比', 'AMZ直营销额占比', '大牌商标销额占比',
                          '中国卖家占比', '冒出品A+占比', '冒出品视频占比', '冒出品QA占比', '冒出品低星占比', '冒出品新品占比', '销级星数比', '三标新品占比', '月销增长率',
                          '加权留评率']
        for traffic_l in traffic_list_3:
            traffic_df[traffic_l] = data_cleaning_util.convert_type(traffic_df, traffic_l, 3)

        df_group = traffic_df.merge(df_product, how='left', on='related_asin').merge(df_duplicate, how='left',
                                                                                     on='related_asin')

        # -------------------------------V2版本-------------------------34个维度聚合------------------------------------

        print("聚合字段计算用时：" + (time.time() - start_time).__str__())

        # 7.添加数据标签
        start_time = time.time()
        df_group_tag = df_group[
            ['related_asin', '相关竞品款数', '有销额竞品款数', '有销额推荐达标款数', '有销额竞品款数占比', '综合竞品推荐度', '达标推荐度占比', '推荐级别v1', 'TOP5月均销额',
             'TOP5月销额', '利基月GMV', '价格中位数', '价格集中度', '预估平均毛利率', '预估平均资金利用率', '加权FBA运费', 'FBM配送占比', '直发FBM产品占比',
             '直发FBM销额占比',
             '直发FBM月均销额', 'AMZ直营销额占比', '大牌商标销额占比', '中国卖家占比', 'TOP5平均LQS', '冒出品平均LQS', '冒出品A+占比', '冒出品视频占比', '冒出品QA占比',
             '动销品平均星级', '冒出品低星占比', 'TOP5销额占比', '非TOP5销额占比', '非TOP5月均销额', '动销品变体中位数', '平均开售月数', '冒出品新品占比', '新品平均月销额',
             '平均星数',
             '销级星数比', '三标新品占比', '月销增长率', '加权留评率', '推荐级别v2', '综合投票分', '综合推荐级别', '数据更新时间']]

        group_tag_list_5 = ['相关竞品款数', '有销额竞品款数', '有销额推荐达标款数']
        for group_i in group_tag_list_5:
            group_tag_i = group_i + "分布"
            calculation_util.get_mround(df_group_tag, group_i, group_tag_i, 5)

        group_tag_list_02 = ['综合竞品推荐度', 'TOP5平均LQS', '冒出品平均LQS', '预估平均资金利用率']
        for group_02 in group_tag_list_02:
            group_tag_02 = group_02 + "分布"
            calculation_util.get_mround(df_group_tag, group_02, group_tag_02, 0.2)

        group_tag_list_05 = ['有销额竞品款数占比', '达标推荐度占比', '价格集中度', 'FBM配送占比', '直发FBM产品占比', '直发FBM销额占比', 'AMZ直营销额占比',
                             '大牌商标销额占比',
                             '中国卖家占比', '冒出品A+占比', '冒出品视频占比', '冒出品QA占比', '冒出品低星占比', 'TOP5销额占比', '非TOP5销额占比', '冒出品新品占比',
                             '销级星数比', '三标新品占比', '月销增长率']
        for group_k in group_tag_list_05:
            group_tag_k = group_k + "分布"
            calculation_util.get_mround(df_group_tag, group_k, group_tag_k, 0.05)

        group_tag_list_10000 = ['TOP5月均销额', 'TOP5月销额', '利基月GMV']
        for group_10000 in group_tag_list_10000:
            group_tag_10000 = group_10000 + "分布"
            calculation_util.get_mround(df_group_tag, group_10000, group_tag_10000, 10000)

        group_tag_list_100 = ['直发FBM月均销额', '非TOP5月均销额', '新品平均月销额', '平均星数']
        for group_100 in group_tag_list_100:
            group_tag_100 = group_100 + "分布"
            calculation_util.get_mround(df_group_tag, group_100, group_tag_100, 100)

        group_tag_list_2 = ['价格中位数', '加权FBA运费']
        for group_2 in group_tag_list_2:
            group_tag_2 = group_2 + "分布"
            calculation_util.get_mround(df_group_tag, group_2, group_tag_2, 2)

        calculation_util.get_mround(df_group_tag, '预估平均毛利率', '预估平均毛利率分布', 0.02)
        calculation_util.get_mround(df_group_tag, '动销品平均星级', '动销品平均星级分布', 0.1)
        calculation_util.get_mround(df_group_tag, '平均开售月数', '平均开售月数分布', 3)
        calculation_util.get_mround(df_group_tag, '加权留评率', '加权留评率分布', 0.005)

        df_group_tag['data_id'] = df_group_tag['related_asin'] + " | " + update_date

        print("添加数据标签用时：" + (time.time() - start_time).__str__())

        # 8.字段整合
        start_time = time.time()

        # 竞品表
        df_traffic_table = df_traffic_recommend[['related_asin',
                                                 'relevance',
                                                 'related_type',
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
                                                 '是否个人定制',
                                                 '是否翻新',
                                                 '推荐度',
                                                 '二级类目',
                                                 '剔除类目',
                                                 'revenue_rank',
                                                 '数据更新时间']]

        df_traffic_table['clear_id'] = df_traffic_table['related_asin'] + " | " + df_traffic_table['ASIN']
        df_traffic_table = duplicate_util.df_cleaning(df_traffic_table, 'clear_id')
        df_traffic_table = df_traffic_table.drop(['clear_id'], axis=1)

        df_traffic_table.rename(columns={'related_asin': '原ASIN',
                                         'relevance': '关联度',
                                         'related_type': '关联类型',
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
                                         'update_time': '引流时间',
                                         'revenue_rank': '销售额排名'}, inplace=True)

        # 聚合表
        df_group = df_group[
            ['related_asin', 'price', 'recommend', '相关竞品款数', '有销额竞品款数', '有销额推荐达标款数', '有销额竞品款数占比', '综合竞品推荐度', '达标推荐度占比',
             '推荐级别v1', 'TOP5月均销额', 'TOP5月销额', '利基月GMV', '价格中位数', '价格集中度', '预估平均毛利率', '预估平均资金利用率', '加权FBA运费', 'FBM配送占比',
             '直发FBM产品占比', '直发FBM销额占比', '直发FBM月均销额', 'blue_ocean_estimate', 'AMZ直营销额占比', '大牌商标销额占比', '中国卖家占比',
             'TOP5平均LQS',
             '冒出品平均LQS', '冒出品A+占比', '冒出品视频占比', '冒出品QA占比', '动销品平均星级', '冒出品低星款数', '冒出品低星占比', 'TOP5销额占比', '非TOP5销额占比',
             '非TOP5月均销额', '动销品变体中位数', '平均开售月数', '冒出品新品占比', '新品平均月销额', '平均星数', '销级星数比', '三标新品占比', '月销增长率', '加权留评率',
             'P得分',
             'M得分', 'PMI得分', 'P标签', 'M标签', 'I标签', '推荐级别v2', '综合投票分', '综合推荐级别', '代表节点', '代表度', 'pmi_rank',
             'duplicate_tag',
             'duplicate_type', 'asin', '数据更新时间']]
        df_group = duplicate_util.df_cleaning(df_group, 'related_asin')

        df_group.rename(columns={'related_asin': '原ASIN',
                                 'price': '价格',
                                 'recommend': '原ASIN推荐度',
                                 'blue_ocean_estimate': '广告蓝海度',
                                 'duplicate_tag': '重复度',
                                 'duplicate_type': '重复利基',
                                 'asin': 'ASINs'}, inplace=True)

        # 聚合tag表
        df_group_tag = df_group_tag[
            ['related_asin', '相关竞品款数分布', '有销额竞品款数分布', '有销额推荐达标款数分布', '有销额竞品款数占比分布', '综合竞品推荐度分布', '达标推荐度占比分布', '推荐级别v1',
             'TOP5月均销额分布', 'TOP5月销额分布', '利基月GMV分布', '价格中位数分布', '价格集中度分布', '预估平均毛利率分布', '预估平均资金利用率分布', '加权FBA运费分布',
             'FBM配送占比分布', '直发FBM产品占比分布', '直发FBM销额占比分布', '直发FBM月均销额分布', 'AMZ直营销额占比分布', '大牌商标销额占比分布', '中国卖家占比分布',
             'TOP5平均LQS分布', '冒出品平均LQS分布', '冒出品A+占比分布', '冒出品视频占比分布', '冒出品QA占比分布', '动销品平均星级分布', '冒出品低星占比分布', 'TOP5销额占比分布',
             '非TOP5销额占比分布', '非TOP5月均销额分布', '动销品变体中位数', '平均开售月数分布', '冒出品新品占比分布', '新品平均月销额分布', '平均星数分布', '销级星数比分布',
             '三标新品占比分布', '月销增长率分布', '加权留评率分布', '推荐级别v2', '综合投票分', '综合推荐级别', '数据更新时间', 'data_id']]

        df_group_tag = duplicate_util.df_cleaning(df_group_tag, 'related_asin')

        df_group_tag.rename(columns={'related_asin': '原ASIN', '动销品变体中位数': '动销品变体中位数分布'}, inplace=True)

        print("字段整合用时：" + (time.time() - start_time).__str__())

        # 8.存入数据库
        start_time = time.time()
        sql_engine.data_to_sql(df_traffic_table, path.product_traffic_history, 'append', config.connet_product_db_sql)
        sql_engine.data_to_sql(df_group, path.product_group_history, 'append', config.connet_product_db_sql)
        sql_engine.data_to_sql(df_group_tag, path.product_group_tag_history, 'append', config.connet_product_db_sql)

        # print("page：" + page.__str__())
        id_start = id_start + id_increment
        print("id_start：" + id_start.__str__())
        print("数据入库用时：" + (time.time() - start_time).__str__())
    print("用时：" + (time.time() - start_time).__str__())

    # 父体去重
    sql_engine.connect_product(config.oe_hostname, config.oe_password, path.product_database, sql.duplicate_sql3)

    """
    # 类目利基去重
    sql_engine.connect_product(config.oe_hostname, config.oe_password, path.product_database, sql.pmi_rank_sql)
    sql_engine.connect_product(config.oe_hostname, config.oe_password, path.product_database, sql.duplicate_sql1)
    sql_engine.connect_product(config.oe_hostname, config.oe_password, path.product_database, sql.duplicate_sql2)
    
    # 历史开售产品去重
    sql_engine.connect_product(config.oe_hostname, config.oe_password, path.product_database, sql.duplicate_sql4)
    sql_engine.connect_product(config.oe_hostname, config.oe_password, path.product_database, sql.duplicate_sql5)
    sql_engine.connect_product(config.oe_hostname, config.oe_password, path.product_database, sql.duplicate_sql6)
    sql_engine.connect_product(config.oe_hostname, config.oe_password, path.product_database, sql.duplicate_sql7)
    sql_engine.connect_product(config.oe_hostname, config.oe_password, path.product_database, sql.duplicate_sql8)
    sql_engine.connect_product(config.oe_hostname, config.oe_password, path.product_database, sql.duplicate_sql9)
    sql_engine.connect_product(config.oe_hostname, config.oe_password, path.product_database, sql.duplicate_sql10)
    """
    sql_engine.connect_product(config.oe_hostname, config.oe_password, path.product_database,
                               sql.update_sql_product_group_tag)

else:
    sys.exit()
