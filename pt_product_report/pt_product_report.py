import time
import warnings

import numpy as np
import pandas as pd
from pandas.errors import SettingWithCopyWarning

import calculate_util
import calculation_util
import common_util
import data_cleaning_util
import duplicate_util
import db_util
import pt_product_report_parameter as para
import pt_product_report_path as path
import pt_product_sql as sql
import pt_table_field as field
from conn import sql_engine, mysql_config as config

# -------------------------------------------------------------------------------------------------------------------

# 忽略与 Pandas SettingWithCopyWarning 模块相关的警告
warnings.filterwarnings('ignore', category=SettingWithCopyWarning)

# 忽略与 Pandas SQL 模块相关的警告
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas.io.sql")

# 忽略除以零的警告
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in log2")

warnings.filterwarnings("ignore", message="pandas only support SQLAlchemy connectable.*")
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")

# 设置选项以选择未来行为
pd.set_option('future.no_silent_downcasting', True)

# config.sellersprite_database = 'sellersprite_202410'

# 辅助表获取
df_famous_brand = sql_engine.connect_pt_product(config.oe_hostname, config.oe_password, path.product_database,
                                                sql.famous_brand_sql)
df_holiday = sql_engine.connect_pt_product(config.oe_hostname, config.oe_password, path.product_database,
                                           sql.holiday_sql)
df_category_risk = sql_engine.connect_pt_product(config.oe_hostname, config.oe_password, path.product_database,
                                                 sql.category_risk_sql)
df_seller_self = sql_engine.connect_pt_product(config.oe_hostname, config.oe_password, path.product_database,
                                               sql.seller_self_sql)
df_brand_self = sql_engine.connect_pt_product(config.oe_hostname, config.oe_password, path.product_database,
                                              sql.brand_self_sql)

update_date = str(config.sellersprite_database)[-6:-2] + "-" + str(config.sellersprite_database)[-2:] + "-01"

sites = ['US', 'UK', 'DE', 'FR']

for site in sites:
    sellersprite_database = config.sellersprite_database + '_' + str(site).lower()

    # 线索数据去重
    sql_engine.connect_product(config.sellersprite_hostname, config.sellersprite_password, sellersprite_database,
                               sql.duplicate_sql_product_report)

    # get_cpc表创建
    sql_engine.connect_product(config.sellersprite_hostname, config.sellersprite_password, sellersprite_database,
                               sql.create_sql_product_get_cpc)
    # get_group表创建
    sql_engine.connect_product(config.sellersprite_hostname, config.sellersprite_password, sellersprite_database,
                               sql.create_sql_product_get_group)
    # pt_keywords创建
    sql_engine.connect_product(config.sellersprite_hostname, config.sellersprite_password, sellersprite_database,
                               sql.create_sql_product_pt_keywords)
    # cpc_from_keywords创建
    sql_engine.connect_product(config.sellersprite_hostname, config.sellersprite_password, sellersprite_database,
                               sql.create_sql_cpc_from_keywords)
    # duplicate表创建
    sql_engine.connect_product(config.sellersprite_hostname, config.sellersprite_password, sellersprite_database,
                               sql.create_sql_pt_product_duplicate)

    # 创建索引
    sql_engine.connect_product(config.sellersprite_hostname, config.sellersprite_password, sellersprite_database,
                               sql.create_index_sql_relevance_1)

    sql_engine.connect_product(config.sellersprite_hostname, config.sellersprite_password, sellersprite_database,
                               sql.create_index_sql_relevance_2)

    start_time = time.time()

    # 循环参数
    id_start = 0
    id_increment = 10000

    if site == 'US':
        id_end = 5000000
    else:
        id_end = 500000

    while id_start < id_end:
        # 1.数据连接
        df_product = sql_engine.connect_pt_product(config.sellersprite_hostname, config.sellersprite_password,
                                                   sellersprite_database,
                                                   db_util.report_asin_sql(id_start, id_increment))
        if df_product.empty:
            id_start += id_increment
            continue

        # 2.数据预处理
        df_product['ASIN'] = df_product['asin']
        df_product = duplicate_util.df_cleaning(df_product, 'ASIN')
        product_con_list_1 = ['category_bsr_growth', 'sales_growth', 'price', 'gross_margin', 'fba_fees']
        for con_i in product_con_list_1:
            df_product[con_i] = data_cleaning_util.convert_type(df_product, con_i, 2)

        product_con_list_2 = ['sales', 'qa', 'ratings', 'variations']
        for con_j in product_con_list_2:
            df_product[con_j] = data_cleaning_util.convert_type(df_product, con_j, 0)

        df_product['reviews_rate'] = data_cleaning_util.convert_type(df_product, 'reviews_rate', 4)
        df_product['rating'] = data_cleaning_util.convert_type(df_product, 'rating', 1)

        product_con_list_3 = ['brand', 'title', 'category_path', 'category', 'sub_category', 'ac_keyword', 'weight',
                              '二级类目']
        for con_l in product_con_list_3:
            df_product[con_l] = data_cleaning_util.convert_str_lower(df_product, con_l)

        df_product['monthly_revenue_increase'] = data_cleaning_util.convert_type(df_product, 'sales_growth', 4)

        df_product['date_available'] = data_cleaning_util.convert_date(df_product, 'date_available')
        df_product['sync_time'] = data_cleaning_util.convert_date(df_product, 'sync_time')

        df_famous_brand['brand'] = data_cleaning_util.convert_str_lower(df_famous_brand, 'brand')
        df_famous_brand['疑似知名品牌'] = data_cleaning_util.convert_type(df_famous_brand, '疑似知名品牌', 0)
        df_holiday['节日关键词'] = data_cleaning_util.convert_str_lower(df_holiday, '节日关键词')
        df_category_risk['category_path'] = data_cleaning_util.convert_str_lower(df_category_risk, 'category_path')
        df_category_risk['prohibited_risk'] = data_cleaning_util.convert_type(df_category_risk, 'prohibited_risk', 0)

        # 3.M相关指标计算
        df_product_weight = df_product[df_product['weight'].notnull()]

        if not df_product_weight.empty:
            df_product_weight['重量(g)'] = calculate_util.weight_g(df_product_weight)
        else:
            # 如果DataFrame为空，创建空的DataFrame并设置重量列为NaN
            df_product_weight = pd.DataFrame(columns=df_product.columns)
            df_product_weight['重量(g)'] = np.nan

        # 直发FBM可能性
        df_product_fbm = calculate_util.get_fbm(df_product_weight)

        df_product = df_product.merge(df_product_fbm, how='left', on='id')
        df_product_fbm['直发FBM可能性'] = df_product_fbm['直发FBM可能性'].fillna(0)

        df_product['预估FBA占比'] = calculate_util.fba_rate(df_product)

        df_product['预估头程占比'] = calculate_util.pre_rate(df_product)

        df_product['预估货值占比'] = calculate_util.product_rate(df_product)

        df_product['预估货值占比'] = data_cleaning_util.convert_type(df_product, '预估货值占比', 2)

        df_product['预估毛利率'] = calculate_util.profit_rate(df_product, site)

        df_product['毛利率级别'] = calculate_util.profit_rate_tag(df_product)

        df_product['毛估资金利用率'] = calculate_util.product_revenue(df_product)

        # 4.推荐度相关指标计算
        # M相关指标
        df_product['高资金利用率'] = calculate_util.high_product_revenue(df_product)

        # I相关指标
        df_product['开售月数'] = calculate_util.month_available(df_product)

        # S相关指标
        df_product['销额级数'] = calculate_util.get_revenue(df_product, site)

        df_product['高销低LQS'] = calculate_util.high_sale_low_lqs(df_product)

        df_product['月均QA数'] = calculate_util.qa_per_month(df_product)

        df_product[['长期上架少Q&A', '长期上架无A+', '长期上架无视频']] = calculate_util.long_term_sale(df_product, site)

        df_product['类轻小直发FBM'] = calculate_util.light_small_fbm(df_product)

        df_product['差评好卖'] = calculate_util.low_star_high_sale(df_product, site)

        # 知名品牌，疑似节日性
        df_product = df_product.merge(df_famous_brand, how='left', on='brand')

        df_product['combined_kw'] = df_product['title'] + " " + df_product['sub_category'] + " " + df_product[
            'ac_keyword']

        df_product[['疑似节日性', '节日名']] = df_product.apply(
            lambda row: pd.Series(calculate_util.match_holidays(row, df_holiday['节日关键词'])), axis=1)

        df_product['知名品牌'] = calculate_util.famous_brand(df_product)

        df_product['疑似节日性'] = np.where(df_product['疑似节日性'] > 3, "3+", df_product['疑似节日性'])

        # 是否个人定制
        df_product['custom_kw'] = df_product.apply(lambda row: pd.Series(calculate_util.match_custom_kw(row)), axis=1)
        df_product['是否个人定制'] = np.where(df_product['custom_kw'] > 0, 1, 0)

        # 是否翻新
        df_product['是否翻新'] = calculate_util.get_renewed(df_product)

        # L相关指标
        df_product['平均变体月销额等级'] = calculate_util.revenue_per_variations(df_product, site)
        df_product['变体等级'] = calculate_util.variations_tag(df_product)
        df_product['少变体'] = calculate_util.few_variations(df_product)

        # E相关指标
        df_product[['新品爬坡快', '新品增评好', '新品NSR', '新品AC标']] = calculate_util.product_new(df_product, site)

        df_product['少评好卖'] = calculate_util.few_star_high_sale(df_product)

        # 推荐度计算
        df_recommend = df_product[
            ['id', '销额级数', '高资金利用率', '高销低LQS', '长期上架少Q&A', '长期上架无A+', '长期上架无视频', '类轻小直发FBM', '差评好卖', '知名品牌', '少变体',
             '新品爬坡快', '新品增评好', '新品NSR', '新品AC标', '少评好卖']]

        df_recommend = df_recommend.astype(float)
        recommend_weights = para.recommend_weights.astype(float)

        df_recommend['推荐度'] = df_recommend.dot(recommend_weights)
        df_recommend['推荐度'] = data_cleaning_util.convert_type(df_recommend, '推荐度', 1)

        df_recommend = df_recommend[['id', '推荐度']]
        df_product = df_product.merge(df_recommend, how='left', on='id')

        # 数据更新日期
        df_product['数据更新时间'] = update_date
        df_product['数据更新时间'] = data_cleaning_util.convert_date(df_product, '数据更新时间')

        # 一级类目清洗
        df_product['category'] = df_product['category'].replace(para.replace_category_dict)

        # 二级类目清洗
        df_product['二级类目'] = df_product['二级类目'].replace(para.replace_category_dict_2)

        # 剔除类目
        df_product.loc[
            df_product['category_path'].str.contains(para.regex_pattern_kw, na=False, regex=True), '剔除类目'] = 1
        df_product['剔除类目'] = np.where(df_product['剔除类目'] == 1, 1, 0)

        # 按类目风险剔除
        df_product = df_product.merge(df_category_risk, how='left', on='category_path')
        df_product['prohibited_risk'] = df_product['prohibited_risk'].fillna(0)
        df_product = df_product[df_product['prohibited_risk'] <= 7]

        # 排除公司内店铺
        df_product = df_product.merge(df_seller_self, how='left', on='buybox_seller')
        df_product['seller_status'] = df_product['seller_status'].fillna(0)
        df_product = df_product[df_product['seller_status'] < 1]

        # 排除公司内品牌
        df_product = df_product.merge(df_brand_self, how='left', on='brand')
        df_product['brand_status'] = df_product['brand_status'].fillna(0)
        df_product = df_product[df_product['brand_status'] < 1]

        # 数据格式整理
        product_con_list = ['预估FBA占比', '预估头程占比', '预估毛利率', '毛估资金利用率', '销额级数', '高资金利用率', '高销低LQS', '类轻小直发FBM',
                            '平均变体月销额等级', '新品爬坡快']
        for con_k in product_con_list:
            df_product[con_k] = data_cleaning_util.convert_type(df_product, con_k, 4)

        # 站点添加
        df_product['site'] = str(site)

        # 5.添加数据标签
        df_product_tag = df_product[field.df_product_tag_pre_list]

        df_product_tag['data_id'] = df_product_tag['ASIN'] + " | " + df_product_tag['site'] + " | " + update_date

        product_tag_list = ['预估FBA占比', '预估毛利率', '毛估资金利用率', '少评好卖']
        for tag in product_tag_list:
            tag_col = tag + '分布'
            df_product_tag[tag_col] = calculation_util.get_mround(df_product_tag, tag, tag_col, 0.05)

        df_product_tag['价格分布'] = round(df_product_tag['price'])
        df_product_tag['开售月数分布'] = calculation_util.get_mround(df_product_tag, '开售月数', '开售月数分布', 3)
        df_product_tag['推荐度分布'] = calculation_util.get_mround(df_product_tag, '推荐度', '推荐度分布', 0.5)
        df_product_tag['销额级数分布'] = calculation_util.get_mround(df_product_tag, '销额级数', '销额级数分布', 0.1)
        df_product_tag['评分数分布'] = calculation_util.get_mround(df_product_tag, 'ratings', '评分数分布', 100)

        # 6.筛选可供爬取CPC的线索
        df_product_cpc = df_product.query('推荐度>= 1.5')

        df_product_cpc = df_product_cpc[['ASIN', 'site', 'price', '推荐度', '疑似知名品牌', '疑似节日性', '剔除类目']]

        conditions_cpc = (df_product_cpc['疑似知名品牌'].isnull())

        df_product_cpc = df_product_cpc[conditions_cpc]

        # 7.字段整合
        df_product_table = df_product[field.df_product_table_list]

        df_product_table = duplicate_util.df_cleaning(df_product_table, 'ASIN')

        df_product_table.rename(columns=field.df_product_table_rename_dict, inplace=True)

        df_product_recommend_modify = df_product_tag[['ASIN', '销额级数', '推荐度', '推荐度分布', '销额级数分布', '数据更新时间']]
        df_product_recommend_modify = duplicate_util.df_cleaning(df_product_recommend_modify, 'ASIN')

        df_product_tag = df_product_tag[field.df_product_tag_list]

        df_product_tag = duplicate_util.df_cleaning(df_product_tag, 'ASIN')

        df_product_tag.rename(columns=field.df_product_tag_rename_dict, inplace=True)

        df_product_cpc = df_product_cpc[['ASIN', 'site', 'price', '推荐度']]
        df_product_cpc = duplicate_util.df_cleaning(df_product_cpc, 'ASIN')

        df_product_cpc.rename(columns={'ASIN': 'asin', 'site': 'country', '推荐度': 'recommend'}, inplace=True)

        # 8.存入数据库
        sql_engine.data_to_sql(df_product_cpc, path.pt_product_get_cpc, 'append',
                               db_util.connet_sellersprite_db_sql(sellersprite_database))

        sql_engine.data_to_sql(df_product_table, path.product_table_history, 'append', config.connet_product_db_sql)
        sql_engine.data_to_sql(df_product_tag, path.product_tag_history, 'append', config.connet_product_db_sql)

        id_start += id_increment
        print("id_start：" + id_start.__str__())
        print("用时：" + (time.time() - start_time).__str__())

    print("用时：" + (time.time() - start_time).__str__())
