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
import pt_product_report_parameter as para
import pt_product_report_path as path
import pt_product_sql as sql
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

# 辅助表获取
df_famous_brand = sql_engine.connect_pt_product(config.oe_hostname, config.oe_password, path.product_database,
                                                sql.famous_brand_sql)
df_holiday = sql_engine.connect_pt_product(config.oe_hostname, config.oe_password, path.product_database,
                                           sql.holiday_sql)

start_time = time.time()

sites = ['US', 'UK', 'DE', 'FR']

for site in sites:

    # 1.数据连接
    df_product = sql_engine.connect_pt_product(config.sellersprite_hostname, config.sellersprite_password,
                                               config.clue_self_database, sql.clue_sql)

    df_product = df_product[df_product['site'] == site]

    if df_product.empty:
        print('df_product.empty')
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

    product_con_list_3 = ['brand', 'title', 'category_path', 'category', 'sub_category', 'ac_keyword', 'weight', '二级类目']
    for con_l in product_con_list_3:
        df_product[con_l] = data_cleaning_util.convert_str_lower(df_product, con_l)

    product_con_list_4 = ['date_available', 'sync_time', '数据更新时间']
    for con_u in product_con_list_4:
        df_product[con_u] = data_cleaning_util.convert_date(df_product, con_u)

    df_product['monthly_revenue_increase'] = df_product['sales_growth']
    data_cleaning_util.convert_type(df_product, 'monthly_revenue_increase', 4)

    # 辅助表
    data_cleaning_util.convert_str_lower(df_famous_brand, 'brand')
    data_cleaning_util.convert_type(df_famous_brand, '疑似知名品牌', 0)
    data_cleaning_util.convert_str_lower(df_holiday, '节日关键词')

    # 3.M相关指标计算
    df_product_weight = df_product[df_product['weight'].notnull()]

    if not df_product_weight.empty:
        # 替换错误单位
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
        ['id', '销额级数', '高资金利用率', '高销低LQS', '长期上架少Q&A', '长期上架无A+', '长期上架无视频', '类轻小直发FBM', '差评好卖', '知名品牌', '少变体', '新品爬坡快',
         '新品增评好', '新品NSR', '新品AC标', '少评好卖']]

    df_recommend = df_recommend.astype(float)
    recommend_weights = para.recommend_weights.astype(float)

    df_recommend['推荐度'] = df_recommend.dot(recommend_weights)
    df_recommend['推荐度'] = data_cleaning_util.convert_type(df_recommend, '推荐度', 1)

    df_recommend = df_recommend[['id', '推荐度']]
    df_product = df_product.merge(df_recommend, how='left', on='id')

    # 一级类目清洗
    df_product['category'] = df_product['category'].replace(para.replace_category_dict)

    # 二级类目清洗
    df_product['二级类目'] = df_product['二级类目'].replace(para.replace_category_dict_2)

    # 剔除类目
    df_product.loc[df_product['category_path'].str.contains(para.regex_pattern_kw, na=False, regex=True), '剔除类目'] = 1
    df_product['剔除类目'] = np.where(df_product['剔除类目'] == 1, 1, 0)

    # 数据格式整理
    product_con_list = ['预估FBA占比', '预估头程占比', '预估毛利率', '毛估资金利用率', '销额级数', '高资金利用率', '高销低LQS', '类轻小直发FBM', '平均变体月销额等级',
                        '新品爬坡快']
    for con_k in product_con_list:
        data_cleaning_util.convert_type(df_product, con_k, 4)

    # 5.添加数据标签
    df_product_tag = df_product[
        ['ASIN', 'country', 'price', 'category', 'seller_type', '直发FBM可能性', '预估FBA占比', '预估毛利率', '毛估资金利用率', '开售月数',
         '月均QA数', '疑似知名品牌', '疑似节日性', '节日名', '是否个人定制', '是否翻新', '推荐度', '销额级数', 'lqs', 'qa', 'ebc_available',
         'video_available', 'rating', '知名品牌', 'variations', 'ratings', 'new_release', 'ac', '少评好卖', '二级类目', '剔除类目',
         '数据更新时间']]

    df_product_tag['data_id'] = df_product_tag['ASIN'] + " | " + pd.to_datetime(df_product_tag['数据更新时间']).dt.strftime(
        '%Y-%m-%d')

    product_tag_list = ['预估FBA占比', '预估毛利率', '毛估资金利用率', '少评好卖']
    for tag in product_tag_list:
        tag_col = tag + '分布'
        df_product_tag[tag_col] = calculation_util.get_mround(df_product_tag, tag, tag_col, 0.05)

    df_product_tag['价格分布'] = round(df_product_tag['price'])
    df_product_tag['开售月数分布'] = calculation_util.get_mround(df_product_tag, '开售月数', '开售月数分布', 3)
    df_product_tag['推荐度分布'] = calculation_util.get_mround(df_product_tag, '推荐度', '推荐度分布', 0.5)
    df_product_tag['销额级数分布'] = calculation_util.get_mround(df_product_tag, '销额级数', '销额级数分布', 0.1)
    df_product_tag['评分数分布'] = calculation_util.get_mround(df_product_tag, 'ratings', '评分数分布', 100)

    # 6.字段整合
    df_product_table = df_product[['ASIN',
                                   'country',
                                   'sku',
                                   'brand',
                                   'brand_link',
                                   'title',
                                   'image',
                                   'parent',
                                   'category_path',
                                   'category',
                                   'category_bsr',
                                   'category_bsr_growth',
                                   'sub_category',
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
                                   'sync_time',
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
                                   '少评好卖',
                                   '是否个人定制',
                                   '是否翻新',
                                   '推荐度',
                                   '二级类目',
                                   '剔除类目',
                                   '数据更新时间']]

    df_product_table = duplicate_util.df_cleaning(df_product_table, 'ASIN')

    df_product_table.rename(columns={'country': 'site',
                                     'sku': 'SKU',
                                     'brand': '品牌',
                                     'brand_link': '品牌链接',
                                     'title': '标题',
                                     'image': '主图',
                                     'parent': '父体',
                                     'category_path': '类目路径',
                                     'category': '大类目',
                                     'category_bsr': '大类BSR',
                                     'category_bsr_growth': '大类BSR增长率',
                                     'sub_category': '小类目',
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
                                     'sync_time': '最近更新'}, inplace=True)

    df_product_tag = df_product_tag[
        ['ASIN', 'country', '价格分布', 'category', 'seller_type', '直发FBM可能性', '预估FBA占比分布', '预估毛利率分布', '毛估资金利用率分布',
         '开售月数分布', '月均QA数', '疑似知名品牌', '疑似节日性', '节日名', '推荐度分布', '销额级数分布', 'lqs', 'qa', 'ebc_available',
         'video_available', 'rating', '知名品牌', 'variations', '评分数分布', 'new_release', 'ac', '少评好卖分布', '是否个人定制', '是否翻新',
         '剔除类目', '数据更新时间', 'data_id']]

    df_product_tag = duplicate_util.df_cleaning(df_product_tag, 'ASIN')

    df_product_tag.rename(columns={'country': 'site',
                                   'category': '一级类目',
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

    df_product_cpc = df_product[['ASIN', 'country', 'price', '推荐度', '数据更新时间']]
    df_product_cpc = duplicate_util.df_cleaning(df_product_cpc, 'ASIN')

    df_product_cpc.rename(columns={'ASIN': 'asin', '推荐度': 'recommend', '数据更新时间': 'update_time'}, inplace=True)

    # 8.存入数据库
    sql_engine.data_to_sql(df_product_cpc, path.pt_product_get_cpc, 'append', config.connet_clue_self_db_sql)

    sql_engine.data_to_sql(df_product_table, path.product_report_self, 'append', config.connet_product_db_sql)

    sql_engine.data_to_sql(df_product_tag, path.product_tag_self, 'append', config.connet_product_db_sql)

    # 状态更新
    # sql_engine.connect_product(
    #     config.sellersprite_hostname,
    #     config.sellersprite_password,
    #     config.clue_self_database,
    #     sql.update_sql_product_get_cpc
    # )

    sql_engine.connect_product(
        config.sellersprite_hostname,
        config.sellersprite_password,
        config.clue_self_database,
        sql.update_clue_sql
    )
    print("用时：" + (time.time() - start_time).__str__())
