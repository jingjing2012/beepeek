import sys
import warnings
from datetime import datetime
import time
import numpy as np
import pandas as pd
from pandas.errors import SettingWithCopyWarning

from conn import sql_engine, mysql_config as config
from util import data_cleaning_util, calculation_util, common_util
import pt_product_report_parameter as para
import pt_product_report_path as path
import pt_product_sql as sql


# 直发FBM可能性
def get_fbm(df):
    df_fbm = df[df['buybox_location'].notnull()]
    df_fbm['按毛利推测FBM可能性'] = np.where(df_fbm['gross_margin'] >= para.gross_margin_upper, 2, 1)
    df_fbm['中国卖家FBM可能性'] = np.where(df_fbm['buybox_location'] == "CN", df_fbm['按毛利推测FBM可能性'], 0)

    conditions_fbm_1 = (df_fbm['seller_type'] == "FBM") & (df_fbm['buybox_location'] != "US") & (
            df_fbm['buybox_location'] != "") & (df_fbm['gross_margin'] >= para.gross_margin_lower)
    conditions_fbm_2 = (df_fbm['fba_fees'] * 1 > 0) | (df_fbm['重量(g)'] <= 2500)
    conditions_fbm_3 = (df_fbm['fba_fees'] <= para.fba_fees_upper) | (
            df_fbm['gross_margin'] >= para.gross_margin_upper)
    df_fbm['直发FBM可能性'] = np.where(conditions_fbm_1 & conditions_fbm_2 & conditions_fbm_3, 1 + df_fbm['中国卖家FBM可能性'], 0)
    df_fbm = df_fbm[['id', '重量(g)', '直发FBM可能性']]
    return df_fbm


# 检查匹配关键词，计算总数并生成节日名字符串
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
    current_date = pd.to_datetime(datetime.now().date())
    df['date_available'] = pd.to_datetime(df['date_available'], errors='coerce')
    df['开售天数'] = (current_date - df['date_available']).dt.days
    # df['date_available'] = np.where(df['开售天数'] * 1 > 0, df['date_available'], '1900-01-01')
    df['头程月数'] = np.where((df['seller_type'] == "FBA") & (df['开售天数'] > 15), 0.5, 0)
    df['开售月数'] = np.fmax(round(df['开售天数'] / 30 - df['头程月数'], 1), 0.1)
    return df


# 销额级数计算
def get_revenue(df):
    df['monthly_revenue_increase'] = df['monthly_revenue_increase'].fillna(0)
    df['近两月销额'] = np.where(df['monthly_revenue_increase'] <= (-1), np.nan,
                           df['monthly_revenue'] + (df['monthly_revenue'] / (1 + df['monthly_revenue_increase'])))
    df['月均销额'] = np.where(df['近两月销额'] * 1 > 0, df['近两月销额'] / np.fmax(np.fmin(df['开售月数'] - 1, 1), 0.5), np.nan)
    df['销额级数'] = np.where(df['月均销额'] * 1 > 0, np.log2(df['月均销额'] / 2 / (para.monthly_revenue_C / 2)), np.nan)
    return df


def df_clear(df):
    df.replace(to_replace=[None], value='', inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.drop_duplicates(subset=['ASIN'])
    df = df.dropna(subset=['ASIN'])
    return df


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

start_time = time.time()

# 1.数据连接
df_product = sql_engine.connect_pt_product(config.sellersprite_hostname, config.sellersprite_password,
                                           config.clue_self_database, sql.clue_sql)
# 辅助表获取
df_famous_brand = sql_engine.connect_pt_product(config.oe_hostname, config.oe_password, path.product_database,
                                                sql.famous_brand_sql)
df_holiday = sql_engine.connect_pt_product(config.oe_hostname, config.oe_password, path.product_database,
                                           sql.holiday_sql)

if df_product.empty:
    print('df_product.empty')
    sys.exit()

# 2.数据预处理
df_product['ASIN'] = df_product['asin']
df_product = df_clear(df_product)
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
    df_product[con_l] = data_cleaning_util.convert_str(df_product, con_l)

product_con_list_4 = ['date_available', 'sync_time', '数据更新时间']
for con_u in product_con_list_4:
    df_product[con_u] = data_cleaning_util.convert_date(df_product, con_u)

# df_product['monthly_revenue_increase'] = pd.to_numeric(df_product['monthly_revenue_increase'].str.rstrip('%'),
#                                                        errors='coerce') / 100
df_product['monthly_revenue_increase'] = df_product['sales_growth']
data_cleaning_util.convert_type(df_product, 'monthly_revenue_increase', 4)

# 辅助表
data_cleaning_util.convert_str(df_famous_brand, 'brand')
data_cleaning_util.convert_type(df_famous_brand, '疑似知名品牌', 0)
data_cleaning_util.convert_str(df_holiday, '节日关键词')

# 3.M相关指标计算
df_product_weight = df_product[df_product['weight'].notnull()]

if not df_product_weight.empty:
    # 替换错误单位
    for error_unit, replacement in para.replace_error_dict.items():
        df_product_weight['weight'] = df_product_weight['weight'].str.replace(error_unit, replacement, regex=False)

    # 一次性分割并创建新列
    weight_split = df_product_weight['weight'].str.split(" ", expand=True)
    df_product_weight['重量值'] = weight_split[0]
    df_product_weight['单位'] = weight_split[1]

    # 一次性分割并创建新列
    weight_split = df_product_weight['weight'].str.split(" ", expand=True)
    df_product_weight['重量值'] = weight_split[0]
    df_product_weight['单位'] = weight_split[1]

    # 去除不合法单位和重量值
    df_product_weight.loc[~df_product_weight['单位'].isin(para.replace_weight_unit_list), '单位'] = np.nan
    df_product_weight['重量值判断'] = df_product_weight['重量值'].str.replace(".", "")
    df_product_weight.loc[~df_product_weight['重量值判断'].str.isdecimal(), '重量值'] = "-1"
    df_product_weight['重量值'] = np.where(df_product_weight['重量值判断'] == "-1", np.nan, df_product_weight['重量值'])

    # 计算换算值
    df_product_weight['换算'] = df_product_weight['单位'].replace(para.replace_dict, regex=False)

    # 计算重量
    df_product_weight['重量(g)'] = np.where(df_product_weight['重量值'].astype(float) * 1 > 0,
                                          round(df_product_weight['重量值'].astype(float) * df_product_weight[
                                              '换算'].astype(float), 4), np.nan)
else:
    # 如果DataFrame为空，创建空的DataFrame并设置重量列为NaN
    df_product_weight = pd.DataFrame(columns=df_product.columns)
    df_product_weight['重量(g)'] = np.nan

# 直发FBM可能性
df_product_fbm = get_fbm(df_product_weight)

df_product = df_product.merge(df_product_fbm, how='left', on='id')
df_product_fbm['直发FBM可能性'].fillna(0)

df_product['预估FBA占比'] = np.where(df_product['fba_fees'] * 1 > 0,
                                 np.fmin(1, df_product['fba_fees'] / df_product['price']), para.fba_fees_rate)
df_product['预估头程占比'] = np.where(df_product['预估FBA占比'] * 1 > 0, np.fmin(1, df_product['预估FBA占比'] / 2.5),
                                para.pre_fees_rate)
df_product['预估货值占比'] = common_util.get_cut(df_product, 'price', '预估货值占比', [0, 6, 10, 15, 30, 50, 100, 200, 9999],
                                           [0.08, 0.1, 0.15, 0.2, 0.25, 0.27, 0.3, 0.35])
df_product['预估货值占比'] = data_cleaning_util.convert_type(df_product, '预估货值占比', 2)

df_product['预估毛利率_FBM'] = df_product['gross_margin'] - df_product['预估头程占比'] * 2 - para.product_fees_rate
df_product['预估毛利率_FBA'] = df_product['gross_margin'] - df_product['预估头程占比'] - para.product_fees_rate
df_product['预估毛利率_反推'] = np.where(
    (df_product['直发FBM可能性'] >= 1) & (df_product['gross_margin'] >= para.gross_margin_upper),
    df_product['预估毛利率_FBM'], df_product['预估毛利率_FBA'])

df_product['预估毛利率'] = np.where(abs(df_product['gross_margin'] * 1) > 0,
                               np.fmax(-1, np.fmin(1, df_product['预估毛利率_反推'])),
                               np.fmax(-1, np.fmin(1, 1 - df_product['预估FBA占比'] - df_product['预估头程占比'] -
                                                   para.referral_fees_rate - df_product['预估货值占比'])))
df_product['毛利率级别_上限'] = np.fmin(calculation_util.get_mround(df_product, '预估毛利率', '毛利率级别_上限', 0.05),
                                 para.gross_rate_upper)
df_product['毛利率级别_下限'] = np.fmax(calculation_util.get_mround(df_product, '预估毛利率', '毛利率级别_下限', -0.05),
                                 para.gross_rate_lower)
df_product['毛利率级别'] = np.where(df_product['预估毛利率'] >= 0, df_product['毛利率级别_上限'], df_product['毛利率级别_下限'])

df_product['毛估资金利用率'] = df_product['预估毛利率'] / (df_product['预估头程占比'] + para.product_fees_rate)

# 4.推荐度相关指标计算
# M相关指标
df_product['高资金利用率'] = np.where(abs(df_product['毛估资金利用率'] * 1) > 0,
                                df_product['毛估资金利用率'] / para.product_revenue_std - 1, 0)

# I相关指标
# 开售月数
month_available(df_product)

# S相关指标
get_revenue(df_product)

conditions_lqs_1 = (df_product['预估毛利率'] >= -0.05) & (df_product['lqs'] * 1 > 0) & (df_product['lqs'] <= 8)
conditions_lqs_2 = (df_product['开售月数'] >= 24) & (df_product['rating'] >= 4) & (df_product['ratings'] >= 10) & (
        df_product['预估毛利率'] >= -0.15) & (df_product['lqs'] * 1 > 0) & (df_product['lqs'] <= 8)
df_product['高销低LQS_pre'] = np.fmin(3, 0.5 + df_product['销额级数'] * para.lqs_std / df_product['lqs'])
df_product['高销低LQS'] = np.where(conditions_lqs_1 | conditions_lqs_2, df_product['高销低LQS_pre'], 0)

df_product['开售月数_QA'] = np.fmin(df_product['开售月数'], 24)
df_product['月均QA数'] = np.where(df_product['qa'] * 1 > 0, round(df_product['qa'] / df_product['开售月数_QA'], 1), 0)

conditions_available = (df_product['开售月数'] >= para.available_std) & (
        df_product['monthly_revenue'] >= para.monthly_revenue_C)

df_product['长期上架少Q&A'] = np.where(conditions_available,
                                  np.fmax(-1, para.qa_std - df_product['月均QA数'] / para.qa_std), 0)

df_product['长期上架无A+'] = np.where(conditions_available & (df_product['ebc_available'] != "Y"), 1, 0)

df_product['长期上架无视频'] = np.where(conditions_available & (df_product['video_available'] != "Y"), 1, 0)

df_product['类轻小直发FBM'] = np.where(df_product['直发FBM可能性'] * 1 > 0,
                                  np.fmax(0, 1 + df_product['销额级数'] * np.fmin(1, df_product['直发FBM可能性'] / 2)), 0)

df_product['差评好卖'] = np.where((df_product['开售月数'] >= para.available_std) & (
        df_product['monthly_revenue'] >= para.monthly_revenue_C / 2) & (df_product['ratings'] >= 10) & (
                                      df_product['rating'] >= 3) & (df_product['rating'] < 4) & (
                                      abs(df_product['预估毛利率'] * 1) > 0) & (df_product['category_bsr_growth'] >= -0.5),
                              0.5 + df_product['销额级数'] * (4.5 - df_product['rating']), 0)

df_product = df_product.merge(df_famous_brand, how='left', on='brand')

df_product['combined_kw'] = df_product['title'] + "" + df_product['sub_category'] + "" + df_product['ac_keyword']

df_product[['疑似节日性', '节日名']] = df_product.apply(lambda row: pd.Series(match_holidays(row)), axis=1)

df_product['知名品牌'] = np.where(df_product['疑似知名品牌'] * 1 > 0,
                              -df_product['疑似知名品牌'] / np.where(df_product['疑似节日性'] * 1 > 0, 2, 1),
                              0)
df_product['知名品牌'].fillna(0)

df_product['疑似节日性'] = np.where(df_product['疑似节日性'] > 3, "3+", df_product['疑似节日性'])

# 是否个人定制
custom_kw = ['custom', 'personalize', 'personalized', 'custom-made', 'customized', 'made-to-order']

df_product['custom_kw'] = df_product.apply(lambda row: pd.Series(match_custom_kw(row)), axis=1)
df_product['是否个人定制'] = np.where(df_product['custom_kw'] * 1 > 0, 1, 0)

# 是否翻新
df_product.loc[df_product['title'].astype(str).str.contains('renewed', na=False, regex=True), '是否翻新'] = 1
df_product['是否翻新'] = np.where(df_product['是否翻新'] == 1, 1, 0)

# L相关指标
df_product['平均变体月销额等级'] = np.where(df_product['monthly_revenue'] * 1 > 0, np.log2(
    df_product['monthly_revenue'] / df_product['variations'] / (para.monthly_revenue_C / 2)), 0)
df_product['变体等级'] = np.log2(df_product['variations'])
df_product['少变体'] = np.where(df_product['variations'] == 1, 0,
                             np.fmax(-10, np.fmin(0, df_product['平均变体月销额等级'] - 0.5 * df_product['变体等级'] + 0.5)))

# E相关指标
conditions_new_product = df_product['开售月数'] < para.available_std

df_product['新品爬坡快'] = np.where(conditions_new_product & (df_product['monthly_revenue'] >= 0), np.log2(
    df_product['monthly_revenue'] / df_product['开售月数'] / (para.monthly_revenue_C / para.revenue_month_C)),
                               0)

df_product['新品增评好'] = np.where(conditions_new_product & (df_product['rating'] >= 4) & (df_product['ratings'] >= 10),
                               np.fmax(0, np.fmin(2, np.log(
                                   df_product['ratings'] / df_product['开售月数'] / df_product['variations'] / np.log(
                                       5)))), 0)

df_product['新品NSR'] = np.where(conditions_new_product & (df_product['new_release'] == "Y"), 1, 0)

df_product['新品AC标'] = np.where(conditions_new_product & (df_product['ac'] == "Y"), 1, 0)

df_product['销级星数比'] = np.where(df_product['rating'] * 1 > 0,
                               df_product['销额级数'] / round(2 + df_product['rating'] / 100), 0)

df_product['少评好卖'] = np.where(df_product['销额级数'] * 1 > 0, np.fmax(-1, round(df_product['销级星数比'] - 1, 2)), 0)

# 推荐度计算
df_recommend = df_product[
    ['id', '销额级数', '高资金利用率', '高销低LQS', '长期上架少Q&A', '长期上架无A+', '长期上架无视频', '类轻小直发FBM', '差评好卖', '知名品牌', '少变体', '新品爬坡快',
     '新品增评好', '新品NSR', '新品AC标', '少评好卖']]
recommend_weights = np.array([0, 0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 1.5, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5])

df_recommend = df_recommend.astype(float)
recommend_weights = recommend_weights.astype(float)

df_recommend['推荐度'] = df_recommend.dot(recommend_weights)
data_cleaning_util.convert_type(df_recommend, '推荐度', 1)

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
    ['ASIN', 'price', 'category', 'seller_type', '直发FBM可能性', '预估FBA占比', '预估毛利率', '毛估资金利用率', '开售月数', '月均QA数',
     '疑似知名品牌', '疑似节日性', '节日名', '是否个人定制', '是否翻新', '推荐度', '销额级数', 'lqs', 'qa', 'ebc_available', 'video_available',
     'rating', '知名品牌', 'variations', 'ratings', 'new_release', 'ac', '少评好卖', '二级类目', '剔除类目', '数据更新时间']]

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

df_product_table = df_clear(df_product_table)

df_product_table.rename(columns={'sku': 'SKU',
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
    ['ASIN', '价格分布', 'category', 'seller_type', '直发FBM可能性', '预估FBA占比分布', '预估毛利率分布', '毛估资金利用率分布', '开售月数分布',
     '月均QA数', '疑似知名品牌', '疑似节日性', '节日名', '推荐度分布', '销额级数分布', 'lqs', 'qa', 'ebc_available', 'video_available',
     'rating', '知名品牌', 'variations', '评分数分布', 'new_release', 'ac', '少评好卖分布', '是否个人定制', '是否翻新', '剔除类目', '数据更新时间',
     'data_id']]

df_product_tag = df_clear(df_product_tag)

df_product_tag.rename(columns={'category': '一级类目',
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

df_product_cpc = df_product[['ASIN', 'price', '推荐度', '数据更新时间']]
df_product_cpc = df_clear(df_product_cpc)

df_product_cpc.rename(columns={'ASIN': 'asin', '推荐度': 'recommend', '数据更新时间': 'update_time'}, inplace=True)

# 8.存入数据库
sql_engine.data_to_sql(df_product_cpc, path.pt_product_get_cpc, 'append', config.connet_clue_self_db_sql)
sql_engine.data_to_sql(df_product_table, path.product_report_self, 'append', config.connet_product_db_sql)
sql_engine.data_to_sql(df_product_tag, path.product_tag_self, 'append', config.connet_product_db_sql)

sql_engine.connect_product(config.sellersprite_hostname, config.sellersprite_password, config.clue_self_database,
                           sql.update_sql_product_get_cpc)
sql_engine.connect_product(config.sellersprite_hostname, config.sellersprite_password, config.clue_self_database,
                           sql.update_clue_sql)
print("用时：" + (time.time() - start_time).__str__())
