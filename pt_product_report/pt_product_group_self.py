import sys
import warnings

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning, SettingWithCopyWarning

import calculate_util
import calculation_util
import common_util
import data_cleaning_util
import db_util
import duplicate_util
import pt_product_report_parameter as para
import pt_product_report_path as path
import pt_product_sql as sql
from conn import sql_engine, mysql_config as config


# 数据类型修正
def convert_type(df, con_str, d):
    df[con_str] = df[con_str].replace('', np.nan)
    df[con_str] = df[con_str].astype('float64').round(decimals=d)
    return df[con_str]


# 字符串类型修正
def convert_str(df, con_str):
    df[con_str] = df[con_str].astype(str)
    df[con_str] = df[con_str].str.strip()
    return df[con_str]


# 日期修正
def convert_date(df, df_str):
    df[df_str] = pd.to_datetime(df[df_str], errors='coerce', format='%Y-%m-%d')
    return df[df_str]


# mround函数实现
def get_mround(df, col_str, mround_str, mround_n):
    df[mround_str] = round(df[col_str] / mround_n) * mround_n
    return df[mround_str]


# 百分比转换
def percent_convert(df, df_str):
    df[df_str] = pd.to_numeric(df[df_str].str.rstrip('%'), errors='coerce') / 100
    return convert_type(df, df_str, 4)


# 数据打标签
def get_cut(df, col_str, bins_cut, labels_cut):
    return pd.cut(df[col_str], bins_cut, right=False, labels=labels_cut, include_lowest=True)


# series转DataFrame
def convert_col(series, col):
    df = pd.DataFrame(series)
    df.columns = [col]
    return df


# 直发FBM可能性
def get_fbm(df):
    df_fbm = df[df['buybox_location'].notnull()]
    df_fbm['按毛利推测FBM可能性'] = np.where(df_fbm['gross_margin'] >= para.gross_margin_upper, 2, 1)
    df_fbm['中国卖家FBM可能性'] = np.where(df_fbm['buybox_location'] == "CN", df_fbm['按毛利推测FBM可能性'], 0)

    conditions_fbm_1 = (df_fbm['seller_type'] == "FBM") & (df_fbm['buybox_location'] != "US") & (
            df_fbm['buybox_location'] != "") & (df_fbm['gross_margin'] >= para.gross_margin_lower)
    conditions_fbm_2 = (df_fbm['fba_fees'] > 0) | (df_fbm['重量(g)'] <= 2500)
    conditions_fbm_3 = (df_fbm['fba_fees'] <= para.fba_fees_upper) | (
            df_fbm['gross_margin'] >= para.gross_margin_upper)
    df_fbm['直发FBM可能性'] = np.where(conditions_fbm_1 & conditions_fbm_2 & conditions_fbm_3, 1 + df_fbm['中国卖家FBM可能性'], 0)
    df_fbm = df_fbm[['id', '重量(g)', '直发FBM可能性']]
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
    current_date = pd.to_datetime(datetime.now().date())
    df['date_available'] = pd.to_datetime(df['date_available'], errors='coerce')
    df['开售天数'] = (current_date - df['date_available']).dt.days
    # df['date_available'] = np.where(df['开售天数'] * 1 > 0, df['date_available'], pd.to_datetime('1900-01-01'))
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


# 排序
def sort_and_rank(df):
    df = df.sort_values(by=['parent', 'ac', '开售月数', 'update_time'], ascending=[True, False, False, False])
    df['rank'] = df.groupby('parent').cumcount() + 1
    return df['rank']


# 产品加权计算
def product_avg(df, col_a, col_b):
    row_a = np.array(df[col_a])
    row_b = np.array(df[col_b])
    if np.sum(row_b * (row_b > 0)) != 0:
        row_avg = np.sum(row_a * row_b * (row_b > 0)) / np.sum(row_b * (row_b > 0))
    else:
        row_avg = np.nan
    return row_avg


# 综合推荐度计算
def product_recommend(df, col_a, col_b, col_str):
    asin_recommend = df.groupby(df['related_asin']).apply(
        lambda x: product_avg(x, col_a, col_b) if not x.empty else np.nan, include_groups=False)
    return convert_col(asin_recommend, col_str)


# 产品数计算
def product_count(df, col_str):
    asins_count = df['asin'].groupby(df['related_asin']).count()
    return convert_col(asins_count, col_str)


# 产品销额计算
def product_sum(df, sum_str, col_str):
    asins_revenue = df[sum_str].groupby(df['related_asin']).sum()
    return convert_col(asins_revenue, col_str)


# 产品均值计算
def product_mean(df, mean_str, col_str):
    asins_mean = df[mean_str].groupby(df['related_asin']).mean()
    return convert_col(asins_mean, col_str)


# 中位数计算
def product_median(df, median_str, col_str):
    asins_median = df[median_str].groupby(df['related_asin']).median()
    return convert_col(asins_median, col_str)


# 标准差计算
def product_std(df, std_str, col_str):
    asins_std = df[std_str].groupby(df['related_asin']).std()
    return convert_col(asins_std, col_str)


# 众数计算
def product_mode(df, std_str, col_str):
    asins_mode = df.groupby('related_asin')[std_str].agg(lambda x: x.value_counts().idxmax() if not x.empty else np.nan)
    return convert_col(asins_mode, col_str)


# PMI计算
def pmi_score(row):
    P_score = 0
    P_tags = []
    M_score = 0
    M_tags = []
    I_tags = []

    for col in para.pmi_list:
        col_pmi = col + "_score"
        col_pmi_tag = col + "_tag"
        if row[col_pmi_tag] in para.p_list:
            P_score += round(row[col_pmi], 1)
            P_tags.append(str(row[col_pmi_tag]))
        elif row[col_pmi_tag] in para.m_list:
            M_score += round(row[col_pmi], 1)
            M_tags.append(str(row[col_pmi_tag]))
        elif row[col_pmi_tag] in para.i_list:
            I_tags.append(str(row[col_pmi_tag]))

    row['P得分'] = P_score
    row['P标签'] = ','.join(P_tags)
    row['M得分'] = M_score
    row['M标签'] = ','.join(M_tags)
    row['I标签'] = ','.join(I_tags)

    return row


# 基础清洗
def df_clear(df, clear_id):
    df = df.replace('none', np.nan, regex=False)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.drop_duplicates(subset=[clear_id])
    df = df.dropna(subset=[clear_id])
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
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented.", category=PerformanceWarning)

site = 'us'

# 1.数据连接
df_famous_brand = sql_engine.connect_pt_product(config.oe_hostname, config.oe_password, path.product_database,
                                                sql.famous_brand_sql)
df_holiday = sql_engine.connect_pt_product(config.oe_hostname, config.oe_password, path.product_database,
                                           sql.holiday_sql)

df_product = sql_engine.connect_pt_product(config.sellersprite_hostname, config.sellersprite_password,
                                           config.clue_self_database, sql.sql_asin_self)

df_relation = sql_engine.connect_pt_product(config.sellersprite_hostname, config.sellersprite_password,
                                            config.clue_self_database, sql.sql_traffic_self)

if df_relation.empty:
    print('df_relation.empty')
    sys.exit()

df_relation = df_relation.merge(df_product, how='left', on='related_asin')

# 2.数据预处理
df_relation['ASIN'] = df_relation['asin']
df_relation = df_relation[(df_relation['relevance'] >= 0.5) & (df_relation['category_relevance'] >= 0)]

# 数据类型转换
product_con_list_1 = ['category_bsr_growth', 'sales_growth', 'price', 'monthly_revenue', 'gross_margin',
                      'fba_fees']

for con_i in product_con_list_1:
    df_relation[con_i] = data_cleaning_util.convert_type(df_relation, con_i, 2)

product_con_list_2 = ['sales', 'qa', 'ratings', 'variations']

for con_j in product_con_list_2:
    df_relation[con_j] = data_cleaning_util.convert_type(df_relation, con_j, 0)

df_relation['reviews_rate'] = data_cleaning_util.convert_type(df_relation, 'reviews_rate', 4)

df_relation['rating'] = data_cleaning_util.convert_type(df_relation, 'rating', 1)

product_con_list_3 = ['brand', 'title', 'category_path', 'category', 'sub_category', 'parent',
                      'seller_type', 'buybox_seller', 'ac_keyword', 'weight', '二级类目']

for con_l in product_con_list_3:
    df_relation[con_l] = data_cleaning_util.convert_str(df_relation, con_l)

product_con_list_5 = ['title', 'category_path', 'category', 'sub_category', 'ac_keyword', 'weight', '二级类目']

for con_lower in product_con_list_5:
    df_relation[con_lower] = df_relation[con_lower].str.lower()

product_con_list_4 = ['date_available', 'update_time']

for con_h in product_con_list_4:
    df_relation[con_h] = data_cleaning_util.convert_date(df_relation, con_h)

df_relation['sales_growth'] = data_cleaning_util.convert_type(df_relation, 'sales_growth', 4)

df_relation['monthly_revenue_increase'] = df_relation['sales_growth']

df_famous_brand['brand'] = data_cleaning_util.convert_str(df_famous_brand, 'brand')

df_famous_brand['疑似知名品牌'] = data_cleaning_util.convert_type(df_famous_brand, '疑似知名品牌', 0)

df_holiday['节日关键词'] = data_cleaning_util.convert_str(df_holiday, '节日关键词')

df_holiday['节日关键词'] = df_holiday['节日关键词'].str.lower()

for error_u, replace_m in para.replace_related_type_dict.items():
    df_relation.loc[:, 'related_type'] = df_relation['related_type'].str.replace(
        error_u, replace_m, regex=False)

df_relation.loc[:, 'related_type'] = df_relation['related_type'].fillna("以图搜图")

# 3.获取主推变体款

# 3.1开售月数
month_available(df_relation)

# 3.2排序
# 销额处理
df_relation['monthly_revenue'] = df_relation['monthly_revenue'].fillna(0)

df_relation_revenue_avg = df_relation.groupby(df_relation['parent'])['monthly_revenue'].mean()
df_relation_revenue_avg = data_cleaning_util.convert_col(df_relation_revenue_avg, 'monthly_revenue_avg')
df_relation = df_relation.merge(df_relation_revenue_avg, how='left', on='parent')

# 竞品销额均为空的处理
df_parent_0 = df_relation.loc[df_relation['monthly_revenue_avg'] == 0]
df_parent_0.loc[:, 'rank'] = calculate_util.sort_and_rank_group(df_parent_0)

# 竞品销额含非空的处理
df_parent_1 = df_relation.loc[
    (df_relation['monthly_revenue_avg'] > 0) & (df_relation['monthly_revenue'] > 0)]
df_parent_1.loc[:, 'rank'] = calculate_util.sort_and_rank_group(df_parent_1)

df_main = pd.concat([df_parent_0[df_parent_0['rank'] == 1], df_parent_1[df_parent_1['rank'] == 1]])

# 4.M相关指标计算
# 去除空值行
df_main_weight = df_main[df_main['weight'].notnull()]

if not df_main_weight.empty:
    # 替换错误单位
    for error_unit, replacement in para.replace_weight_error_dict.items():
        df_main_weight['重量(g)'] = calculate_util.weight_g(df_main_weight)
else:
    # 如果DataFrame为空，创建空的DataFrame并设置重量列为NaN
    df_main_weight = pd.DataFrame(columns=df_main.columns)
    df_main_weight['重量(g)'] = np.nan

# 直发FBM可能性
df_traffic_fbm = get_fbm(df_main_weight)

df_traffic = df_main.merge(df_traffic_fbm, how='left', on='id')
df_traffic['直发FBM可能性'] = df_traffic['直发FBM可能性'].fillna(0)

df_traffic['预估FBA占比'] = calculate_util.fba_rate(df_traffic)

df_traffic['预估头程占比'] = calculate_util.pre_rate(df_traffic)

df_traffic['预估货值占比'] = calculate_util.product_rate(df_traffic)

df_traffic['预估货值占比'] = data_cleaning_util.convert_type(df_traffic, '预估货值占比', 2)

df_traffic['预估毛利率'] = calculate_util.profit_rate(df_traffic, site)

df_traffic['毛利率级别'] = calculate_util.profit_rate_tag(df_traffic)

df_traffic['毛估资金利用率'] = calculate_util.product_revenue(df_traffic)

# 5.推荐度相关指标计算
# M相关指标
df_traffic['高资金利用率'] = calculate_util.high_product_revenue(df_traffic)

# I相关指标
df_traffic['开售月数'] = calculate_util.month_available(df_traffic)

# S相关指标
df_traffic['销额级数'] = calculate_util.get_revenue(df_traffic, site)

df_traffic['高销低LQS'] = calculate_util.high_sale_low_lqs(df_traffic)

df_traffic['月均QA数'] = calculate_util.qa_per_month(df_traffic)

df_traffic[['长期上架少Q&A', '长期上架无A+', '长期上架无视频']] = calculate_util.long_term_sale(df_traffic, site)

df_traffic['类轻小直发FBM'] = calculate_util.light_small_fbm(df_traffic)

df_traffic['差评好卖'] = calculate_util.low_star_high_sale(df_traffic, site)

# 知名品牌
df_traffic_brand = df_traffic[df_traffic['brand'].notnull()]
if df_traffic_brand.empty:
    df_traffic['疑似知名品牌'] = 0
else:
    df_traffic = df_traffic.merge(df_famous_brand, how='left', on='brand')

df_traffic['combined_kw'] = df_traffic['title'] + "" + df_traffic['sub_category'] + "" + df_traffic['ac_keyword']

df_traffic[['疑似节日性', '节日名']] = df_traffic.apply(lambda row: pd.Series(match_holidays(row)), axis=1)

df_traffic['知名品牌_1'] = np.where(df_traffic['疑似知名品牌'] > 0,
                                -df_traffic['疑似知名品牌'] / np.where(df_traffic['疑似节日性'] * 1 > 0, 2, 1), 0)

conditions_brand = (df_traffic['seller_type'] == 'AMZ') | (df_traffic['buybox_seller'] == 'Amazon') | (
        df_traffic['buybox_seller'] == 'Amazon.com')
df_traffic['知名品牌_2'] = np.where(conditions_brand, -3, 0)

df_traffic['知名品牌'] = np.where(df_traffic['brand'] == 'Generic', 1, df_traffic['知名品牌_1'] + df_traffic['知名品牌_2'])

df_traffic['知名品牌'] = df_traffic['知名品牌'].fillna(0)

# 疑似节日性
df_traffic['疑似节日性'] = np.where(df_traffic['疑似节日性'] * 1 > 3, "3+", df_traffic['疑似节日性'])

# 是否个人定制
df_traffic['custom_kw'] = df_traffic.apply(lambda row: pd.Series(calculate_util.match_custom_kw(row)),
                                           axis=1)
df_traffic['是否个人定制'] = np.where(df_traffic['custom_kw'] > 0, 1, 0)

# 是否翻新
df_traffic['是否翻新'] = calculate_util.get_renewed(df_traffic)

# L相关指标
df_traffic['平均变体月销额等级'] = calculate_util.revenue_per_variations(df_traffic, site)
df_traffic['变体等级'] = calculate_util.variations_tag(df_traffic)
df_traffic['少变体'] = calculate_util.few_variations(df_traffic)

# E相关指标
df_traffic[['新品爬坡快', '新品增评好', '新品NSR', '新品AC标']] = calculate_util.product_new(df_traffic, site)

df_traffic['少评好卖'] = calculate_util.few_star_high_sale(df_traffic)

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

# 类目清洗
df_traffic['category'] = df_traffic['category'].replace(para.replace_category_dict, regex=False)
df_traffic['二级类目'] = df_traffic['二级类目'].replace(para.replace_category_dict_2, regex=False)

# 剔除类目
df_traffic.loc[
    df_traffic['category_path'].str.contains(para.regex_pattern_kw, na=False, regex=False), '剔除类目'] = 1
df_traffic['剔除类目'] = np.where(df_traffic['剔除类目'] == 1, 1, 0)

# 数据格式整理
product_con_list = ['预估FBA占比', '预估头程占比', '预估毛利率', '毛估资金利用率', '销额级数', '高资金利用率', '高销低LQS', '类轻小直发FBM', '平均变体月销额等级',
                    '新品爬坡快', '长期上架少Q&A']
for con_k in product_con_list:
    convert_type(df_traffic, con_k, 4)

# 6.聚合字段计算
# 获取有销额产品
traffic_revenue_pass = df_traffic.query('monthly_revenue > 0')

# 获取销额TOP5关联竞品
traffic_revenue_pass['revenue_rank'] = traffic_revenue_pass.groupby(df_traffic['related_asin'])[
    'monthly_revenue'].rank(ascending=False)
traffic_top5_df = traffic_revenue_pass.query('revenue_rank <= 5')

# 聚合表生成
traffic_df = traffic_top5_df.groupby('related_asin').agg(
    {'relevance': 'first',
     'asin': lambda x: ','.join(x.astype(str)),
     '数据更新时间': 'first'}).reset_index()

# 相关竞品款数
traffic_count = product_count(df_traffic, '相关竞品款数')

# 有销额竞品款数
traffic_revenue_count = product_count(traffic_revenue_pass, '有销额竞品款数')
traffic_revenue_pass = traffic_revenue_pass.merge(traffic_revenue_count, how='left', on='related_asin')

# 有销额推荐达标款数
traffic_recommend = traffic_revenue_pass.query('推荐度>=3')
traffic_recommend_count = product_count(traffic_recommend, '有销额推荐达标款数')

traffic_df = traffic_df.merge(traffic_count, how='left', on='related_asin') \
    .merge(traffic_revenue_count, how='left', on='related_asin') \
    .merge(traffic_recommend_count, how='left', on='related_asin')

# 综合竞品推荐度
traffic_df_recommend_all = product_recommend(df_traffic, '推荐度', 'relevance', '综合竞品推荐度_all')
traffic_df_recommend_revenue = product_recommend(traffic_revenue_pass, '推荐度', 'relevance', '综合竞品推荐度_有销额')

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
df_traffic_recommend = traffic_top5_df.merge(traffic_recommend_tag, how='left', on='related_asin')

# -------------------------------V2版本-------------------------34个维度聚合------------------------------------

# S规模分析
# TOP5月销额
traffic_revenue_top5 = product_sum(traffic_top5_df, 'monthly_revenue', 'TOP5月销额')

traffic_df = traffic_df.merge(traffic_revenue_top5, how='left', on='related_asin')

# TOP5月均销额
traffic_df['TOP5月均销额'] = np.where(traffic_df['有销额竞品款数'] > 0,
                                  traffic_df['TOP5月销额'] / np.fmin(5, traffic_df['有销额竞品款数']), np.nan)

traffic_df['TOP5月均销额_score'] = get_cut(traffic_df, 'TOP5月均销额', para.s_sales_bins, para.s_sales_labels)
traffic_df['TOP5月均销额_tag'] = get_cut(traffic_df, 'TOP5月均销额', para.s_sales_bins, para.s_sales_tags)

# 利基月GMV
traffic_revenue = product_sum(traffic_revenue_pass, 'monthly_revenue', '利基月GMV')

traffic_df = traffic_df.merge(traffic_revenue, how='left', on='related_asin')

# 有销额竞品款数占比
traffic_df['有销额竞品款数占比'] = np.where(traffic_df['相关竞品款数'] > 0, traffic_df['有销额竞品款数'] / traffic_df['相关竞品款数'],
                                   np.nan)

# M资金回报分析
# 价格中位数
traffic_price_median = product_median(df_traffic, 'price', '价格中位数')

traffic_df = traffic_df.merge(traffic_price_median, how='left', on='related_asin')

# 价格标准差
traffic_price_std = product_std(df_traffic, 'price', '价格标准差')

traffic_df = traffic_df.merge(traffic_price_std, how='left', on='related_asin')

# 价格集中度
traffic_df['价格集中度'] = np.where(traffic_df['价格中位数'] > 0, 1 - traffic_df['价格标准差'] / traffic_df['价格中位数'], np.nan)

traffic_df['价格集中度_score'] = get_cut(traffic_df, '价格集中度', para.m_price_bins, para.m_price_labels)
traffic_df['价格集中度_tag'] = get_cut(traffic_df, '价格集中度', para.m_price_bins, para.m_price_tags)

# 预估平均毛利率
traffic_gross_margin = product_mean(traffic_revenue_pass, '预估毛利率', '预估平均毛利率')

traffic_df = traffic_df.merge(traffic_gross_margin, how='left', on='related_asin')

traffic_df['预估平均毛利率_score'] = get_cut(traffic_df, '预估平均毛利率', para.m_gross_bins, para.m_gross_labels)
traffic_df['预估平均毛利率_tag'] = get_cut(traffic_df, '预估平均毛利率', para.m_gross_bins, para.m_gross_tags)

# 预估平均资金利用率
traffic_revenue_reward = product_mean(traffic_revenue_pass, '毛估资金利用率', '预估平均资金利用率')

traffic_df = traffic_df.merge(traffic_revenue_reward, how='left', on='related_asin')

traffic_df['预估平均资金利用率_score'] = get_cut(traffic_df, '预估平均资金利用率', para.m_revenue_bins, para.m_revenue_labels)
traffic_df['预估平均资金利用率_tag'] = get_cut(traffic_df, '预估平均资金利用率', para.m_revenue_bins, para.m_revenue_tags)

# 获取FBA运费不为空行
traffic_fba_fees = df_traffic.query('fba_fees > 0')

# 加权FBA运费
traffic_fba_fees_mean = product_mean(traffic_fba_fees, 'fba_fees', '加权FBA运费')

traffic_df = traffic_df.merge(traffic_fba_fees_mean, how='left', on='related_asin')

traffic_df['加权FBA运费_score'] = get_cut(traffic_df, '加权FBA运费', para.m_fba_bins, para.m_fba_labels)
traffic_df['加权FBA运费_tag'] = get_cut(traffic_df, '加权FBA运费', para.m_fba_bins, para.m_fba_tags)

# 获取FBM配送产品
traffic_fbm_df = traffic_revenue_pass.query('seller_type == "FBM"')

# FBM配送产品数
traffic_fbm_count = product_count(traffic_fbm_df, 'FBM配送产品数')

traffic_df = traffic_df.merge(traffic_fbm_count, how='left', on='related_asin')

# FBM配送占比
traffic_df['FBM配送占比'] = np.where(traffic_df['有销额竞品款数'] > 0, traffic_df['FBM配送产品数'] / traffic_df['有销额竞品款数'],
                                 np.nan)

traffic_df['FBM配送占比_score'] = get_cut(traffic_df, 'FBM配送占比', para.m_fbm_bins, para.m_fbm_labels)
traffic_df['FBM配送占比_tag'] = get_cut(traffic_df, 'FBM配送占比', para.m_fbm_bins, para.m_fbm_tags)

# 获取直发FBM产品
traffic_fbm_cal_df = traffic_revenue_pass.query('直发FBM可能性 >= 1')

# 直发FBM产品数
traffic_fbm_cal_count = product_count(traffic_fbm_cal_df, '直发FBM产品数')

traffic_df = traffic_df.merge(traffic_fbm_cal_count, how='left', on='related_asin')

# 直发FBM产品占比
traffic_df['直发FBM产品占比'] = np.where(traffic_df['有销额竞品款数'] > 0, traffic_df['直发FBM产品数'] / traffic_df['有销额竞品款数'],
                                   np.nan)

# 直发FBM销额
traffic_fbm_cal_revenue = product_sum(traffic_fbm_cal_df, 'monthly_revenue', '直发FBM销额')

traffic_df = traffic_df.merge(traffic_fbm_cal_revenue, how='left', on='related_asin')

# 直发FBM销额占比
traffic_df['直发FBM销额占比'] = np.where(traffic_df['利基月GMV'] > 0, traffic_df['直发FBM销额'] / traffic_df['利基月GMV'],
                                   np.nan)

conditions_fbm_rate = (traffic_df['直发FBM产品占比'] >= 0.15) | (traffic_df['直发FBM销额占比'] >= 0.2)

traffic_df['直发FBM产品占比_score'] = np.where(conditions_fbm_rate, 2, np.nan)
traffic_df['直发FBM产品占比_tag'] = np.where(conditions_fbm_rate, "直发FBM多", np.nan)

# 直发FBM月均销额
traffic_fbm_cal_revenue_mean = product_mean(traffic_fbm_cal_df, 'monthly_revenue', '直发FBM月均销额')

traffic_df = traffic_df.merge(traffic_fbm_cal_revenue_mean, how='left', on='related_asin')

traffic_df['直发FBM月均销额_score'] = get_cut(traffic_df, '直发FBM月均销额', para.m_fbm_cal_sales_bins,
                                        para.m_fbm_cal_sales_labels)
traffic_df['直发FBM月均销额_tag'] = get_cut(traffic_df, '直发FBM月均销额', para.m_fbm_cal_sales_bins, para.m_fbm_cal_sales_tags)

traffic_df['直发FBM月均销额_score'] = np.where(traffic_df['直发FBM产品数'] >= 3, traffic_df['直发FBM月均销额_score'], np.nan)
traffic_df['直发FBM月均销额_tag'] = np.where(traffic_df['直发FBM产品数'] >= 3, traffic_df['直发FBM月均销额_tag'], np.nan)

# I内卷分析
# 广告蓝海度

# 获取AMZ直营产品
traffic_amz_df = traffic_revenue_pass.query('seller_type == "AMZ"')

# AMZ直营销额
traffic_amz_revenue = product_sum(traffic_amz_df, 'monthly_revenue', 'AMZ直营销额')

traffic_df = traffic_df.merge(traffic_amz_revenue, how='left', on='related_asin')

# AMZ直营销额占比
traffic_df['AMZ直营销额占比'] = np.where(traffic_df['利基月GMV'] > 0, traffic_df['AMZ直营销额'] / traffic_df['利基月GMV'],
                                   np.nan)

traffic_df['AMZ直营销额占比_score'] = get_cut(traffic_df, 'AMZ直营销额占比', para.i_amz_bins, para.i_amz_labels)
traffic_df['AMZ直营销额占比_tag'] = get_cut(traffic_df, 'AMZ直营销额占比', para.i_amz_bins, para.i_amz_tags)

# 获取大牌商标产品
traffic_famous_df = traffic_revenue_pass.query('疑似知名品牌 >= 4')
# 大牌商标销额
traffic_famous_revenue = product_sum(traffic_famous_df, 'monthly_revenue', '大牌商标销额')

traffic_df = traffic_df.merge(traffic_famous_revenue, how='left', on='related_asin')

# 大牌商标销额占比
traffic_df['大牌商标销额占比'] = np.where(traffic_df['利基月GMV'] > 0, traffic_df['大牌商标销额'] / traffic_df['利基月GMV'],
                                  np.nan)

traffic_df['大牌商标销额占比_score'] = get_cut(traffic_df, '大牌商标销额占比', para.i_famous_bins, para.i_famous_labels)
traffic_df['大牌商标销额占比_tag'] = get_cut(traffic_df, '大牌商标销额占比', para.i_famous_bins, para.i_famous_tags)

# 获取中国卖家产品
traffic_cn_df = df_traffic.query('buybox_location == "CN"')
# 中国卖家产品数
traffic_cn_count = product_count(traffic_cn_df, '中国卖家产品数')

traffic_df = traffic_df.merge(traffic_cn_count, how='left', on='related_asin')

# 中国卖家占比
traffic_df['中国卖家占比'] = np.where(traffic_df['相关竞品款数'] > 0, traffic_df['中国卖家产品数'] / traffic_df['相关竞品款数'],
                                np.nan)

traffic_df['中国卖家占比_score'] = get_cut(traffic_df, '中国卖家占比', para.i_cn_bins, para.i_cn_labels)
traffic_df['中国卖家占比_tag'] = get_cut(traffic_df, '中国卖家占比', para.i_cn_bins, para.i_cn_tags)

# TOP5平均LQS
traffic_top5_lqs = product_mean(traffic_top5_df, 'lqs', 'TOP5平均LQS')

traffic_df = traffic_df.merge(traffic_top5_lqs, how='left', on='related_asin')

traffic_df['TOP5平均LQS_score'] = get_cut(traffic_df, 'TOP5平均LQS', para.i_lqs_top5_bins, para.i_lqs_top5_labels)
traffic_df['TOP5平均LQS_tag'] = get_cut(traffic_df, 'TOP5平均LQS', para.i_lqs_top5_bins, para.i_lqs_top5_tags)

# 获取冒出产品
traffic_good_df = traffic_revenue_pass.query('monthly_revenue >= 600')

# 冒出品产品数
traffic_good_count = product_count(traffic_good_df, '冒出品产品数')

traffic_df = traffic_df.merge(traffic_good_count, how='left', on='related_asin')

# 冒出品平均LQS
traffic_good_lqs = product_mean(traffic_good_df, 'lqs', '冒出品平均LQS')

traffic_df = traffic_df.merge(traffic_good_lqs, how='left', on='related_asin')

traffic_df['冒出品平均LQS_score'] = get_cut(traffic_df, '冒出品平均LQS', para.i_lqs_bins, para.i_lqs_labels)
traffic_df['冒出品平均LQS_tag'] = get_cut(traffic_df, '冒出品平均LQS', para.i_lqs_bins, para.i_lqs_tags)

# 获取冒出品有A+产品
traffic_good_ebc_df = traffic_good_df.query('ebc_available == "Y"')

# 冒出品有A+产品数
traffic_good_ebc_count = product_count(traffic_good_ebc_df, '冒出品有A+产品数')

traffic_df = traffic_df.merge(traffic_good_ebc_count, how='left', on='related_asin')

# 冒出品A+占比
traffic_df['冒出品A+占比'] = np.where(traffic_df['冒出品产品数'] >= 5, traffic_df['冒出品有A+产品数'] / traffic_df['冒出品产品数'],
                                 np.nan)

traffic_df['冒出品A+占比_score'] = get_cut(traffic_df, '冒出品A+占比', para.i_ebc_bins, para.i_ebc_labels)
traffic_df['冒出品A+占比_tag'] = get_cut(traffic_df, '冒出品A+占比', para.i_ebc_bins, para.i_ebc_tags)

# 获取冒出品有视频产品
traffic_good_video_df = traffic_good_df.query('video_available == "Y"')

# 冒出品有视频产品数
traffic_good_video_count = product_count(traffic_good_video_df, '冒出品有视频产品数')

traffic_df = traffic_df.merge(traffic_good_video_count, how='left', on='related_asin')

# 冒出品视频占比
traffic_df['冒出品视频占比'] = np.where(traffic_df['冒出品产品数'] >= 5, traffic_df['冒出品有视频产品数'] / traffic_df['冒出品产品数'],
                                 np.nan)

traffic_df['冒出品视频占比_score'] = get_cut(traffic_df, '冒出品视频占比', para.i_video_bins, para.i_video_labels)
traffic_df['冒出品视频占比_tag'] = get_cut(traffic_df, '冒出品视频占比', para.i_video_bins, para.i_video_tags)

# 获取冒出品有QA产品
traffic_good_qa_df = traffic_good_df.query('qa >= 1')

# 冒出品有QA产品数
traffic_good_qa_count = product_count(traffic_good_qa_df, '冒出品有QA产品数')

traffic_df = traffic_df.merge(traffic_good_qa_count, how='left', on='related_asin')

# 冒出品QA占比
traffic_df['冒出品QA占比'] = np.where(traffic_df['冒出品产品数'] >= 5, traffic_df['冒出品有QA产品数'] / traffic_df['冒出品产品数'],
                                 np.nan)

traffic_df['冒出品QA占比_score'] = get_cut(traffic_df, '冒出品QA占比', para.i_qa_bins, para.i_qa_labels)
traffic_df['冒出品QA占比_tag'] = get_cut(traffic_df, '冒出品QA占比', para.i_qa_bins, para.i_qa_tags)

# 获取动销产品
traffic_available_df = traffic_revenue_pass.query('monthly_revenue >= 100')

# 动销品平均星级
traffic_available_rating = product_mean(traffic_available_df, 'rating', '动销品平均星级')

traffic_df = traffic_df.merge(traffic_available_rating, how='left', on='related_asin')

traffic_df['动销品平均星级_score'] = get_cut(traffic_df, '动销品平均星级', para.i_rating_bins, para.i_rating_labels)
traffic_df['动销品平均星级_tag'] = get_cut(traffic_df, '动销品平均星级', para.i_rating_bins, para.i_rating_tags)

# 获取冒出品低星产品
traffic_good_rating_lower = traffic_good_df.query('rating < 3.9')

# 冒出品低星产品数(冒出品低星款数)
traffic_good_rating_lower_count = product_count(traffic_good_rating_lower, '冒出品低星款数')

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

traffic_df['TOP5销额占比_score'] = get_cut(traffic_df, 'TOP5销额占比', para.l_sales_top5_bins, para.l_sales_top5_labels)
traffic_df['TOP5销额占比_tag'] = get_cut(traffic_df, 'TOP5销额占比', para.l_sales_top5_bins, para.l_sales_top5_tags)

# 非TOP5销额占比
traffic_df['非TOP5销额占比'] = np.where(traffic_df['利基月GMV'] > 0, 1 - traffic_df['TOP5销额占比'], np.nan)

traffic_df['非TOP5销额占比_score'] = get_cut(traffic_df, '非TOP5销额占比', para.l_sales_rate_bins, para.l_sales_rate_labels)
traffic_df['非TOP5销额占比_tag'] = get_cut(traffic_df, '非TOP5销额占比', para.l_sales_rate_bins, para.l_sales_rate_tags)

# 非TOP5月均销额
traffic_df['非TOP5月均销额'] = np.where(traffic_df['有销额竞品款数'] > 5,
                                   (traffic_df['利基月GMV'] - traffic_df['TOP5月销额']) / (traffic_df['有销额竞品款数'] - 5),
                                   np.nan)

traffic_df['非TOP5月均销额_score'] = get_cut(traffic_df, '非TOP5月均销额', para.l_sales_bins, para.l_sales_labels)
traffic_df['非TOP5月均销额_tag'] = get_cut(traffic_df, '非TOP5月均销额', para.l_sales_bins, para.l_sales_tags)

# 动销品变体中位数
traffic_available_variations = product_median(traffic_available_df, 'variations', '动销品变体中位数')

traffic_df = traffic_df.merge(traffic_available_variations, how='left', on='related_asin')

traffic_df['动销品变体中位数_score'] = get_cut(traffic_df, '动销品变体中位数', para.l_variations_bins, para.l_variations_labels)
traffic_df['动销品变体中位数_tag'] = get_cut(traffic_df, '动销品变体中位数', para.l_variations_bins, para.l_variations_tags)

# E新品冒出
# 平均开售月数
traffic_revenue_month = product_mean(traffic_revenue_pass, '开售月数', '平均开售月数')

traffic_df = traffic_df.merge(traffic_revenue_month, how='left', on='related_asin')

traffic_df['平均开售月数_score'] = get_cut(traffic_df, '平均开售月数', para.e_month_bins, para.e_month_labels)
traffic_df['平均开售月数_tag'] = get_cut(traffic_df, '平均开售月数', para.e_month_bins, para.e_month_tags)

# 获取冒出品新品产品
traffic_good_new = traffic_good_df.query('开售月数 <= 9')

# 冒出品新品产品数
traffic_good_new_count = product_count(traffic_good_new, '冒出品新品产品数')

traffic_df = traffic_df.merge(traffic_good_new_count, how='left', on='related_asin')

# 冒出品新品占比
traffic_df['冒出品新品占比'] = np.where(traffic_df['冒出品产品数'] >= 5, traffic_df['冒出品新品产品数'] / traffic_df['冒出品产品数'],
                                 np.nan)

traffic_df['冒出品新品占比_score'] = get_cut(traffic_df, '冒出品新品占比', para.e_new_bins, para.e_new_labels)
traffic_df['冒出品新品占比_tag'] = get_cut(traffic_df, '冒出品新品占比', para.e_new_bins, para.e_new_tags)

# 获取新品产品
traffic_revenue_new = traffic_revenue_pass.query('有销额竞品款数 >= 5 and 开售月数 <= 9')

# 新品平均月销额
traffic_revenue_new_revenue = product_mean(traffic_revenue_new, 'monthly_revenue', '新品平均月销额')

traffic_df = traffic_df.merge(traffic_revenue_new_revenue, how='left', on='related_asin')

traffic_df['新品平均月销额_score'] = get_cut(traffic_df, '新品平均月销额', para.e_new_sales_bins, para.e_new_sales_labels)
traffic_df['新品平均月销额_tag'] = get_cut(traffic_df, '新品平均月销额', para.e_new_sales_bins, para.e_new_sales_tags)

# 平均星数
traffic_available_ratings = product_mean(traffic_revenue_pass, 'ratings', '平均星数')

traffic_df = traffic_df.merge(traffic_available_ratings, how='left', on='related_asin')

traffic_df['平均星数_score'] = get_cut(traffic_df, '平均星数', para.e_ratings_bins, para.e_ratings_labels)
traffic_df['平均星数_tag'] = get_cut(traffic_df, '平均星数', para.e_ratings_bins, para.e_ratings_tags)

# 销级星数比
traffic_sales_mean = product_mean(df_traffic, '销级星数比', '销级星数比')

traffic_df = traffic_df.merge(traffic_sales_mean, how='left', on='related_asin')

# 获取三标新品产品
conditions_lables = ((traffic_revenue_pass['best_seller'] == 'Y') | (traffic_revenue_pass['ac'] == 'Y') | (
        traffic_revenue_pass['new_release'] == 'Y'))
traffic_lables_df = traffic_revenue_pass[conditions_lables]
traffic_new_lables_df = traffic_revenue_pass[(traffic_revenue_pass['开售月数'] <= 9) & conditions_lables]

# 三标产品数
traffic_lables_count = product_count(traffic_lables_df, '三标产品数')

traffic_df = traffic_df.merge(traffic_lables_count, how='left', on='related_asin')

# 三标新品产品数
traffic_new_lables_count = product_count(traffic_new_lables_df, '三标新品产品数')

traffic_df = traffic_df.merge(traffic_new_lables_count, how='left', on='related_asin')

# 三标新品占比
traffic_df['三标新品占比'] = np.where(traffic_df['三标产品数'] >= 5, traffic_df['三标新品产品数'] / traffic_df['三标产品数'],
                                np.nan)

traffic_df['三标新品占比_score'] = get_cut(traffic_df, '三标新品占比', para.e_new_lables_bins, para.e_new_lables_labels)
traffic_df['三标新品占比_tag'] = get_cut(traffic_df, '三标新品占比', para.e_new_lables_bins, para.e_new_lables_tags)

# 月销增长率不为空产品数
traffic_sales_increase = traffic_revenue_pass.loc[traffic_revenue_pass['monthly_revenue_increase'].notnull()]

traffic_sales_increase_count = product_count(traffic_sales_increase, '月销增长率不为空产品数')

traffic_sales_increase = traffic_sales_increase.merge(traffic_sales_increase_count, how='left',
                                                      on='related_asin')

# 月销增长率
traffic_sales_increase_revenue = traffic_sales_increase.query('月销增长率不为空产品数 >= 10')

traffic_sales_increase_rate = product_recommend(traffic_sales_increase_revenue, 'monthly_revenue_increase',
                                                'monthly_revenue', '月销增长率')

traffic_df = traffic_df.merge(traffic_sales_increase_rate, how='left', on='related_asin')

traffic_df['月销增长率_score'] = get_cut(traffic_df, '月销增长率', para.e_sales_bins, para.e_sales_labels)
traffic_df['月销增长率_tag'] = get_cut(traffic_df, '月销增长率', para.e_sales_bins, para.e_sales_tags)

# 平均留评率
traffic_ratings_rate_mean = product_mean(traffic_revenue_pass, 'reviews_rate', '平均留评率')

traffic_revenue_pass = traffic_revenue_pass.merge(traffic_ratings_rate_mean, how='left', on='related_asin')

# 留评率标准差
traffic_ratings_rate_std = product_std(traffic_revenue_pass, 'reviews_rate', '留评率标准差')

traffic_revenue_pass = traffic_revenue_pass.merge(traffic_ratings_rate_std, how='left', on='related_asin')

# 筛选留评率计算数据
traffic_revenue_pass['留评率筛选标准'] = traffic_revenue_pass['平均留评率'] + 2 * traffic_revenue_pass['留评率标准差']

traffic_ratings_rate_pass = traffic_revenue_pass.loc[
    traffic_revenue_pass['reviews_rate'] < traffic_revenue_pass['留评率筛选标准']]

# 加权留评率
traffic_ratings_avg = traffic_ratings_rate_pass.loc[traffic_ratings_rate_pass['reviews_rate'] <= 0.1]

traffic_ratings_rate = product_recommend(traffic_revenue_pass, 'reviews_rate', 'monthly_revenue', '加权留评率')
traffic_df = traffic_df.merge(traffic_ratings_rate, how='left', on='related_asin')

# PMI计算
traffic_df = traffic_df.apply(lambda row: pmi_score(row), axis=1)

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
df_category = product_mode(df_traffic_category, 'sub_category', '代表节点')

df_traffic = df_traffic.merge(df_category, how='left', on='related_asin')
traffic_df = traffic_df.merge(df_category, how='left', on='related_asin')

# 同代表节点竞品数
df_category_main = df_traffic.loc[df_traffic['sub_category'] == df_traffic['代表节点']]

traffic_duplicate = product_count(df_category_main, '同代表节点竞品数')

traffic_df = traffic_df.merge(traffic_duplicate, how='left', on='related_asin')

# 代表度计算
traffic_df['代表度'] = np.where(traffic_df['相关竞品款数'] * 1 > 0,
                             round(traffic_df['同代表节点竞品数'] / traffic_df['相关竞品款数'], 2), np.nan)

traffic_df['重复利基'] = np.where(traffic_df['代表度'] * 1 < 0.4, 0, 1)

# -------------------------------V3版本-------------------------推荐级别重算------------------------------------

# 推荐级别v2
traffic_df['推荐级别v2'] = get_cut(traffic_df, 'PMI得分', para.pmi_bins, para.pmi_labels)

# 综合投票分
traffic_df['推荐级别v1'] = traffic_df['推荐级别v1'].astype(int)
traffic_df['推荐级别v2'] = traffic_df['推荐级别v2'].astype(int)
traffic_df['综合投票分'] = np.where(abs(traffic_df['推荐级别v1'] * traffic_df['推荐级别v2']) > 0,
                               np.where(traffic_df['PMI得分'] > -8, round(
                                   abs((traffic_df['推荐级别v1'] + traffic_df['推荐级别v2']) / 2)) * np.where(
                                   (traffic_df['推荐级别v1'] + traffic_df['推荐级别v2']) / 2 >= 0, 1, -1), -2), 0)

traffic_df['综合推荐级别'] = get_cut(traffic_df, '推荐级别v2', para.recommend_bins, para.recommend_labels)

traffic_df['综合推荐级别'] = np.where(traffic_df['有销额竞品款数'] >= 5, traffic_df['综合推荐级别'], '<数据不足>')

# -------------------------------V3版本-------------------------推荐级别重算------------------------------------

# 格式转换
traffic_list_0 = ['TOP5月均销额', 'TOP5月销额', '利基月GMV', '直发FBM月均销额', '冒出品低星款数', '非TOP5月均销额', '动销品变体中位数',
                  '新品平均月销额', '平均星数', '重复利基']
for traffic_i in traffic_list_0:
    convert_type(traffic_df, traffic_i, 0)

traffic_list_1 = ['综合竞品推荐度', 'TOP5平均LQS', '冒出品平均LQS', '平均开售月数']
for traffic_j in traffic_list_1:
    convert_type(traffic_df, traffic_j, 1)

traffic_list_2 = ['综合竞品推荐度', '有销额竞品款数占比', '价格中位数', '预估平均资金利用率', '加权FBA运费', '动销品平均星级', 'TOP5销额占比', '非TOP5销额占比',
                  '代表度', 'P得分', 'M得分', 'PMI得分']
for traffic_k in traffic_list_2:
    convert_type(traffic_df, traffic_k, 2)

traffic_list_3 = ['达标推荐度占比', '价格集中度', '预估平均毛利率', 'FBM配送占比', '直发FBM产品占比', '直发FBM销额占比', 'AMZ直营销额占比', '大牌商标销额占比',
                  '中国卖家占比', '冒出品A+占比', '冒出品视频占比', '冒出品QA占比', '冒出品低星占比', '冒出品新品占比', '销级星数比', '三标新品占比', '月销增长率',
                  '加权留评率']
for traffic_l in traffic_list_3:
    convert_type(traffic_df, traffic_l, 3)

df_group = traffic_df

# -------------------------------V2版本-------------------------34个维度聚合------------------------------------


# 7.添加数据标签
df_group_tag = df_group[
    ['related_asin', '相关竞品款数', '有销额竞品款数', '有销额推荐达标款数', '有销额竞品款数占比', '综合竞品推荐度', '达标推荐度占比', '推荐级别v1', 'TOP5月均销额',
     'TOP5月销额', '利基月GMV', '价格中位数', '价格集中度', '预估平均毛利率', '预估平均资金利用率', '加权FBA运费', 'FBM配送占比', '直发FBM产品占比', '直发FBM销额占比',
     '直发FBM月均销额', 'AMZ直营销额占比', '大牌商标销额占比', '中国卖家占比', 'TOP5平均LQS', '冒出品平均LQS', '冒出品A+占比', '冒出品视频占比', '冒出品QA占比',
     '动销品平均星级', '冒出品低星占比', 'TOP5销额占比', '非TOP5销额占比', '非TOP5月均销额', '动销品变体中位数', '平均开售月数', '冒出品新品占比', '新品平均月销额', '平均星数',
     '销级星数比', '三标新品占比', '月销增长率', '加权留评率', '推荐级别v2', '综合投票分', '综合推荐级别', '数据更新时间']]

group_tag_list_5 = ['相关竞品款数', '有销额竞品款数', '有销额推荐达标款数']
for group_i in group_tag_list_5:
    group_tag_i = group_i + "分布"
    get_mround(df_group_tag, group_i, group_tag_i, 5)

group_tag_list_02 = ['综合竞品推荐度', 'TOP5平均LQS', '冒出品平均LQS', '预估平均资金利用率']
for group_02 in group_tag_list_02:
    group_tag_02 = group_02 + "分布"
    get_mround(df_group_tag, group_02, group_tag_02, 0.2)

group_tag_list_05 = ['有销额竞品款数占比', '达标推荐度占比', '价格集中度', 'FBM配送占比', '直发FBM产品占比', '直发FBM销额占比', 'AMZ直营销额占比', '大牌商标销额占比',
                     '中国卖家占比', '冒出品A+占比', '冒出品视频占比', '冒出品QA占比', '冒出品低星占比', 'TOP5销额占比', '非TOP5销额占比', '冒出品新品占比',
                     '销级星数比', '三标新品占比', '月销增长率']
for group_k in group_tag_list_05:
    group_tag_k = group_k + "分布"
    get_mround(df_group_tag, group_k, group_tag_k, 0.05)

group_tag_list_10000 = ['TOP5月均销额', 'TOP5月销额', '利基月GMV']
for group_10000 in group_tag_list_10000:
    group_tag_10000 = group_10000 + "分布"
    get_mround(df_group_tag, group_10000, group_tag_10000, 10000)

group_tag_list_100 = ['直发FBM月均销额', '非TOP5月均销额', '新品平均月销额', '平均星数']
for group_100 in group_tag_list_100:
    group_tag_100 = group_100 + "分布"
    get_mround(df_group_tag, group_100, group_tag_100, 100)

group_tag_list_2 = ['价格中位数', '加权FBA运费']
for group_2 in group_tag_list_2:
    group_tag_2 = group_2 + "分布"
    get_mround(df_group_tag, group_2, group_tag_2, 2)

get_mround(df_group_tag, '预估平均毛利率', '预估平均毛利率分布', 0.02)
get_mround(df_group_tag, '动销品平均星级', '动销品平均星级分布', 0.1)
get_mround(df_group_tag, '平均开售月数', '平均开售月数分布', 3)
get_mround(df_group_tag, '加权留评率', '加权留评率分布', 0.005)

df_group_tag['data_id'] = df_group_tag['related_asin'] + " | " + pd.to_datetime(df_group_tag['数据更新时间']).dt.strftime(
    '%Y-%m-%d')

# 8.字段整合
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
df_traffic_table = df_clear(df_traffic_table, 'clear_id')
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
    ['related_asin', '相关竞品款数', '有销额竞品款数', '有销额推荐达标款数', '有销额竞品款数占比',
     '综合竞品推荐度', '达标推荐度占比', '推荐级别v1', 'TOP5月均销额', 'TOP5月销额', '利基月GMV', '价格中位数', '价格集中度', '预估平均毛利率', '预估平均资金利用率',
     '加权FBA运费', 'FBM配送占比', '直发FBM产品占比', '直发FBM销额占比', '直发FBM月均销额', 'AMZ直营销额占比', '大牌商标销额占比', '中国卖家占比', 'TOP5平均LQS',
     '冒出品平均LQS', '冒出品A+占比', '冒出品视频占比', '冒出品QA占比', '动销品平均星级', '冒出品低星占比', 'TOP5销额占比', '非TOP5销额占比', '非TOP5月均销额',
     '动销品变体中位数', '平均开售月数', '冒出品新品占比', '新品平均月销额', '平均星数', '销级星数比', '三标新品占比', '月销增长率', '加权留评率', 'P得分', 'M得分', 'PMI得分',
     'P标签', 'M标签', 'I标签', '推荐级别v2', '综合投票分', '综合推荐级别', '代表节点', '代表度', '重复利基', 'asin', '数据更新时间']]
df_group = df_clear(df_group, 'related_asin')

df_group.rename(columns={'related_asin': '原ASIN',
                         # 'price': '价格',
                         # 'recommend': '原ASIN推荐度',
                         # 'blue_ocean_estimate': '广告蓝海度',
                         'asin': 'ASINs'}, inplace=True)

# 聚合tag表
df_group_tag = df_group_tag[
    ['related_asin', '相关竞品款数分布', '有销额竞品款数分布', '有销额推荐达标款数分布', '有销额竞品款数占比分布', '综合竞品推荐度分布', '达标推荐度占比分布', '推荐级别v1',
     'TOP5月均销额分布', 'TOP5月销额分布', '利基月GMV分布', '价格中位数分布', '价格集中度分布', '预估平均毛利率分布', '预估平均资金利用率分布', '加权FBA运费分布',
     'FBM配送占比分布', '直发FBM产品占比分布', '直发FBM销额占比分布', '直发FBM月均销额分布', 'AMZ直营销额占比分布', '大牌商标销额占比分布', '中国卖家占比分布',
     'TOP5平均LQS分布', '冒出品平均LQS分布', '冒出品A+占比分布', '冒出品视频占比分布', '冒出品QA占比分布', '动销品平均星级分布', '冒出品低星占比分布', 'TOP5销额占比分布',
     '非TOP5销额占比分布', '非TOP5月均销额分布', '动销品变体中位数', '平均开售月数分布', '冒出品新品占比分布', '新品平均月销额分布', '平均星数分布', '销级星数比分布',
     '三标新品占比分布', '月销增长率分布', '加权留评率分布', '推荐级别v2', '综合投票分', '综合推荐级别', '数据更新时间', 'data_id']]

df_group_tag = df_clear(df_group_tag, 'related_asin')

df_group_tag.rename(columns={'related_asin': '原ASIN', '动销品变体中位数': '动销品变体中位数分布'}, inplace=True)

# 8.存入数据库
sql_engine.data_to_sql(df_group, path.product_group_self, 'append', config.connet_product_db_sql)
sql_engine.data_to_sql(df_group_tag, path.product_group_tag_self, 'append', config.connet_product_db_sql)
sql_engine.data_to_sql(df_traffic_table, path.product_traffic_self, 'append', config.connet_product_db_sql)

sql_engine.connect_product(config.oe_hostname, config.oe_password, path.product_database, sql.update_sql_group_self_tag)
sql_engine.connect_product(config.sellersprite_hostname, config.sellersprite_password, path.clue_self,
                           sql.update_clue_group_sql)
