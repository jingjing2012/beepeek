from datetime import datetime

import numpy as np
import pandas as pd

import calculation_util
import common_util
import data_cleaning_util
import pt_product_report_parameter as para


# 重量(g)转换
def weight_g(df):
    # 替换错误单位
    for error_unit, replacement in para.replace_weight_error_dict.items():
        df['weight'] = df['weight'].str.replace(error_unit, replacement, regex=False)

    # 一次性分割并创建新列
    weight_split = df['weight'].str.split(" ", expand=True)
    df['重量值'] = weight_split[0]
    df['单位'] = weight_split[1]

    # 去除不合法单位和重量值
    df.loc[~df['单位'].isin(para.replace_weight_unit_list), '单位'] = np.nan
    df['重量值判断'] = df['重量值'].str.replace(".", "")
    df.loc[~df['重量值判断'].str.isdecimal(), '重量值'] = "-1"
    df['重量值'] = np.where(df['重量值判断'] == "-1", np.nan, df['重量值'])

    # 计算换算值
    df['换算'] = df['单位'].replace(para.replace_weight_dict, regex=False)

    # 计算重量
    return np.where(df['重量值'].astype(float) * 1 > 0, round(df['重量值'].astype(float) * df['换算'].astype(float), 4), np.nan)


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


# 预估FBA占比
def fba_rate(df):
    return np.where(df['fba_fees'] * 1 > 0, np.fmin(1, df['fba_fees'] / df['price']), para.fba_fees_rate)


# 预估头程占比
def pre_rate(df):
    return np.where(df['预估FBA占比'] * 1 > 0, np.fmin(1, df['预估FBA占比'] / 2.5), para.pre_fees_rate)


# 预估货值占比
def product_rate(df):
    return common_util.get_cut(df, 'price', para.product_price_list, para.product_fees_list)


# 预估毛利率
def profit_rate(df, site):
    if site == 'US':
        vat = para.vat_us
    elif site == 'UK':
        vat = para.vat_uk
    else:
        vat = para.vat_de
    df['预估毛利率_FBM'] = df['gross_margin'] - df['预估头程占比'] * 2 - para.product_fees_rate - vat
    df['预估毛利率_FBA'] = df['gross_margin'] - df['预估头程占比'] - para.product_fees_rate - vat
    df['预估毛利率_反推'] = np.where(
        (df['直发FBM可能性'] >= 1) & (df['gross_margin'] >= para.gross_margin_upper), df['预估毛利率_FBM'], df['预估毛利率_FBA'])

    margin_non_zero = abs(df['gross_margin']) > 0
    fallback_margin = 1 - df['预估FBA占比'] - df['预估头程占比'] - para.referral_fees_rate - df['预估货值占比'] - vat
    safe_margin = np.where(margin_non_zero, df['预估毛利率_反推'], fallback_margin)
    return np.clip(safe_margin, -1, 1)


# 毛利率级别
def profit_rate_tag(df):
    df['毛利率级别_上限'] = np.fmin(calculation_util.get_mround(df, '预估毛利率', '毛利率级别_上限', 0.05), para.gross_rate_upper)
    df['毛利率级别_下限'] = np.fmax(calculation_util.get_mround(df, '预估毛利率', '毛利率级别_下限', -0.05), para.gross_rate_lower)
    return np.where(df['预估毛利率'] >= 0, df['毛利率级别_上限'], df['毛利率级别_下限'])


# 毛估资金利用率
def product_revenue(df):
    return df['预估毛利率'] / (df['预估头程占比'] + para.product_fees_rate)


# 开售月数计算
def month_available(df):
    current_date = pd.to_datetime(datetime.now().date())
    df['date_available'] = pd.to_datetime(df['date_available'], errors='coerce')
    df['开售天数'] = (current_date - df['date_available']).dt.days
    # df['date_available'] = np.where(df['开售天数'] * 1 > 0, df['date_available'], pd.to_datetime('1900-01-01'))
    df['头程月数'] = np.where((df['seller_type'] == "FBA") & (df['开售天数'] > 15), 0.5, 0)
    return np.fmax(round(df['开售天数'] / 30 - df['头程月数'], 1), 0.1)


# 销额级数计算
def get_revenue(df, site):
    if site == 'US':
        monthly_revenue_C = para.monthly_revenue_C
    elif site == 'UK':
        monthly_revenue_C = para.monthly_revenue_C * para.exchange_us_uk
    else:
        monthly_revenue_C = para.monthly_revenue_C * para.exchange_us_de
    df['monthly_revenue_increase'] = df['monthly_revenue_increase'].fillna(0)
    df['近两月销额'] = np.where(df['monthly_revenue_increase'] <= (-1), np.nan, df['monthly_revenue'] + (
            df['monthly_revenue'] / (1 + df['monthly_revenue_increase'] + 0.00001)))
    df['月均销额'] = np.where(df['近两月销额'] * 1 > 0, df['近两月销额'] / np.fmax(np.fmin(df['开售月数'] - 1, 1), 0.5), np.nan)
    return np.fmax(1, np.log2(df['月均销额'] / 2 / (monthly_revenue_C / 2)))


# 高资金利用率
def high_product_revenue(df):
    product_revenue_non_zero = abs(df['毛估资金利用率']) > 0
    return np.where(product_revenue_non_zero, df['毛估资金利用率'] / para.product_revenue_std - 1, 0)


# 高销低LQS
def high_sale_low_lqs(df):
    conditions_lqs_1 = (df['预估毛利率'] >= -0.05) & (df['lqs'] * 1 > 0) & (df['lqs'] <= 8)
    conditions_lqs_2 = (df['开售月数'] >= 24) & (df['rating'] >= 4) & (df['ratings'] >= 10) & (
            df['预估毛利率'] >= -0.15) & (df['lqs'] * 1 > 0) & (df['lqs'] <= 8)
    df['高销低LQS_pre'] = np.fmax(0, np.fmin(3, 0.5 + df['销额级数'] * para.lqs_std / df['lqs']))
    return np.where(conditions_lqs_1 | conditions_lqs_2, df['高销低LQS_pre'], 0)


# 月均QA数
def qa_per_month(df):
    df['开售月数_QA'] = np.fmin(df['开售月数'], 24)
    return np.where(df['qa'] * 1 > 0, round(df['qa'] / df['开售月数_QA'], 1), 0)


# 长期上架系列指标
def long_term_sale(df, site):
    if site == 'US':
        monthly_revenue_C = para.monthly_revenue_C
    elif site == 'UK':
        monthly_revenue_C = para.monthly_revenue_C * para.exchange_us_uk
    else:
        monthly_revenue_C = para.monthly_revenue_C * para.exchange_us_de
    conditions_available = (df['开售月数'] >= para.available_std) & (df['monthly_revenue'] >= monthly_revenue_C)

    long_term_sale_no_qa = np.where(conditions_available, np.fmax(-1, para.qa_std - df['月均QA数'] / para.qa_std), 0)
    long_term_sale_no_a = np.where(conditions_available & (df['ebc_available'] != "Y"), 1, 0)
    long_term_sale_no_video = np.where(conditions_available & (df['video_available'] != "Y"), 1, 0)

    return np.vstack([long_term_sale_no_qa, long_term_sale_no_a, long_term_sale_no_video]).T


# 类轻小直发FBM
def light_small_fbm(df):
    return np.where(df['直发FBM可能性'] * 1 > 0, np.fmax(0, 1 + df['销额级数'] * np.fmin(1, df['直发FBM可能性'] / 2)), 0)


# 差评好卖
def low_star_high_sale(df, site):
    if site == 'US':
        monthly_revenue_C = para.monthly_revenue_C
    elif site == 'UK':
        monthly_revenue_C = para.monthly_revenue_C * para.exchange_us_uk
    else:
        monthly_revenue_C = para.monthly_revenue_C * para.exchange_us_de
    margin_non_zero = abs(df['gross_margin']) > 0
    return np.where((df['开售月数'] >= para.available_std) & (df['monthly_revenue'] >= monthly_revenue_C / 2) & (
            df['ratings'] >= 10) & (df['rating'] >= 3) & (df['rating'] < 4) & margin_non_zero & (
                            df['category_bsr_growth'] >= -0.5), 0.5 + df['销额级数'] * (4.5 - df['rating']), 0)


# 疑似节日性
def match_holidays(row, holiday_kw):
    matched_holidays = [keyword for keyword in holiday_kw if keyword in row['combined_kw']]
    holidays_count = len(matched_holidays)
    holidays_str = ", ".join(matched_holidays) if matched_holidays else ""
    return holidays_count, holidays_str


# 知名品牌
def famous_brand(df):
    return np.where(df['疑似知名品牌'] > 0, -df['疑似知名品牌'] / np.where(df['疑似节日性'] * 1 > 0, 2, 1), 0)


# 是否个人定制
def match_custom_kw(row):
    custom_kws = [keyword for keyword in para.custom_kw if keyword in row['title']]
    custom_kws_count = len(custom_kws)
    return custom_kws_count


# 是否翻新
def get_renewed(df):
    return df['title'].astype(str).str.contains('renewed', na=False, regex=True).astype(int)


# 平均变体月销额等级
def revenue_per_variations(df, site):
    if site == 'US':
        monthly_revenue_C = para.monthly_revenue_C
    elif site == 'UK':
        monthly_revenue_C = para.monthly_revenue_C * para.exchange_us_uk
    else:
        monthly_revenue_C = para.monthly_revenue_C * para.exchange_us_de
    return np.where(df['monthly_revenue'] * 1 > 0,
                    np.log2(df['monthly_revenue'] / df['variations'] / (monthly_revenue_C / 2)), 0)


# 变体等级
def variations_tag(df):
    return np.log2(df['variations'])


# 少变体
def few_variations(df):
    return np.where(df['variations'] <= 2, 0, np.fmax(-10, np.fmin(0, df['平均变体月销额等级'] - 0.5 * df['变体等级'] + 0.5)))


def product_new(df, site):
    if site == 'US':
        monthly_revenue_C = para.monthly_revenue_C
    elif site == 'UK':
        monthly_revenue_C = para.monthly_revenue_C * para.exchange_us_uk
    else:
        monthly_revenue_C = para.monthly_revenue_C * para.exchange_us_de

    df = df.fillna(0)

    # 新品条件
    conditions_new_product = df['开售月数'] < para.available_std

    product_new_high_sale = np.where(
        conditions_new_product & (df['monthly_revenue'] >= 0),
        np.fmax(0, np.log2(df['monthly_revenue'] / df['开售月数'] / (monthly_revenue_C / para.revenue_month_C))),
        0)

    product_new_high_star = np.where(
        conditions_new_product & (df['rating'] >= 4) & (df['ratings'] >= 10),
        np.fmax(0, np.fmin(2, np.log(df['ratings'] / df['开售月数'] / df['variations'] / np.log(5)))),
        0)

    product_new_nsr = np.where(conditions_new_product & (df['new_release'] == "Y"), 1, 0)

    product_new_ac = np.where(conditions_new_product & (df['ac'] == "Y"), 1, 0)

    return pd.DataFrame({
        "product_new_high_sale": product_new_high_sale,
        "product_new_high_star": product_new_high_star,
        "product_new_nsr": product_new_nsr,
        "product_new_ac": product_new_ac
    })


# 少评好卖
def few_star_high_sale(df):
    df['销级星数比'] = np.where(df['rating'] * 1 > 0, df['销额级数'] / round(2 + df['rating'] / 100), 0)
    return np.where(df['销额级数'] * 1 > 0, np.fmax(0, round(df['销级星数比'] - 1, 2)), 0)


# 加权CPC计算
def cpc_avg_old(df):
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
def sp_amz_related_old(df):
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

    df['contains_chinese'] = df['keyword'].apply(common_util.contains_chinese)
    df['异常'] = np.where(df['contains_chinese'] == False, 0, 1)

    df['ASIN_KW相关度'] = np.where(df['异常'] != 0, 0, df['ASIN_KW相关度'])
    return df


# 排序
def sort_and_rank_old(df):
    df = df.groupby('asin', group_keys=False).apply(
        lambda x: x.sort_values(by=['ASIN_KW相关度', 'searches'], ascending=[False, False]))
    df['row_rank'] = df.reset_index(drop=True).index
    df['rank'] = df['row_rank'].groupby(df['asin']).rank()
    return df


# 加权CPC计算
def cpc_avg(df):
    row_related = np.array(df['ASIN_KW相关度'])
    row_bid_amz = np.array(df['bid_rangeMedian'])
    bid_avg = sum(row_bid_amz * row_related * (row_bid_amz > 0)) / sum(row_related * (row_bid_amz > 0))
    return bid_avg


# 蓝海度计算
def blue_ocean_estimate(p_cr_a, p_cr_b, df):
    df['预期CR'] = p_cr_a * (df['price'].pow(p_cr_b))
    data_cleaning_util.convert_type(df, df['预期CR'], 4)
    df['转化净值'] = df['price'] * df['预期CR']
    df['预期CPC'] = df['转化净值'] * para.product_acos
    df['CPC因子'] = np.where(df['预期CPC'] * df['加权CPC'] > 0, df['预期CPC'] / df['加权CPC'], np.nan)
    df['市场蓝海度'] = np.where(df['CPC因子'] > 0, para.MS_a + para.MS_b /
                           (1 + (para.MS_e ** (- (df['CPC因子'] - para.MS_cs * para.MS_c)))), np.nan)
    return df


# 排序
def sort_and_rank(df):
    df = df.groupby('asin', group_keys=False).apply(lambda x: x.sort_values(by=['ASIN_KW相关度'], ascending=[False]))
    df['row_rank'] = df.reset_index(drop=True).index
    df['rank'] = df['row_rank'].groupby(df['asin']).rank()
    return df


# 排序
def sort_and_rank_group(df):
    df = df.sort_values(by=['parent', 'ac', '开售月数', 'update_time'], ascending=[True, False, False, False])
    df['rank'] = df.groupby('parent').cumcount() + 1
    return df['rank']


# PMI计算
def pmi_score(row):
    P_score, M_score = 0, 0
    P_tags, M_tags, I_tags = [], [], []
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


# 综合推荐度计算
def product_recommend(df, col_a, col_b, col_str):
    asin_recommend = df.groupby(df['related_asin']).apply(
        lambda x: calculation_util.product_avg(x, col_a, col_b) if not x.empty else np.nan, include_groups=False)
    return data_cleaning_util.convert_col(asin_recommend, col_str)
