from datetime import datetime
import numpy as np
import pandas as pd

import common_util, calculation_util, data_cleaning_util
from better.better.pt_product_report import pt_product_report_parameter as para


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


def match_holidays(row, holiday_kw):
    matched_holidays = [keyword for keyword in holiday_kw if keyword in row['combined_kw']]
    holidays_count = len(matched_holidays)
    holidays_str = ", ".join(matched_holidays) if matched_holidays else ""
    return holidays_count, holidays_str


def match_custom_kw(row, custom_kw):
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
    df['销额级数'] = np.fmax(1, np.log2(df['月均销额'] / 2 / (para.monthly_revenue_C / 2)))
    return df


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
