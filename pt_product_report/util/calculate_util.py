import numpy as np

from better.util import common_util, calculation_util


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
