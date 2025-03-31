"""
数据计算工具类
"""
import numpy as np
import pandas as pd
import data_cleaning_util as clean_util


# mround函数实现
def get_mround(df, df_str, df_str_mround, mround_n):
    df[df_str_mround] = round(df[df_str] / mround_n) * mround_n
    return df[df_str_mround]


# 百分比转换
def percent_convert(df, df_str):
    df[df_str] = pd.to_numeric(df[df_str].str.rstrip('%'), errors='coerce') / 100
    return clean_util.convert_type(df, df_str, 4)


# 加权计算
def product_avg(df, col_a, col_b):
    row_a = np.array(df[col_a])
    row_b = np.array(df[col_b])
    if np.sum(row_b * (row_b > 0)) != 0:
        row_avg = np.sum(row_a * row_b * (row_b > 0)) / np.sum(row_b * (row_b > 0))
    else:
        row_avg = np.nan
    return row_avg


# 产品数计算
def product_count(df, group_str, count_str, col_str):
    asins_count = df[count_str].groupby(df[group_str]).count()
    return clean_util.convert_col(asins_count, col_str)


# 去重计数
def product_unique(df, group_str, count_str, col_str):
    asins_count = df[count_str].groupby(df[group_str]).unique()
    return clean_util.convert_col(asins_count, col_str)


# 产品销额计算
def product_sum(df, group_str, sum_str, col_str):
    asins_revenue = df[sum_str].groupby(df[group_str]).sum()
    return clean_util.convert_col(asins_revenue, col_str)


# 产品均值计算
def product_mean(df, group_str, mean_str, col_str):
    asins_mean = df[mean_str].groupby(df[group_str]).mean()
    return clean_util.convert_col(asins_mean, col_str)


# 中位数计算
def product_median(df, group_str, median_str, col_str):
    asins_median = df[median_str].groupby(df[group_str]).median()
    return clean_util.convert_col(asins_median, col_str)


# 标准差计算
def product_std(df, group_str, std_str, col_str):
    asins_std = df[std_str].groupby(df[group_str]).std()
    return clean_util.convert_col(asins_std, col_str)


# 众数计算
def product_mode(df, group_str, std_str, col_str):
    asins_mode = df.groupby(group_str)[std_str].agg(lambda x: x.value_counts().idxmax() if not x.empty else np.nan)
    return clean_util.convert_col(asins_mode, col_str)
