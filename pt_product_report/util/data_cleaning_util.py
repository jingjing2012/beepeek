"""
字符串数据清洗工具类
"""
import numpy as np
import pandas as pd


def str_replace_1(df, column_replace, str_before, str_after):
    """ 字符串替换 """
    return df[column_replace].replace(str_before, str_after, inplace=True)


def list_convert_type(df, data_list, d):
    for i in data_list:
        df[i] = df[i].replace('', np.nan)
        df[i] = df[i].astype('float64').round(decimals=d)
    return data_list


# 数据类型修正
def convert_type(df, con_str, d):
    df = df.copy()
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


# series转DataFrame
def convert_col(series, col):
    df = pd.DataFrame(series)
    df.columns = [col]
    return df
