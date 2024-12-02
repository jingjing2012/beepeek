"""
字符串替换
"""

import pandas as pd


def str_replace_1(df, column_replace, str_before, str_after):
    return df[column_replace].replace(str_before, str_after, inplace=True)


def str_replace_2(df, column_replace, str_before, str_after):
    return df[column_replace].replace(str_before, str_after, inplace=True)
