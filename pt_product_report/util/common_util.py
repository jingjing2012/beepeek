"""
常用工具类
"""
import re
import pandas as pd


# 数据打标签
def get_cut(df, col, col_cut, bins_cut, labels_cut):
    df[col_cut] = pd.cut(df[col], bins_cut, labels=labels_cut, include_lowest=True)
    return df[col_cut]


# 检查是否包含中文字符
def contains_chinese(text):
    pattern = re.compile('[\u4e00-\u9fa5]+')
    return bool(pattern.search(text))
