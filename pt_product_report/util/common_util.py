"""
常用工具类
"""
import re
import pandas as pd


# 数据打标签
def get_cut(df, col_str, bins_cut, labels_cut):
    return pd.cut(df[col_str], bins_cut, right=False, labels=labels_cut, include_lowest=True)


# 检查是否包含中文字符
def contains_chinese(text):
    pattern = re.compile('[\u4e00-\u9fa5]+')
    return bool(pattern.search(text))
