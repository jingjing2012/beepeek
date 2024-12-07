"""
去除重复值和去除空值
"""

import numpy as np


# df去重
def df_cleaning(df, clear_id):
    df = df.replace('none', np.nan, regex=False)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.drop_duplicates(subset=[clear_id])
    df = df.dropna(subset=[clear_id])
    return df
