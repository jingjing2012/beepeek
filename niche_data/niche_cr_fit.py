import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas import DataFrame

import pymysql
import niche_data_path as path
from better.conn import sql_engine, mysql_config as config


def connect_mysql_oe(host, user, password, database, sql):
    conn_oe = pymysql.connect(host=host, user=user, passwd=password, database=database, charset="utf8")
    df = pd.read_sql(sql, conn_oe)
    return df


def price_cr_convert(df):
    df = df.groupby('平均价格分布')['转化净值'].mean()
    df = pd.DataFrame(df)
    df.columns = ['平均值项:转化净值']
    df_new = df.reset_index()
    df_new['平均值项:转化净值'] = round(df_new['平均值项:转化净值'], 2)
    return df_new


def price_cr_fit(df, corr_val, price, df_r):
    # 对数据进行对数变换
    x = np.log(df['平均价格分布'])
    y = np.log(df['平均值项:转化净值'])
    # 拟合线性模型
    coeffs = np.polyfit(x, y, deg=1)
    # 输出拟合参数
    a = coeffs[0]
    b = np.exp(coeffs[1])
    # print(f'拟合出的幂函数为: y = {b} * x^{a}')
    df_r['pearson'] = corr_val
    df_r['power_a'] = a
    df_r['power_b'] = b
    df_r['price_max'] = price
    return df_r


def price_cr_plot(a, b, df):
    # 使用拟合参数生成拟合线
    xfit = np.linspace(df['平均价格分布'].min(), df['平均价格分布'].max(), 1000)
    yfit = b * xfit ** a
    # 绘制散点图和拟合线
    plt.scatter(df['平均价格分布'], df['平均值项:转化净值'], label='数据')
    plt.plot(xfit, yfit, color='red', label=f'拟合线: y = {b:.3f} * x^{a:.3f}')
    plt.xlabel('平均价格分布')
    plt.ylabel('平均值项:转化净值')
    plt.title('幂函数拟合')
    plt.legend()
    plt.show()


def price_cr_judge(df, df_r):
    price_max = 150
    while price_max >= 30:
        df = df.loc[df['平均价格分布'] <= price_max]
        df_fit = price_cr_convert(df)
        corr_cr = df_fit['平均价格分布'].corr(df_fit['平均值项:转化净值'])
        # print(f"平均价格分布和平均值项:转化净值的相关系数为: {corr_cr}")
        if corr_cr >= 0.75:
            # print(price_max)
            return price_cr_fit(df_fit, corr_cr, price_max, df_r)
        price_max -= 5


def df_clear(df):
    # df = df.replace([np.inf, -np.inf], np.nan)
    df = df.drop_duplicates()
    return df


def save_to_sql(df, table, args, oe_conn):
    df_clear(df)
    df.to_sql(table, oe_conn, if_exists=args, index=False)


'''
------------------------------------------------------------------------------------------------------------------
'''
CR_sql = 'SELECT `平均价格`,`转化净值`,`一级类目`,`站点` FROM niche_smile where `平均价格`<=150 and `站点`="DE" ' \
         'and `一级类目` is not null and `数据更新时间`= (SELECT MAX(`数据更新时间`) FROM niche_smile)'
df_niche_cr = connect_mysql_oe(config.betterin_hostname, config.betterin_username, config.betterin_password,
                               config.betterin_database, CR_sql)

CR_date_sql = 'SELECT `站点`,MAX(`数据更新时间`) "数据更新时间" FROM niche_smile where `站点`="DE"'
df_cr_fit = connect_mysql_oe(config.betterin_hostname, config.betterin_username, config.betterin_password,
                             config.betterin_database, CR_date_sql)

df_niche_cr['平均价格分布'] = round(df_niche_cr['平均价格'])
df_cr = df_niche_cr.query('平均价格分布>=5')

df_cr_fit = price_cr_judge(df_cr, df_cr_fit)

niche_smile_conn = sql_engine.create_conn()
save_to_sql(df_cr_fit, path.niche_category_cr_fit, "append", niche_smile_conn)

# 4.按类目拟合
# 按一级类目分组
group_df_list = list(df_cr.groupby(df_cr['一级类目'], as_index=False))

# 遍历类目
category_cr_fit_frame = pd.DataFrame()
for tuple_df in group_df_list:
    cr_df: DataFrame = tuple_df[1].reset_index()
    df_cr_fit['一级类目'] = cr_df['一级类目']
    category_cr_fit_df = price_cr_judge(cr_df, df_cr_fit)
    if category_cr_fit_df is None:
        print(f"empty: {df_cr_fit['一级类目']}")
        continue
    niche_smile_conn = sql_engine.create_conn()
    save_to_sql(category_cr_fit_df, path.niche_category_cr_fit, "append", niche_smile_conn)
    # print(category_cr_fit_df)
    # category_cr_fit_frame.append(category_cr_fit_df)
    # print(category_cr_fit_frame)
    # print(type(category_cr_fit_frame))

# df_category_cr_fit = pd.concat(category_cr_fit_frame)

# print(df_category_cr_fit)
# niche_smile_conn = sql_engine.create_conn()
# save_to_sql(df_category_cr_fit, path.niche_category_cr_fit, "append", niche_smile_conn)
