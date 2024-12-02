import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame

import pymysql
import niche_data_path as path
from better.conn import mysql_config as config


def connect_mysql_oe(host, user, password, database, sql):
    conn_oe = pymysql.connect(host=host, user=user, passwd=password, database=database, charset="utf8")
    # df = pd.read_sql((sql + " limit 1000"), conn_oe)
    df = pd.read_sql(sql, conn_oe)
    return df


def price_cr_convert(df):
    df = df.groupby('平均价格分布')['转化净值'].mean()
    df = pd.DataFrame(df)
    df.columns = ['转化净值平均值']
    df_new = df.reset_index()
    df_new['转化净值平均值'] = round(df_new['转化净值平均值'], 2)
    return df_new


# 作废
def power_func(x, a, b):
    return a * np.power(x, b)


def power_fit(df):
    x = np.log(df['平均价格分布'])
    y = np.log(df['转化净值平均值'])
    coeffs_cr = np.polyfit(x, y, deg=1)
    a = coeffs_cr[0]
    b = np.exp(coeffs_cr[1])
    print(f"拟合出的幂函数为：y = {b} * x ^{a}")
    plt_fit(a, b, df)
    return a, b


def plt_fit(a, b, df):
    x_fit = np.linspace(df['平均价格分布'].min(), df['平均价格分布'].max(), 1000)
    y_fit = b * x_fit ** a
    plt.scatter(df['平均价格分布'], df['转化净值平均值'], label='数据')
    plt.plot(x_fit, y_fit, color='red', label=f'拟合线：y = {b:.3f} * x^{a:.3f}')
    plt.xlabel('平均价格分布')
    plt.ylabel('平均价格分布')
    plt.title('幂函数拟合')
    plt.legend()
    # plt.show()


# 1.读取利基数据
start_time = time.time()
price_sql = 'SELECT `利基站点`,`站点`,`平均价格` FROM ' + path.niche_price + ' where `平均价格`<=150'
# price_sql = 'SELECT `利基站点`,`平均价格` FROM ' + path.niche_price
df_niche_price = connect_mysql_oe(config.betterin_hostname, config.betterin_username, config.betterin_password,
                                  config.betterin_database, price_sql)

# CR_sql = 'SELECT `利基站点`,`CR_KW`,`一级类目` FROM niche_size where `一级类目` = "automotive" and `数据更新时间`= (SELECT MAX(`数据更新时间`) FROM niche_size)'
CR_sql = 'SELECT `利基站点`,`CR_KW`,`一级类目` FROM niche_size where `一级类目` is not null and `数据更新时间`= (SELECT MAX(`数据更新时间`) FROM niche_size)'
# CR_sql = 'SELECT `利基站点`,`CR_KW`,`一级类目` FROM ' + path.Freight_sql_name
df_niche_cr = connect_mysql_oe(config.betterin_hostname, config.betterin_username, config.betterin_password,
                               config.betterin_database, CR_sql)

ppo_sql = 'SELECT `利基站点`,`Products Per Order` as "平均每单件数" FROM niche WHERE `Data Update Date`= (SELECT MAX(`Data Update Date`) FROM niche)'
df_niche_ppo = connect_mysql_oe(config.betterin_hostname, config.betterin_username, config.betterin_password,
                                config.betterin_database, ppo_sql)
print("第1步用时：" + (time.time() - start_time).__str__())

# 2.字段整合
start_time = time.time()
# df_cr = df_niche_price.merge(df_niche_cr, how='left', on='利基站点')
df_cr = df_niche_price.merge(df_niche_cr, how='inner', on='利基站点')
df_cr = df_cr.merge(df_niche_ppo, how='left', on='利基站点')
df_cr = df_cr.query('平均价格 > 0 and CR_KW > 0')

df_cr.loc[df_cr['平均每单件数'] < 1, '平均每单件数'] = 1
df_cr['per'] = np.power(df_cr['平均每单件数'], 0.9)
df_cr.loc[df_cr['per'] > 2, 'per'] = 2
df_cr['转化净值'] = round(df_cr['平均价格'] * df_cr['CR_KW'] * df_cr['per'], 4)
df_cr = df_cr.query('转化净值 > 0')
print("第2步用时：" + (time.time() - start_time).__str__())

# 3.整体拟合数据
# 获取价格分布
df_cr['平均价格分布'] = round(df_cr['平均价格'], 0)

price_min = 5
price_max = 150
price_terminal = 30
# 筛选价格分布区间为[5,150]的数据
df_cr_fit = df_cr.query('平均价格分布>=5 & 平均价格分布<=150 & 站点=="US"')

# 筛选价格分布区间为[5,150]的数据
while price_max >= price_terminal:
    df_cr_fit = df_cr_fit.loc[df_cr_fit['平均价格分布'] <= price_max]
    price_df = price_cr_convert(df_cr_fit)
    corr_cr = round(price_df['平均价格分布'].corr(price_df['转化净值平均值']), 2)
    print(f"相关系数为：{corr_cr}")
    if corr_cr >= 0.75:
        print(price_df.dtypes)
        power_fit(price_df)
        continue
    else:
        price_max -= 5
        continue

# 4.按类目拟合
# 按一级类目分组
group_df_list = list(df_cr.groupby(df_cr['一级类目'], as_index=False))

# 遍历类目
weekly_data_frame = []
for tuple_df in group_df_list:
    asin_df: DataFrame = tuple_df[1].reset_index()

# 遍历价格（5,50,200）
# 判断如果相关系数大于0.8，进入下一步，否则退出
# 按照圈定的价格范围计算幂函数公式的相关系数和指数
# 计算得出转化净值偏差


"""

import numpy as np
from scipy.optimize import curve_fit

# 定义一个幂函数
def power_func(x, a, b):
    return a * np.power(x, b)

# x 和 y 是你的两列数据，假设已经存在
x_data = np.array([1, 2, 3, 4, 5])
y_data = np.array([2, 4, 9, 16, 25])

# 使用curve_fit函数来拟合我们的数据，popt是参数的最优值
popt, pcov = curve_fit(power_func, x_data, y_data)

# 打印出a和b的值
print("a = %f, b = %f" % (popt[0], popt[1]))

import matplotlib.pyplot as plt

# 创建散点图
plt.scatter(x_data, y_data, label='Data')

# 创建拟合的函数图
x_fit = np.linspace(min(x_data), max(x_data), 1000)
y_fit = power_func(x_fit, *popt)
plt.plot(x_fit, y_fit, 'r', label='Fit: a = %0.2f, b = %0.2f' % tuple(popt))

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


"""
