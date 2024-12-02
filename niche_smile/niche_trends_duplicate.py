import numpy as np
import pandas as pd
import warnings
from pandas.errors import SettingWithCopyWarning
import datetime
import time
from dateutil.relativedelta import *

import mysql_config as config
import pymysql
from better.better.niche_data import niche_data_path as path
from better.conn import sql_engine


# 连接数据库并读取数据
def connect_niche_original(sql):
    conn_oe_str = sql_engine.conn_url(config.oe_hostname, config.oe_username, config.oe_password, config.niche_database)
    with sql_engine.create_conn(conn_oe_str) as conn_oe:
        df = pd.read_sql(sql, conn_oe)
    return df


# 表名转换操作
def connect_niche(sql):
    try:
        conn_oe = pymysql.connect(host=config.oe_hostname, user=config.oe_username, passwd=config.oe_password,
                                  database=config.niche_database, charset='utf8')
        with conn_oe.cursor() as cur:
            cur.execute(sql)
            conn_oe.commit()
            cur.close()
            conn_oe.close()
    except pymysql.MySQLError as e:
        print(f"Error connecting to the database: {e}")


# 字符串格式转换
def convert_str(df, col):
    df[col] = df[col].astype(str)
    df[col] = df[col].str.strip()
    return df[col]


# df去重
def df_cleaning(df, clear_id):
    convert_str(df, clear_id)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[clear_id])
    df = df.drop_duplicates(subset=[clear_id], inplace=True, ignore_index=True)
    return df


# 导入目标库
def oe_save_to_sql(df, table, args, oe_conn):
    df.to_sql(table, oe_conn, if_exists=args, index=False, chunksize=1000)


# ---------------------------------------------------------------------------------------------------------

# 忽略与 Pandas SettingWithCopyWarning 模块相关的警告
warnings.filterwarnings('ignore', category=SettingWithCopyWarning)
warnings.filterwarnings("ignore", message="pandas only support SQLAlchemy connectable.*")
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")

week_start = datetime.date(2021, 8, 29)
week_end = datetime.date.today()
table_clear = path.niche_trends_original
table_new = table_clear + '_clear'

start_time = time.time()
while week_start <= week_end:
    clear_sql = 'select * from ' + table_clear + ' where `所属日期`= "' + str(week_start) + '"'
    df_clear = connect_niche_original(clear_sql)

    if df_clear.empty:
        week_start = week_start + relativedelta(days=+7)
        continue

    # 数据清洗
    df_clear['time_rank'] = df_clear['数据更新时间'].groupby(df_clear['利基站点']).rank(method='first')

    df_clear = df_clear[df_clear['time_rank'] == 1]
    df_cleaning(df_clear, '利基站点')

    df_clear = df_clear.drop(['id', 'time_rank'], axis=1)

    # 数据入库
    conn = sql_engine.create_conn(config.connet_niche_db_sql)
    oe_save_to_sql(df_clear, table_new, 'append', conn)
    week_start = week_start + relativedelta(days=+7)
    print("week：" + week_start.__str__())

# 将niche_trends_original和niche_trends_original_clear交换表名
rename_niche_trends_original_sql = "RENAME TABLE " + table_clear + " TO table_temp," + \
                                   table_new + " TO " + table_clear + ",table_temp TO " + table_new
connect_niche(rename_niche_trends_original_sql)

# 清空niche_trends_original_clear下次备用
clear_niche_trends_original_sql = "TRUNCATE TABLE " + table_new
connect_niche(clear_niche_trends_original_sql)
print("用时：" + (time.time() - start_time).__str__())
