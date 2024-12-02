import pandas as pd
import pymysql
from sqlalchemy import create_engine
import mysql_config as config


# 全局连接数据库，只需要连接一次即可
def create_conn(conn_str):
    # pymysql.install_as_MySQLdb()
    engine = create_engine(conn_str)
    # 从 Engine 获取 Connection
    conn = engine.connect()
    return conn


# 数据库连接url
def conn_url(hostname, username, password, database):
    con_db_sql = f'mysql+pymysql://{username}:{password}@{hostname}/{database}?charset=utf8mb4'
    return con_db_sql
