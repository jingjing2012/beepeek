import pandas as pd
import pymysql
import mysql.connector
from sqlalchemy import create_engine
from better.conn import mysql_config as config


# 数据连接
def connect_pt_product(hostname, password, databasename, product_sql_pt):
    conn_str = conn_url(hostname, config.oe_username, password, databasename)
    with create_conn(conn_str) as conn_oe:
        df = pd.read_sql(product_sql_pt, conn_oe)
    return df

# 分页查询
def connect_pt_product_page(hostname, password, database, product_sql_pt):
    conn_pt = mysql.connector.connect(host=hostname, user=config.oe_username, password=password, database=database)
    cur = conn_pt.cursor()
    cur.execute(product_sql_pt)
    result = cur.fetchall()
    df_result = pd.DataFrame(result, columns=[i[0] for i in cur.description])
    cur.close()
    conn_pt.close()
    return df_result


# 数据库操作
def connect_product(hostname, password, database, product_sql):
    try:
        # 建立数据库连接
        conn_oe = pymysql.connect(
            host=hostname,
            user=config.oe_username,
            passwd=password,
            database=database,
            charset='utf8'
        )
        # 创建游标对象
        with conn_oe.cursor() as cur:
            # 执行SQL查询
            cur.execute(product_sql)
            # 提交更改
            conn_oe.commit()
            cur.close()
            conn_oe.close()
    except pymysql.MySQLError as e:
        print(f"Error connecting to the database: {e}")





def connect_product_update(hostname, password, database, product_sql, update_data):
    try:
        # 建立数据库连接
        conn_oe = pymysql.connect(
            host=hostname,
            user=config.oe_username,
            passwd=password,
            database=database,
            charset='utf8'
        )
        # 创建游标对象
        with conn_oe.cursor() as cur:
            # 使用 executemany 批量更新
            cur.executemany(product_sql, update_data)
            # 提交更改
            conn_oe.commit()
        # 关闭连接
        conn_oe.close()
        print("Batch update completed successfully!")
    except pymysql.MySQLError as e:
        print(f"Error connecting to the database: {e}")


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


# 数据入库
def data_to_sql(df, table, args, conn_str):
    with create_conn(conn_str) as data_conn:
        df.to_sql(table, con=data_conn, if_exists=args, index=False, chunksize=1000)
