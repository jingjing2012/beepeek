import json
import traceback
from datetime import datetime

import pymysql
from dbutils.pooled_db import PooledDB



def singleton(cls):

    instances = {}

    def _singleton(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return _singleton


# 单例+连接池
# @singleton
class MySqlHelper(object):
    def __init__(self, database_name=None):
        config = FileRW.read("SellerspriteConfig.json")
        host = config["host"]
        port = config["port"]
        dbuser = config["user"]
        password = config["password"]
        database = config["database"] if database_name is None else database_name

        self.pool = PooledDB(
            creator=pymysql,  # 使用链接数据库的模块
            maxconnections=16,  # 连接池允许的最大连接数，0和None表示不限制连接数
            mincached=2,  # 初始化时，链接池中至少创建的空闲的链接，0表示不创建
            maxcached=16,  # 链接池中最多闲置的链接，0和None不限制
            maxshared=0, # 链接池中最多共享的链接数量，0和None表示全部共享。PS: 无用，因为pymysql和MySQLdb等模块的 threadsafety都为1，所有值无论设置为多少，_maxcached永远为0，所以永远是所有链接都共享。
            blocking=True,  # 连接池中如果没有可用连接后，是否阻塞等待。True，等待；False，不等待然后报错
            maxusage=None,  # 一个链接最多被重复使用的次数，None表示无限制
            setsession=[],  # 开始会话前执行的命令列表。如：["set datestyle to ...", "set time zone ..."]
            ping=0,
            # ping MySQL服务端，检查是否服务可用。# 如：0 = None = never, 1 = default = whenever it is requested, 2 = when a cursor is created, 4 = when a query is executed, 7 = always
            host=host,
            port=int(port),
            user=dbuser,
            password=password,
            database=database,
            charset='utf8mb4'
        )

    def create_conn_cursor(self):
        conn = self.pool.connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        return conn, cursor

    # 查询所有
    def execute_fetch_all(self, sql, args=None):
        try:
            conn, cursor = self.create_conn_cursor()
            cursor.execute(sql, args)
            execute_result = cursor.fetchall()
            cursor.close()
            conn.close()
            return execute_result
        except Exception as e:
            print(e)
            print(traceback.format_exc())

    # 执行查询数量、插入、更新、删除
    def execute_sql(self, sql, args=None):
        try:
            conn, cursor = self.create_conn_cursor()
            execute_result = cursor.execute(sql, args)
            conn.commit()
            print("记录数:" + str(execute_result))
            cursor.close()
            conn.close()
            return execute_result
        except Exception as e:
            print(e)
            print(traceback.format_exc())

    # 批量插入、更新、删除 args一个可迭代对象。
    def execute_sql_batch(self, sql, args=None):
        try:
            conn, cursor = self.create_conn_cursor()
            executemany_result = cursor.executemany(sql, args)
            conn.commit()
            print("记录数:" + str(executemany_result))
            cursor.close()
            conn.close()
            return executemany_result
        except Exception as e:
            print(e)
            print(traceback.format_exc())


if __name__ == '__main__':

    sql_helper = MySqlHelper()

    # ks = ('pokémon blanket', 'shake')
    # rest = sql_helper.execute_fetch_all(f"select `id`,`keyword` from `pt_keywords` where keyword in {ks}")
    # print(rest)

    # 查询所有
    # rest = sql_helper.execute_fetch_all("select id, country, asin from `pt_product_get_cpc` where status=0  LIMIT 1000")
    # print(rest)

    # 单条更新
    # sql_helper.update("update user SET name=%s WHERE  id=%s",("yinwangba",1))

    # 单条插入
    # sql_helper.execute_sql("insert into `pt_keywords` (`top_clicked_product_2`, `top_clicked_product_3`) values (%s,%s)", ('T4', 'uXZ0'))

    # 批量更新
    # sql_helper.execute_sql_batch("update `pt_keywords` set `search_term_id`=%s, `niche_id`=%s where `keyword`=%s",
    #                              [['9051a6b18ba5a99d802e0a2ef976cfe7', "9051a6b18ba5a99d802e0a2ef976cfe7", 'stocking stuffers for adults']])

    # 查询表是否存在
    result = sql_helper.execute_sql("SHOW TABLES LIKE 'pt_keywords';")
    print(result)

    rest = sql_helper.execute_fetch_all("SELECT count(*) FROM pt_product_get_cpc WHERE status=0")
    print(rest[0]["count(*)"])
