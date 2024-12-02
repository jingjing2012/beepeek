# 数据库的增删改查操作
import pymysql


# 添加记录(每次添加一条)
def sql_insert_once(hostname, username, password, database, insert_sql):
    conn = pymysql.connect(host=hostname, port=3306, user=username, passwd=password, db=database, charset='utf8')
    cursor = conn.cursor()
    affected_rows = cursor.execute(insert_sql)
    conn.commit()
    cursor.close()
    conn.close()
    pk = cursor.lastrowid
    print('pk=%d;affected_rows=%d' % (pk, affected_rows))


# 添加记录(每次添加多条)
def sql_insert(hostname, usernanme, password, database, insert_sql, insert_list):
    conn = pymysql.connect(host=hostname, port=3306, user=usernanme, passwd=password, db=database, charset='utf8')
    cursor = conn.cursor()
    affected_rows = cursor.executemany(insert_sql, insert_list)
    conn.commit()
    cursor.close()
    conn.close()
    pk = cursor.lastrowid
    print('pk=%d;affected_rows=%d' % (pk, affected_rows))


# 修改记录
def sql_update(hostname, username, password, database, update_sql):
    conn = pymysql.connect(host=hostname, port=3306, user=username, passwd=password, db=database, charset='utf8')
    cursor = conn.cursor()
    affected_rows = cursor.execute(update_sql)
    conn.commit()
    cursor.close()
    conn.close()
    print('affected_rows=%d' % affected_rows)


# 删除记录
def sql_delete(hostname, username, password, database, delete_sql):
    conn = pymysql.connect(host=hostname, port=3306, user=username, passwd=password, db=database, charset='utf8')
    cursor = conn.cursor()
    affected_rows = cursor.execute(delete_sql)
    conn.commit()
    cursor.close()
    conn.close()
    print('affected_rows=%d' % affected_rows)


# 查询记录
def sql_select(hostname, username, password, database, select_sql):
    conn = pymysql.connect(host=hostname, port=3306, user=username, passwd=password, db=database, charset='utf8')
    cursor = conn.cursor()
    cursor.execute(select_sql)
    conn.commit()
    cursor.close()
    conn.close()

