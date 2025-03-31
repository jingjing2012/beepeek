import pandas as pd
import pymysql
from sqlalchemy import create_engine
import mysql_config as config
import niche_sql


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


def niche_key_sql(id_start, id_increment):
    sql_niche_key = """
    SELECT
    *
    FROM
    pt_niche
    """
    sql_asin = sql_niche_key + ' WHERE id >' + str(id_start) + ' ORDER BY id ASC LIMIT ' + str(id_increment)
    return sql_asin


def niche_trends_sql(id_start, id_increment):
    sql_niche_trends = 'select *,pt_niche_trends.product_count as "product_count_7" from (' + \
                       niche_key_sql(id_start, id_increment) + \
                       ') pt_niche_title left join pt_niche_trends on pt_niche_title.niche_id = pt_niche_trends.niche_id'
    return sql_niche_trends


def niche_asin_sql(id_start, id_increment):
    sql_niche_asin = 'select * from (' + niche_key_sql(id_start, id_increment) + \
                     ') pt_niche_title left join pt_niche_commodity on ' \
                     'pt_niche_title.niche_id =pt_niche_commodity.niche_id left join pt_commodity on ' \
                     'pt_niche_commodity.asin =pt_commodity.asin'
    return sql_niche_asin


def niche_keywords_sql(id_start, id_increment):
    sql_niche_keywords = 'select ' \
                         'pt_niche_title.niche_title,' \
                         'pt_niche_title.mkid,' \
                         'pt_keywords.keyword,' \
                         'pt_keywords.niche_id,' \
                         'pt_keywords.search_term_id,' \
                         'pt_keywords.search_conversion_rate,' \
                         'pt_keywords.search_volume_t_90,' \
                         'pt_keywords.search_volume_qoq,' \
                         'pt_keywords.search_volume_yoy,' \
                         'pt_keywords.search_volume_t_360,' \
                         'pt_keywords.search_volume_growth_t_360_yoy,' \
                         'pt_keywords.search_conversion_rate_t_360,' \
                         'pt_keywords.click_share,' \
                         'pt_keywords.click_share_t_360 from (' + \
                         niche_key_sql(id_start, id_increment) + \
                         ') pt_niche_title left join pt_keywords on pt_niche_title.niche_id =pt_keywords.niche_id'
    return sql_niche_keywords


def niche_keywords_top5_sql(id_start, id_increment):
    sql_niche_keywords_top5 = 'SELECT * FROM (' \
                              'SELECT *,ROW_NUMBER() OVER (PARTITION by keywords.niche_id ' \
                              'ORDER BY keywords.search_volume_t_360 DESC) AS sv_rank from(' + \
                              niche_keywords_sql(id_start, id_increment) + \
                              ' WHERE pt_keywords.id is NOT NULL) keywords) keywords_top WHERE keywords_top.sv_rank<=5'
    return sql_niche_keywords_top5


def niche_cpc_sql(id_start, id_increment):
    sql_niche_cpc = 'SELECT ' \
                    'niche_title,' \
                    'mkid,' \
                    'keyword,' \
                    'niche_id,' \
                    'search_volume_t_360,' \
                    'bid_rangeMedian,' \
                    'bid_rangeEnd FROM(' + \
                    niche_keywords_top5_sql(id_start, id_increment) + \
                    ') keywords_top5 LEFT JOIN (' + niche_sql.niche_keywords_cpc_sql + ') keywords_cpc' + \
                    ' ON keywords_top5.keyword = keywords_cpc.keyword_amz WHERE keywords_cpc.parent_id IS NOT NULL'
    return sql_niche_cpc
