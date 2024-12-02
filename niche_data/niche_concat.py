import datetime
import time
import pymysql
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

import niche_data_path as path
from better.conn import sql_engine, mysql_config as config


def connect_simle(sql):
    conn_betterin = pymysql.connect(host=config.betterin_hostname, user=config.betterin_username,
                                    passwd=config.betterin_password, database=config.betterin_database, charset="utf8")
    # sql = 'select * from ' + table + ' where `数据更新时间`= "' + str(week_start) + '"'
    # df = pd.read_sql((sql + " limit 500"), conn_betterin)
    df = pd.read_sql(sql, conn_betterin)
    return df


# 连接数据库并读取数据
def connect_niche_original(sql):
    conn_oe = pymysql.connect(host=config.oe_hostname, user=config.oe_username, passwd=config.oe_password,
                              database=config.niche_database, charset="utf8")
    # df = pd.read_sql((sql + " limit 500"), conn_oe)
    df = pd.read_sql(sql, conn_oe)
    return df


# 定义一个通用的函数来获取数据
def get_data(table_name, week_start):
    sql = f'select * from {table_name} where `数据更新时间`= "{week_start}"'
    return connect_simle(sql)


def niche_save_to_sql(df, table, args, niche_conn):
    df.to_sql(table, niche_conn, if_exists=args, index=False, chunksize=1000)


start_time = time.time()

week_start = datetime.date(2023, 9, 10)
# week_start = datetime.date(2021, 9, 19)
week_end = datetime.date(2024, 1, 24)

while week_start <= week_end:
    # 获取数据
    niche_sql = 'select `利基站点`,`Niche_ID` as "niche_id",`Niche` as "所属利基",`Station` as "站点",' + \
                '`Data Update Date` as "数据更新时间",`Average Price` as "平均价格",`是否探索` from ' + \
                path.Niche_sql_name + ' where `Station`="US" and `Data Update Date`= "' + str(week_start) + '"'
    df_niche = connect_simle(niche_sql)
    if df_niche.empty:
        week_start = week_start + relativedelta(days=+7)
        continue

    df_smile_s = get_data(path.niche_smile_s, week_start)
    df_smile_m = get_data(path.niche_smile_m, week_start)
    df_smile_i = get_data(path.niche_smile_i, week_start)
    df_smile_l = get_data(path.niche_smile_l, week_start)
    df_smile_e = get_data(path.niche_smile_e, week_start)
    df_smile_a = get_data(path.niche_smile_a, week_start)

    # 单独处理价格数据和翻译数据
    price_sql = 'select `利基站点`,`平均价格`,`价格集中度` from ' + path.niche_price_original + ' where `数据更新时间`= "' + str(
        week_start) + '"'
    df_price = connect_niche_original(price_sql)
    translate_sql = 'select `利基站点`,`中文名` from ' + path.niche_translate
    df_translate = connect_simle(translate_sql)

    # 连表
    df_niche = df_niche[['利基站点', 'niche_id', '所属利基', '站点', '数据更新时间', '平均价格', '是否探索']]
    df_smile_s = df_smile_s[
        ['利基站点', 'Scale', '月均GMV', '搜索量_360', 'TOP5平均月销额', '总售出件数_360', '搜索量_90', '总售出件数_90', '最新周搜索量']]
    df_smile_m = df_smile_m[
        ['利基站点', 'Monetary', '资金利用效率AVG', '实抛偏差率', '重量分布', '实抛分布', '平均每单件数', '头程', '体积超重占比', '加权平均重量', '加权平均体积重',
         '规格分布', '加权平均FBA', '头程占比', 'FBA占比', '货值占比', 'FBA货值比', '营销前毛利反推', '存疑利基']]
    df_smile_i = df_smile_i[
        ['利基站点', 'Involution', '总点击量', '转化净值', '转化净值偏差', '转化净值分级', 'CPC', '搜索点击比', 'CR', '毛估蓝海度', 'CR_KW',
         'f(SR,AR)', 'f(VR)', 'f(NS,SP)', 'f(CC)', 'bf(CC)和f(SR,AR)', '毛估CPC因子', 'CPC', 'CPC因子', '广告蓝海度', '蓝海度差异分',
         '广告权重', '综合蓝海度', '搜索广告占比档', '搜索广告商品占比_360', '搜索广告商品占比_90', '平均缺货率', '平均缺货率_360', '平均缺货率_90', '平均产品星级',
         '商品listing平均得分_90', '周SCR_AVG', '季节性标签']]
    df_smile_l = df_smile_l[
        ['利基站点', 'Longtail', '点击最多的商品数', '长尾指数', '品牌长尾指数', '非TOP5平均月销额', '利基平均月销售额', '品牌数量_90', '品牌数量_360',
         '非TOP5单品月销量', '平均单品月销量', 'TOP5产品点击占比_360', 'TOP5产品点击占比_90', 'TOP5品牌点击占比_360', 'TOP5品牌点击占比_90', '知名品牌依赖']]
    df_smile_e = df_smile_e[
        ['利基站点', 'Emerging', '搜索量增长_360', '搜索量增长_90', '数据趋势覆盖周数', 'ASIN平均上架年数', '留评意愿强度', '平均评论数', '销量潜力得分',
         '近半年新品成功率', '近180天上架新品数_90', '近180天上架新品数_360', '近180天成功上架新品数_90', '近180天成功上架新品数_360']]
    df_smile_a = df_smile_a[
        ['利基站点', '利基图片URL', 'ASIN1', 'ASIN1品牌', 'ASIN1完整类名', 'ASIN2', 'ASIN2品牌', 'ASIN2完整类名', 'ASIN3', 'ASIN3品牌',
         'ASIN3完整类名', '知名品牌', '类目路径', '一级类目']]
    df_price = df_price[['利基站点', '平均价格', '价格集中度']]

    # df_smile_h = pd.concat([df_smile_s, df_smile_m, df_smile_i, df_smile_l, df_smile_e, df_smile_a], axis=1,
    #                        join='inner', ignore_index=False)                  # 此方法为按索引进行合并，不能指定列名，会出错

    df_niche_smile_pre = df_niche.merge(df_smile_s, how='left', on='利基站点').merge(df_smile_m, how='left', on='利基站点') \
        .merge(df_smile_i, how='left', on='利基站点').merge(df_smile_l, how='left', on='利基站点') \
        .merge(df_smile_e, how='left', on='利基站点').merge(df_smile_a, how='left', on='利基站点') \
        .merge(df_price, how='left', on='利基站点').merge(df_translate, how='left', on='利基站点')

    # 数据计算
    df_niche_smile_pre['Scale'].fillna(1, inplace=True)
    df_niche_smile_pre['SMILE打分'] = np.where(df_niche_smile_pre['Monetary'] * 1 > 0,
                                             df_niche_smile_pre['Scale'] + df_niche_smile_pre['Monetary'] +
                                             df_niche_smile_pre['Involution'] + df_niche_smile_pre['Longtail'] +
                                             df_niche_smile_pre['Emerging'],
                                             df_niche_smile_pre['Scale'] + df_niche_smile_pre['Involution'] +
                                             df_niche_smile_pre['Longtail'] + df_niche_smile_pre['Emerging'])
    conditions_h = [
        (df_niche_smile_pre['Scale'] >= 4) & (df_niche_smile_pre['Involution'] >= 2) & (
                df_niche_smile_pre['Longtail'] >= 2) & (df_niche_smile_pre['Emerging'] >= 2),
        (df_niche_smile_pre['Scale'] >= 3) & (df_niche_smile_pre['Involution'] >= 2.5) & (
                df_niche_smile_pre['Longtail'] >= 2.5) & (df_niche_smile_pre['Emerging'] >= 2),
        (df_niche_smile_pre['Scale'] >= 0) & (df_niche_smile_pre['Involution'] >= 2) & (
                df_niche_smile_pre['Longtail'] >= 2) & (df_niche_smile_pre['Emerging'] >= 2) | (
                df_niche_smile_pre['SMILE打分'] >= 11.5)]
    labels_h = ["SA/B/C", "B/C", "C"]
    df_niche_smile_pre['业务线'] = np.select(conditions_h, labels_h, default=np.nan)
    df_niche_smile_pre['业务线'] = df_niche_smile_pre['业务线'].astype('object').fillna("-")

    df_niche_smile_pre['平均价格'] = np.where(df_niche_smile_pre['平均价格_y'] * 1 > 0, df_niche_smile_pre['平均价格_y'],
                                          df_niche_smile_pre['平均价格_x'])

    # 整合到表
    df_smile_h = df_niche_smile_pre[
        ['利基站点', '利基图片URL', '所属利基', '站点', '是否探索', '中文名', '业务线', 'SMILE打分', 'Scale', 'Monetary', 'Involution',
         'Longtail', 'Emerging', '平均价格', '价格集中度', '搜索量_360', '月均GMV', 'TOP5平均月销额', '总售出件数_360', '搜索量_90', '总售出件数_90',
         '最新周搜索量', '平均每单件数', '体积超重占比', '加权平均重量', '加权平均体积重', '头程', '实抛偏差率', '规格分布', '加权平均FBA', '头程占比', 'FBA占比', '货值占比',
         'FBA货值比', '资金利用效率AVG', '营销前毛利反推', '重量分布', '实抛分布', '总点击量', '转化净值', '转化净值偏差', '转化净值分级', '搜索点击比', '毛估蓝海度', 'CR',
         'CR_KW', 'f(SR,AR)', 'f(VR)', 'f(NS,SP)', 'f(CC)', 'bf(CC)和f(SR,AR)', '毛估CPC因子', 'CPC', 'CPC因子', '广告蓝海度',
         '蓝海度差异分', '广告权重', '综合蓝海度', '搜索广告占比档', '搜索广告商品占比_90', '搜索广告商品占比_360', '平均缺货率', '平均缺货率_90', '平均缺货率_360',
         'TOP5品牌点击占比_360', 'ASIN平均上架年数', '平均产品星级', '商品listing平均得分_90', '周SCR_AVG', '季节性标签', '长尾指数', '点击最多的商品数',
         '品牌长尾指数', '品牌数量_90', '品牌数量_360', '非TOP5单品月销量', '平均单品月销量', '非TOP5平均月销额', '利基平均月销售额', 'TOP5产品点击占比_90',
         'TOP5产品点击占比_360', 'TOP5品牌点击占比_90', '知名品牌依赖', '搜索量增长_360', '搜索量增长_90', '数据趋势覆盖周数', '留评意愿强度', '平均评论数', '销量潜力得分',
         '近半年新品成功率', '近180天上架新品数_90', '近180天上架新品数_360', '近180天成功上架新品数_90', '近180天成功上架新品数_360', 'ASIN1', 'ASIN1品牌',
         'ASIN1完整类名', 'ASIN2', 'ASIN2品牌', 'ASIN2完整类名', 'ASIN3', 'ASIN3品牌', 'ASIN3完整类名', '知名品牌', '类目路径', '一级类目',
         '数据更新时间']]

    # 数据入库
    conn = sql_engine.create_conn(config.connet_db_sql)
    niche_save_to_sql(df_smile_h, path.niche_smile_h_clear, "append", conn)
    print("week：" + week_start.__str__())
    print("用时：" + (time.time() - start_time).__str__())
    week_start = week_start + relativedelta(days=+7)
print("用时：" + (time.time() - start_time).__str__())
