import pandas as pd

from conn import sql_engine, mysql_config as config


def data_read(pt_table):
    sql = 'select * from ' + pt_table
    df_pt_table = sql_engine.connect_pt_product(config.sellersprite_hostname, config.sellersprite_password,
                                                config.clue_shop_database, sql)
    df_pt_table = df_pt_table.drop('id', axis=1)
    return df_pt_table


table_list = ['pt_brand_insight', 'pt_brand_competing_report', 'pt_sellers_insight', 'pt_sellers_product', 'pt_task']
for table in table_list:
    table_st = table + '_st'
    df_table = data_read(table)
    df_table['task_tag'] = '某杂货公司'
    df_table_st = data_read(table_st)
    df_table_st['task_tag'] = '三态'
    df_shop_table = pd.concat([df_table, df_table_st], ignore_index=True)

    shop_table = table + '1'
    sql_engine.data_to_sql(df_shop_table, shop_table, "append", config.connet_clue_shop_db_sql)
