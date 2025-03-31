from conn import sql_engine, mysql_config as config
import db_util
import pt_product_report_path as path
import pt_product_sql as sql


def get_previous_periods(period):
    # 将 period 拆分成年和月
    year = int(period[:4])
    month = int(period[4:])

    # 计算上一个月
    if month == 1:
        prev_year = year - 1
        prev_month = 12
    else:
        prev_year = year
        prev_month = month - 1
    period1 = f"{prev_year}{prev_month:02d}"

    # 计算上上个月
    if prev_month == 1:
        prev2_year = prev_year - 1
        prev2_month = 12
    else:
        prev2_year = prev_year
        prev2_month = prev_month - 1
    period2 = f"{prev2_year}{prev2_month:02d}"

    return period1, period2


# -------------------------------------关联流量历史数据复用-------------------------------------

sellersprite_month = str(config.sellersprite_database)[-6:]
sellersprite_month_old, sellersprite_month_older = get_previous_periods(sellersprite_month)

# sites = ['us', 'uk', 'de', 'fr']
sites = ['us']

for site in sites:
    sellersprite_database = config.sellersprite_database + '_' + str(site)

    sellersprite_database_old = 'sellersprite_' + str(sellersprite_month_old)
    sellersprite_database_older = 'sellersprite_' + str(sellersprite_month_older)

    # sellersprite_database_old = 'sellersprite_' + str(sellersprite_month_old) + '_' + str(site)
    # sellersprite_database_older = 'sellersprite_' + str(sellersprite_month_older) + '_' + str(site)

    df_get_group = sql_engine.connect_pt_product(
        config.sellersprite_hostname, config.sellersprite_password, sellersprite_database, sql.sql_get_group
    )

    df_get_group_older = sql_engine.connect_pt_product(
        config.sellersprite_hostname, config.sellersprite_password, sellersprite_database_older, sql.sql_get_group
    )

    df_get_duplicate = sql_engine.connect_pt_product(
        config.sellersprite_hostname, config.sellersprite_password, sellersprite_database, sql.sql_get_duplicate
    )

    print(site)

    if df_get_duplicate.empty and (not df_get_group.empty) and (not df_get_group_older.empty):

        # 状态更新
        sql_engine.connect_product(
            config.sellersprite_hostname,
            config.sellersprite_password,
            sellersprite_database,
            db_util.group_traffic_status_sql(sellersprite_database_old, sellersprite_database, -10)
        )

        sql_engine.connect_product(
            config.sellersprite_hostname,
            config.sellersprite_password,
            sellersprite_database,
            db_util.group_traffic_status_sql(sellersprite_database_older, sellersprite_database, -20)
        )

        # 复用表创建
        sql_engine.connect_product(
            config.sellersprite_hostname,
            config.sellersprite_password,
            sellersprite_database,
            db_util.group_traffic_old_create_sql(path.pt_relation_traffic, path.pt_relation_traffic_old)
        )

        sql_engine.connect_product(
            config.sellersprite_hostname,
            config.sellersprite_password,
            sellersprite_database,
            db_util.group_traffic_old_create_sql(path.pt_relation_traffic, path.pt_relation_traffic_older)
        )

        sql_engine.connect_product(
            config.sellersprite_hostname,
            config.sellersprite_password,
            sellersprite_database,
            db_util.group_traffic_old_create_sql(path.pt_relevance_asins, path.pt_relevance_asins_old)
        )

        sql_engine.connect_product(
            config.sellersprite_hostname,
            config.sellersprite_password,
            sellersprite_database,
            db_util.group_traffic_old_create_sql(path.pt_relevance_asins, path.pt_relevance_asins_older)
        )

        # 复用表写入
        sql_engine.connect_product(
            config.sellersprite_hostname,
            config.sellersprite_password,
            sellersprite_database,
            db_util.group_traffic_old_insert_sql(sellersprite_database_old, sellersprite_database,
                                                 path.pt_relation_traffic, path.pt_relation_traffic_old, -10)
        )

        sql_engine.connect_product(
            config.sellersprite_hostname,
            config.sellersprite_password,
            sellersprite_database,
            db_util.group_traffic_old_insert_sql(sellersprite_database_older, sellersprite_database,
                                                 path.pt_relation_traffic, path.pt_relation_traffic_older, -20)
        )

        sql_engine.connect_product(
            config.sellersprite_hostname,
            config.sellersprite_password,
            sellersprite_database,
            db_util.group_traffic_old_insert_sql(sellersprite_database_old, sellersprite_database,
                                                 path.pt_relevance_asins, path.pt_relevance_asins_old, -10)
        )

        sql_engine.connect_product(
            config.sellersprite_hostname,
            config.sellersprite_password,
            sellersprite_database,
            db_util.group_traffic_old_insert_sql(sellersprite_database_older, sellersprite_database,
                                                 path.pt_relevance_asins, path.pt_relevance_asins_older, -20)
        )

    else:
        continue
