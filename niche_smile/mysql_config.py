# 连接mysql用到的配置信息放在这里


# 市调数据源库
better_hostname = "175.178.34.126"
better_port = 3306  # ssh端口
better_username = 'beeinvest'
better_password = "$bee.invest$"
better_database = "BeeInvest"
connet_better_db_sql = 'mysql+pymysql://' + better_username + ':' + better_password + '#@' + better_hostname + '/' + \
                       better_database + '?charset=utf8mb4'

# 利基目标数据库
betterin_hostname = "rm-8vbo07e0tf0ojgx72jo.mysql.zhangbei.rds.aliyuncs.com"
betterin_port = 3306  # ssh端口
betterin_username = "betterin"
betterin_password = "Bt220518#"
betterin_database = "betterin"
connet_betterin_db_sql = 'mysql+pymysql://' + betterin_username + ':' + betterin_password + '@' + \
                         betterin_hostname + '/' + betterin_database + '?charset=utf8mb4'

# 利基数据源
oe_hostname = "rm-8vbth3p40ky8y57v8wo.mysql.zhangbei.rds.aliyuncs.com"
oe_port = 3306  # ssh端口
oe_username = 'betterniche'
oe_password = "betterreport168#"

# 利基数据源库
# 数据日期
oe_data_date = "20250301"
# 数据市场
oe_data_country = "us"
# oe_data_country = "uk"
# oe_data_country = "de"
# 数据库
oe_database = "oe_" + oe_data_country + "_" + oe_data_date
connet_oe_db_sql = 'mysql+pymysql://' + oe_username + ':' + oe_password + '@' + oe_hostname + '/' + \
                   oe_database + '?charset=utf8mb4'

# 利基历史数据库
niche_database = "niche_original"
connet_niche_db_sql = 'mysql+pymysql://' + oe_username + ':' + oe_password + '@' + oe_hostname + '/' + \
                      niche_database + '?charset=utf8mb4'

# 利基数据源表
oe_table_id = "pt_niche_commodity"
oe_table_niche = "pt_niche"
oe_table_asin = "pt_commodity"
oe_table_keywords = "pt_keywords"
oe_table_trends = "pt_niche_trends"

# 利基数据源补充表
# 关键词表
oe_database_keywords = "oe_us_20231118"
# ASIN表
oe_database_asin = "oe_us_20231125"
# CPC表
oe_database_cpc = "oe_us_20240106"

# 利基历史数据表
asin_h10 = "niche_asin_h10"
niche_asin_original = "niche_asin_original"
niche_price_original = "niche_price_original"
niche_trends_original = "niche_trends_original"
niche_category = "niche_category"

niche = 'niche'
niche_asin = 'niche_asin'
