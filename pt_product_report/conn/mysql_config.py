# 连接mysql用到的配置信息放在这里

# 本地数据库
hostname = "127.0.0.1"
port = 3306  # ssh端口
username = 'root'
password = "Bux04508"
database = "marketing_report"

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
# oe_hostname = "rm-wz9timfv5l31j2mfx0o.rwlb.rds.aliyuncs.com"
oe_hostname = "rm-8vbth3p40ky8y57v8wo.mysql.zhangbei.rds.aliyuncs.com"
oe_port = 3306  # ssh端口
oe_username = 'betterniche'
oe_password = "betterreport168#"

# 利基数据源库
# 数据日期
oe_data_date = "20240622"
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

# sbi数据库
sbi_database = "sbi"
connet_sbi_db_sql = 'mysql+pymysql://' + oe_username + ':' + oe_password + '@' + oe_hostname + '/' + \
                    sbi_database + '?charset=utf8mb4'

# 原精铺算法数据源库
pt_product_database = 'sellersprite_202401'
connet_product_pt_db_sql = 'mysql+pymysql://' + oe_username + ':' + oe_password + '@' + oe_hostname + '/' + \
                           pt_product_database + '?charset=utf8mb4'

# 精铺算法目标数据库
product_database = 'product_report'
connet_product_db_sql = 'mysql+pymysql://' + oe_username + ':' + oe_password + '@' + oe_hostname + '/' + \
                        product_database + '?charset=utf8mb4'

# 精铺算法数据源库
sellersprite_database = 'sellersprite_202502'
sellersprite_database_old = 'sellersprite_202412'
sellersprite_hostname = 'rm-8vbodje181md80v052o.mysql.zhangbei.rds.aliyuncs.com'
sellersprite_port = 3306  # ssh端口
sellersprite_username = 'betterniche'
sellersprite_password = "original123#"
connet_sellersprite_db_sql = 'mysql+pymysql://' + sellersprite_username + ':' + sellersprite_password + '@' + \
                             sellersprite_hostname + '/' + sellersprite_database + '?charset=utf8mb4'

# 线索检验数据源库
clue_sampling_database = 'clue_sampling'
connet_clue_sampling_db_sql = 'mysql+pymysql://' + sellersprite_username + ':' + sellersprite_password + '@' + \
                              sellersprite_hostname + '/' + clue_sampling_database + '?charset=utf8mb4'

# 线索自主提报数据源库
clue_self_database = 'clue_self'
connet_clue_self_db_sql = 'mysql+pymysql://' + sellersprite_username + ':' + sellersprite_password + '@' + \
                          sellersprite_hostname + '/' + clue_self_database + '?charset=utf8mb4'

# amzCPC数据库
sellersprite_cpc = 'sellersprite_cpc'

# 竞品定位竞品提报数据源库
clue_position_database = 'clue_position'
connet_clue_position_db_sql = 'mysql+pymysql://' + sellersprite_username + ':' + sellersprite_password + '@' + \
                              sellersprite_hostname + '/' + clue_position_database + '?charset=utf8mb4'

# 店铺挖掘数据源库
clue_shop_database = 'competitor_shop_monitor'
connet_clue_shop_db_sql = 'mysql+pymysql://' + sellersprite_username + ':' + sellersprite_password + '@' + \
                          sellersprite_hostname + '/' + clue_shop_database + '?charset=utf8mb4'

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

# 利基目标数据表
niche_smile_s = "niche_smile_s"
niche_smile_m = "niche_smile_m"
niche_smile_i = "niche_smile_i"
niche_smile_l = "niche_smile_l"
niche_smile_e = "niche_smile_e"
niche_smile_a = "niche_top_asin"

niche = 'niche'
niche_asin = 'niche_asin'

oe_database_0214 = "oe_20230214"
oe_database_0223 = "oe_20230223"
oe_database_0301 = "oe_20230301"
oe_database_0309 = "oe_20230309"
oe_database_0311 = "oe_20230311"
oe_database_0318 = "oe_20230318"
oe_database_0325 = "oe_20230325"
