# 精铺推荐ASIN算法数据路径

# 数据源
pt_product_table = 'pt_product_report'
pt_product_get_cpc = 'pt_product_get_cpc'
pt_product_get_group = 'pt_product_get_group'
pt_product_duplicate = 'pt_product_duplicate'

pt_keywords = 'pt_keywords'
cpc_from_keywords = 'cpc_from_keywords'
pt_relation_traffic = 'pt_relation_traffic'
pt_relevance_asins = 'pt_relevance_asins'
supplement_competitors = 'supplement_competitors'

# pt_keywords_new = 'pt_keywords'
pt_relation_traffic_old = 'pt_relation_traffic_old'
pt_relevance_asins_old = 'pt_relevance_asins_old'

pt_relation_traffic_older = 'pt_relation_traffic_older'
pt_relevance_asins_older = 'pt_relevance_asins_older'

product_famous_brand = 'brand'
product_holiday = 'holiday'
product_category_risk = 'category_risk'
seller_self = 'seller_self'
brand_self = 'brand_self'

# 目标数据库
product_database = 'product_report'

# 目标数据表
# 第一轮
product_table = 'product_report'
product_tag = 'product_tag'
# 第二轮
product_cpc = 'product_cpc'
product_keywords = 'product_keywords'
# 第三轮
product_traffic = 'product_traffic'
product_traffic_tag = 'product_traffic_tag'
product_group = 'product_group'
product_group_tag = 'product_group_tag'

# tag更新表
product_cpc_tag_temporary = 'product_cpc_tag_temporary'
product_traffic_tag_temporary = 'product_traffic_tag_temporary'
# 第一轮
product_table_history = 'product_report_history'
product_tag_history = 'product_tag_history'
# 第二轮
product_cpc_history = 'product_cpc_history'
product_keywords_history = 'product_keywords_history'
# 第三轮
product_group_history = 'product_group_history'
product_group_tag_history = 'product_group_tag_history'
product_traffic_history = 'product_traffic_history'
product_traffic_tag_history = 'product_traffic_tag_history'

# 修复表
product_recommend_modify = 'product_recommend_modify'
product_keyword_temporary = 'product_keyword_temporary'

# ---------------------------------自主提报---------------------------------
# 数据源
# 文件路径，确保是挂载后在本地系统中的路径，例如Windows中可能是 'Z:\\path\\to\\your\\file.xlsx'
pt_product_table_self = r'\\192.168.10.244\数字化选品\精铺线索自主提报\精铺线索自主提报表.xlsx'
sheet_self = 'Sheet1'

pt_product_table_fbm = r'\\192.168.10.244\数字化选品\直发历史开品提报\直发历史开品提报表.xlsx'
sheet_fbm1 = '直发开品'
sheet_fbm2 = '直发线索'

position_table_self = r'\\192.168.10.244\数字化选品\精铺定位相似竞品提报\定位相似竞品提报表.xlsx'
sheet_position = 'report_asin'
pt_clue_asin = 'pt_clue_asin'

pt_shop_table_self = r'\\192.168.10.244\数字化选品\店铺挖掘自主提报\店铺挖掘自主提报表.xlsx'
sheet_shop = 'Sheet1'

# 数据源库
clue_self = 'clue_self'
product_clue_self = 'product_clue_self'
# 目标数据
# 第一轮
product_report_self = 'product_report_self'
product_tag_self = 'product_tag_self'
# 第二轮
product_cpc_self = 'product_cpc_self'
product_keywords_self = 'product_keywords_self'
# 第三轮
product_group_self = 'product_group_self'
product_group_tag_self = 'product_group_tag_self'
product_traffic_self = 'product_traffic_self'
product_traffic_tag_self = 'product_traffic_tag_self'

# 线索检验
# 数据源库
clue_sampling = 'clue_sampling'
# product_clue_sampling = 'product_clue_sampling'
product_clue_sampling = 'product_clue_sampling_fbm'
product_clue_sbi = 'product_clue_sbi'
product_clue_fbm = 'product_clue_fbm'
# 目标数据
# 第一轮
product_report_sampling = 'product_report_sampling'
product_tag_sampling = 'product_tag_sampling'
# 第二轮
product_cpc_sampling = 'product_cpc_sampling'
product_keywords_sampling = 'product_keywords_sampling'
# 第三轮
product_group_sampling = 'product_group_sampling'
product_group_tag_sampling = 'product_group_tag_sampling'
product_traffic_sampling = 'product_traffic_sampling'

# pmi校正
product_group_correction_sampling = 'product_group_correction_sampling'
product_group_correction_pmi = 'product_group_correction_pmi'
product_group_sampling_pmi = 'product_group_sampling_pmi'

product_group_sampling_predict = 'product_group_sampling_predict'

# 定位竞品提报
# 数据库
clue_position = 'clue_position'
# 表
expand_competitors = 'expand_competitors'
competitors_ai = 'competitors_ai'
position_supplement_competitors = 'supplement_competitors'
pt_clue_tag = 'pt_clue_tag'
pt_clue_profit = 'pt_clue_profit'

# 店铺监控
# 数据库
clue_shop_database = 'competitor_shop_monitor'
# 表
pt_task = 'pt_task'
pt_seed_asin = 'pt_seed_asin'
pt_brand_insight = 'pt_brand_insight'
pt_brand_competing_report = 'pt_brand_competing_report'
pt_sellers_insight = 'pt_sellers_insight'
pt_sellers_product = 'pt_sellers_product'

pt_brand_report_tag = 'pt_brand_report_tag'
pt_sellers_tag = 'pt_sellers_tag'
pt_sellers_product_follow = 'pt_sellers_product_follow'

seller_product_follow = 'seller_product_follow'
