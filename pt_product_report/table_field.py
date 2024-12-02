"""
数据库表格所有字段
"""

# ----------------------------数据源库表---------------------------------

# pt_product_report
pt_report_id = 'id'  # 主键
pt_report_asin = 'asin'  # ASIN
pt_report_sku = 'sku'  # SKU
pt_report_brand = 'brand'  # 品牌
pt_report_brand_link = 'brand_link'  # 品牌链接
pt_report_title = 'title'  # 标题
pt_report_image = 'image'  # 主图
pt_report_parent = 'parent'  # 父体
pt_report_category_path = 'category_path'  # 类目路径
pt_report_category = 'category'  # 大类目
pt_report_category_bsr = 'category_bsr'  # 大类BSR
pt_report_category_bsr_increase = 'category_bsr_increase'  # 大类BSR增长数
pt_report_category_bsr_growth = 'category_bsr_growth'  # 大类BSR增长率
pt_report_sub_category = 'sub_category'  # 小类目
pt_report_sub_category_bsr = 'sub_category_bsr'  # 小类BSR
pt_report_sales = 'sales'  # 月销量
pt_report_sales_growth = 'sales_growth'  # 月销量增长率
pt_report_monthly_revenue = 'monthly_revenue'  # 月销售额($)
pt_report_monthly_revenue_increase = 'monthly_revenue_increase'  # 月销售额增长率
pt_report_sub_sales = 'sub_sales'  # 子体销量
pt_report_sub_sales_volume = 'sub_sales_volume'  # 子体销售额($)
pt_report_variations = 'variations'  # 变体数
pt_report_price = 'price'  # 价格($)
pt_report_prime_price = 'prime_price'  # prime价格($)
pt_report_qa = 'qa'  # Q&A
pt_report_ratings = 'ratings'  # 评分数
pt_report_monthly_rating_increase = 'monthly_rating_increase'  # 月新增评分数
pt_report_rating = 'rating'  # 评分
pt_report_reviews_rate = 'reviews_rate'  # 留评率
pt_report_fba_fees = 'fba_fees'  # FBA运费($)
pt_report_gross_margin = 'gross_margin'  # 毛利率
pt_report_date_available = 'date_available'  # 上架时间
pt_report_seller_type = 'seller_type'  # 配送方式
pt_report_lqs = 'lqs'  # LQS
pt_report_sellers = 'sellers'  # 卖家数
pt_report_buybox_seller = 'buybox_seller'  # BuyBox卖家
pt_report_buybox_location = 'buybox_location'  # 卖家所属地
pt_report_seller_info = 'seller_info'  # 卖家信息
pt_report_seller_link = 'seller_link'  # 卖家首页
pt_report_buybox_type = 'buybox_type'  # BuyBox类型
pt_report_best_seller = 'best_seller'  # Best Seller标识
pt_report_ac = 'ac'  # Amazon s Choice
pt_report_new_release = 'new_release'  # New Release标识
pt_report_ebc_available = 'ebc_available'  # A+页面
pt_report_video_available = 'video_available'  # 视频介绍
pt_report_ac_keyword = 'ac_keyword'  # AC关键词
pt_report_weight = 'weight'  # 重量
pt_report_dimensions = 'dimensions'  # 体积
pt_report_sync_time = 'sync_time'  # 最近更新

pt_product_report_field = [
    'id',  # 主键
    'asin',  # ASIN
    'sku',  # SKU
    'brand',  # 品牌
    'brand_link',  # 品牌链接
    'title',  # 标题
    'image',  # 主图
    'parent',  # 父体
    'category_path',  # 类目路径
    'category',  # 大类目
    'category_bsr',  # 大类BSR
    'category_bsr_increase',  # 大类BSR增长数
    'category_bsr_growth',  # 大类BSR增长率
    'sub_category',  # 小类目
    'sub_category_bsr',  # 小类BSR
    'sales',  # 月销量
    'sales_growth',  # 月销量增长率
    'monthly_revenue',  # 月销售额($)
    'monthly_revenue_increase',  # 月销售额增长率
    'sub_sales',  # 子体销量
    'sub_sales_volume',  # 子体销售额($)
    'variations',  # 变体数
    'price',  # 价格($)
    'prime_price',  # prime价格($)
    'qa',  # Q&A
    'ratings',  # 评分数
    'monthly_rating_increase',  # 月新增评分数
    'rating',  # 评分
    'reviews_rate',  # 留评率
    'fba_fees',  # FBA运费($)
    'gross_margin',  # 毛利率
    'date_available',  # 上架时间
    'seller_type',  # 配送方式
    'lqs',  # LQS
    'sellers',  # 卖家数
    'buybox_seller',  # BuyBox卖家
    'buybox_location',  # 卖家所属地
    'seller_info',  # 卖家信息
    'seller_link',  # 卖家首页
    'buybox_type',  # BuyBox类型
    'best_seller',  # Best Seller标识
    'ac',  # Amazon s Choice
    'new_release',  # New Release标识
    'ebc_available',  # A+页面
    'video_available',  # 视频介绍
    'ac_keyword',  # AC关键词
    'weight',  # 重量
    'dimensions',  # 体积
    'sync_time'  # 最近更新
]

# pt_product_get_cpc
pt_get_cpc_id = 'id'  # 主键
pt_get_cpc_country = 'country'  # 站点
pt_get_cpc_asin = 'asin'  # ASIN
pt_get_cpc_price = 'price'  # 价格
pt_get_cpc_recommend = 'recommend'  # 推荐度
pt_get_cpc_recommend_rank = 'recommend_rank'  # 推荐度排名
pt_get_cpc_status = 'status'  # 卖家精灵状态

pt_product_get_cpc_field = [
    'id',  # 主键
    'country',  # 站点
    'asin',  # ASIN
    'price',  # 价格
    'recommend',  # 推荐度
    'recommend_rank',  # 推荐度排名
    'status'  # 卖家精灵状态
]

# pt_keywords
pt_keywords_id = 'id'  # 主键
pt_keywords_asin = 'asin'  # ASIN
pt_keywords_keyword = 'keyword'  # 关键词
pt_keywords_searches = 'searches'  # 月搜索量
pt_keywords_bid = 'bid'  # BID竞价
pt_keywords_purchases = 'purchases'  # 月购买量
pt_keywords_products = 'products'  # 商品数
pt_keywords_supply_demand_ratio = 'supply_demand_ratio'  # 供需比
pt_keywords_ad_products = 'ad_products'  # 广告竞品数
pt_keywords_rank_position_position = 'rank_position_position'  # ASIN自然排名
pt_keywords_rank_position_page = 'rank_position_page'  # ASIN自然排名页
pt_keywords_ad_position_position = 'ad_position_position'  # ASIN广告排名
pt_keywords_ad_position_page = 'ad_position_page'  # ASIN广告排名页
pt_keywords_update_time = 'update_time'  # 最近更新
pt_keywords_crawler_status = 'crawler_status'  # 采集亚马逊CPC状态 0未采集 1采集 -1异常

pt_keywords_field = [
    'id',  # 主键
    'asin',  # ASIN
    'keyword',  # 关键词
    'searches',  # 月搜索量
    'bid',  # BID竞价
    'purchases',  # 月购买量
    'products',  # 商品数
    'supply_demand_ratio',  # 供需比
    'ad_products',  # 广告竞品数
    'rank_position_position',  # ASIN自然排名
    'rank_position_page',  # ASIN自然排名页
    'ad_position_position',  # ASIN广告排名
    'ad_position_page',  # ASIN广告排名页
    'update_time',  # 最近更新
    'crawler_status'  # 采集亚马逊CPC状态 0未采集 1采集 -1异常
]

# cpc_from_keywords
pt_cpc_id = 'id'  # 主键
pt_cpc_parent_id = 'parent_id'  # 关键词id
pt_cpc_keyword = 'keyword'  # 关键词id
pt_cpc_auxiliary_k = 'auxiliary_k'  # 关键词_AMZ
pt_cpc_bid_rangeMedian = 'bid_rangeMedian'  # 推荐竞价
pt_cpc_bid_rangeEnd = 'bid_rangeEnd'  # 竞价上限
pt_cpc_refresh_time_stamp = 'refresh_time_stamp'  # 刷新时间

cpc_from_keywords_field = [
    'id',  # 主键
    'parent_id',  # 关键词id
    'keyword',  # 关键词id
    'auxiliary_k',  # 关键词_AMZ
    'bid_rangeMedian',  # 推荐竞价
    'bid_rangeEnd',  # 竞价上限
    'refresh_time_stamp'  # 刷新时间
]

# pt_product_get_group
pt_get_group_id = 'id'  # 主键
pt_get_group_country = 'country'  # 站点
pt_get_group_asin = 'asin'  # ASIN
pt_get_group_price = 'price'  # 价格
pt_get_group_recommend = 'recommend'  # 推荐度
pt_get_group_blue_ocean_estimate = 'blue_ocean_estimate'  # 蓝海度
pt_get_group_status = 'status'  # 卖家精灵状态

pt_product_get_group_field = [
    'id',  # 主键
    'country',  # 站点
    'asin',  # ASIN
    'price',  # 价格
    'recommend',  # 推荐度
    'blue_ocean_estimate',  # 蓝海度
    'status'  # 卖家精灵状态
]

# pt_relation_traffic
pt_relation_id = 'id'  # 主键
pt_relation_asin = 'asin'  # ASIN
pt_relation_related_asin_num = 'related_asin_num'  # 关联ASIN数
pt_relation_related_asins = 'related_asins'  # 关联ASIN
pt_relation_related_type = 'related_type'  # 关联类型
pt_relation_sku = 'sku'  # SKU
pt_relation_brand = 'brand'  # 品牌
pt_relation_title = 'title'  # 商品标题
pt_relation_product_link = 'product_link'  # 商品详情链接
pt_relation_image = 'image'  # 商品主图
pt_relation_parent = 'parent'  # 父体
pt_relation_category_path = 'category_path'  # 类目路径
pt_relation_category = 'category'  # 大类目
pt_relation_category_bsr = 'category_bsr'  # 大类BSR
pt_relation_category_bsr_increase = 'category_bsr_increase'  # 大类BSR增长数
pt_relation_category_bsr_growth = 'category_bsr_growth'  # 大类BSR增长率
pt_relation_sub_category = 'sub_category'  # 小类目
pt_relation_sub_category_bsr = 'sub_category_bsr'  # 小类BSR
pt_relation_sales = 'sales'  # 月销量
pt_relation_sales_growth = 'sales_growth'  # 月销量增长率
pt_relation_monthly_revenue = 'monthly_revenue'  # 月销售额
pt_relation_monthly_revenue_increase = 'monthly_revenue_increase'  # 月销售额增长率
pt_relation_variation_sold = 'variation_sold'  # 子体销量
pt_relation_variation_revenue = 'variation_revenue'  # 子体销售额($)
pt_relation_price = 'price'  # 价格
pt_relation_qa = 'qa'  # Q&A
pt_relation_gross_margin = 'gross_margin'  # 毛利率
pt_relation_fba_fees = 'fba_fees'  # FBA运费
pt_relation_ratings = 'ratings'  # 评分数
pt_relation_reviews_rate = 'reviews_rate'  # 留评率
pt_relation_rating = 'rating'  # 评分
pt_relation_monthly_rating_increase = 'monthly_rating_increase'  # 月新增评分数
pt_relation_date_available = 'date_available'  # 上架时间
pt_relation_seller_type = 'seller_type'  # 配送方式
pt_relation_lqs = 'lqs'  # LQS
pt_relation_variations = 'variations'  # 变体数
pt_relation_sellers = 'sellers'  # 卖家数
pt_relation_buybox_seller = 'buybox_seller'  # BuyBox卖家
pt_relation_buybox_location = 'buybox_location'  # 卖家所属地
pt_relation_seller_info = 'seller_info'  # 卖家信息
pt_relation_buybox_type = 'buybox_type'  # BuyBox类型
pt_relation_best_seller = 'best_seller'  # Best Seller标识
pt_relation_ac = 'ac'  # Amazon s Choice
pt_relation_new_release = 'new_release'  # New Release标识
pt_relation_ebc_available = 'ebc_available'  # A+页面
pt_relation_video_available = 'video_available'  # 视频介绍
pt_relation_ac_keyword = 'ac_keyword'  # AC关键词
pt_relation_weight = 'weight'  # 重量
pt_relation_dimensions = 'dimensions'  # 体积
pt_relation_update_time = 'update_time'  # 引流时间

pt_relation_traffic_field = [
    'id',  # 主键
    'asin',  # ASIN
    'related_asin_num',  # 关联ASIN数
    'related_asins',  # 关联ASIN
    'related_type',  # 关联类型
    'sku',  # SKU
    'brand',  # 品牌
    'title',  # 商品标题
    'product_link',  # 商品详情链接
    'image',  # 商品主图
    'parent',  # 父体
    'category_path',  # 类目路径
    'category',  # 大类目
    'category_bsr',  # 大类BSR
    'category_bsr_increase',  # 大类BSR增长数
    'category_bsr_growth',  # 大类BSR增长率
    'sub_category',  # 小类目
    'sub_category_bsr',  # 小类BSR
    'sales',  # 月销量
    'sales_growth',  # 月销量增长率
    'monthly_revenue',  # 月销售额
    'monthly_revenue_increase',  # 月销售额增长率
    'variation_sold',  # 子体销量
    'variation_revenue',  # 子体销售额($)
    'price',  # 价格
    'qa',  # Q&A
    'gross_margin',  # 毛利率
    'fba_fees',  # FBA运费
    'ratings',  # 评分数
    'reviews_rate',  # 留评率
    'rating',  # 评分
    'monthly_rating_increase',  # 月新增评分数
    'date_available',  # 上架时间
    'seller_type',  # 配送方式
    'lqs',  # LQS
    'variations',  # 变体数
    'sellers',  # 卖家数
    'buybox_seller',  # BuyBox卖家
    'buybox_location',  # 卖家所属地
    'seller_info',  # 卖家信息
    'buybox_type',  # BuyBox类型
    'best_seller',  # Best Seller标识
    'ac',  # Amazon s Choice
    'new_release',  # New Release标识
    'ebc_available',  # A+页面
    'video_available',  # 视频介绍
    'ac_keyword',  # AC关键词
    'weight',  # 重量
    'dimensions',  # 体积
    'update_time'  # 引流时间
]

# pt_relevance_asins
pt_relevance_id = 'id'  # 主键
pt_relevance_pt_relation_traffic_id = 'pt_relation_traffic_id'  # 关联流量id
pt_relevance_product_pt_report_id = 'product_pt_report_id'  # 关键词反查id
pt_relevance_asin = 'asin'  # 关联ASIN
pt_relevance_category_relevance = 'category_relevance'  # 相关性
pt_relevance_relevance = 'relevance'  # 关联度

pt_relevance_asins_field = [
    'id',  # 主键
    'pt_relation_traffic_id',  # 关联流量id
    'product_pt_report_id',  # 关键词反查id
    'asin',  # 关联ASIN
    'category_relevance',  # 相关性
    'relevance'  # 关联度
]

# --------------------------------目标库表--------------------------------------

# product_report
report_id = 'id'  # 主键
report_asin = 'asin'  # ASIN
report_sku = 'sku'  # SKU
report_brand = 'brand'  # 品牌
report_brand_link = 'brand_link'  # 品牌链接
report_title = 'title'  # 标题
report_image = 'image'  # 主图
report_parent = 'parent'  # 父体
report_category_path = 'category_path'  # 类目路径
report_category = 'category'  # 大类目
report_category_bsr = 'category_bsr'  # 大类BSR
report_category_bsr_growth = 'category_bsr_growth'  # 大类BSR增长率
report_sales = 'sales'  # 月销量
report_monthly_revenue = 'monthly_revenue'  # 月销售额($)
report_price = 'price'  # 价格($)
report_prime_price = 'prime_price'  # prime价格($)
report_qa = 'qa'  # Q&A
report_gross_margin = 'gross_margin'  # 毛利率
report_fba_fees = 'fba_fees'  # FBA运费($)
report_ratings = 'ratings'  # 评分数
report_reviews_rate = 'reviews_rate'  # 留评率
report_rating = 'rating'  # 评分
report_monthly_rating_increase = 'monthly_rating_increase'  # 月新增评分数
report_date_available = 'date_available'  # 上架时间
report_seller_type = 'seller_type'  # 配送方式
report_lqs = 'lqs'  # LQS
report_variations = 'variations'  # 变体数
report_sellers = 'sellers'  # 卖家数
report_buybox_seller = 'buybox_seller'  # BuyBox卖家
report_buybox_location = 'buybox_location'  # 卖家所属地
report_buybox_type = 'buybox_type'  # BuyBox类型
report_best_seller = 'best_seller'  # Best Seller标识
report_ac = 'ac'  # Amazon s Choice
report_new_release = 'new_release'  # New Release标识
report_ebc_available = 'ebc_available'  # A+页面
report_video_available = 'video_available'  # 视频介绍
report_ac_keyword = 'ac_keyword'  # AC关键词
report_weight = 'weight'  # 重量
report_weight_gram = 'weight(g)'  # 重量(g)
report_dimensions = 'dimensions'  # 体积
report_sync_time = 'sync_time'  # 最近更新
report_fbm_possibility = 'fbm_possibility'  # 直发FBM可能性
report_fba_proportion = 'fba_proportion'  # 预估FBA占比
report_first_mail_proportion = 'first_mail_proportion'  # 预估头程占比
report_product_proportion = 'product_proportion'  # 预估货值占比
report_gross_margin_proportion = 'gross_margin_proportion'  # 预估毛利率
report_gross_margin_level = 'gross_margin_level'  # 毛利率级别
report_product_use_level = 'product_use_level'  # 毛估资金利用率
report_month_available = 'month_available'  # 开售月数
report_qa_month_avg = 'qa_month_avg'  # 月均QA数
report_famous_brand_possibility = 'famous_brand_possibility'  # 疑似知名品牌
report_festival_possibility = 'festival_possibility'  # 疑似节日性
report_festival_tag = 'festival_tag'  # 节日名
report_monthly_revenue_level = 'monthly_revenue_level'  # 销额级数
report_high_product_use_level = 'high_product_use_level'  # 高资金利用率
report_high_sales_low_lqs = 'high_sales_low_lqs'  # 高销低LQS
report_long_available_few_qa = 'long_available_few_qa'  # 长期上架少Q&A
report_long_available_no_ebc = 'long_available_no_ebc'  # 长期上架无A+
report_long_available_no_video = 'long_available_no_video'  # 长期上架无视频
report_light_small_fbm = 'light_small_fbm'  # 类轻小直发FBM
report_low_rating_high_sales = 'low_rating_high_sales'  # 差评好卖
report_famous_brand = 'famous_brand'  # 知名品牌
report_monthly_revenue_level_variations_avg = 'monthly_revenue_level_variations_avg'  # 平均变体月销额等级
report_variations_level = 'variations_level'  # 变体等级
report_few_variations = 'few_variations'  # 少变体
report_new_available_high_sales = 'new_available_high_sales'  # 新品爬坡快
report_new_available_high_ratings = 'new_available_high_ratings'  # 新品增评好
report_new_available_nsr = 'new_available_nsr'  # 新品NSR
report_new_available_ac = 'new_available_ac'  # 新品AC标
report_few_ratings_high_sales = 'few_ratings_high_sales'  # 少评好卖
report_customize = 'customize'  # 是否个人定制
report_remake = 'remake'  # 是否翻新
report_recommend = 'recommend'  # 推荐度
report_secondary_category = 'secondary_category'  # 二级类目
report_ignore_category = 'ignore_category'  # 剔除类目
report_update_time = 'update_time'  # 数据更新时间

product_report_field = [
    'id',  # 主键
    'asin',  # ASIN
    'sku',  # SKU
    'brand',  # 品牌
    'brand_link',  # 品牌链接
    'title',  # 标题
    'image',  # 主图
    'parent',  # 父体
    'category_path',  # 类目路径
    'category',  # 大类目
    'category_bsr',  # 大类BSR
    'category_bsr_growth',  # 大类BSR增长率
    'sales',  # 月销量
    'monthly_revenue',  # 月销售额($)
    'price',  # 价格($)
    'prime_price',  # prime价格($)
    'qa',  # Q&A
    'gross_margin',  # 毛利率
    'fba_fees',  # FBA运费($)
    'ratings',  # 评分数
    'reviews_rate',  # 留评率
    'rating',  # 评分
    'monthly_rating_increase',  # 月新增评分数
    'date_available',  # 上架时间
    'seller_type',  # 配送方式
    'lqs',  # LQS
    'variations',  # 变体数
    'sellers',  # 卖家数
    'buybox_seller',  # BuyBox卖家
    'buybox_location',  # 卖家所属地
    'buybox_type',  # BuyBox类型
    'best_seller',  # Best Seller标识
    'ac',  # Amazon s Choice
    'new_release',  # New Release标识
    'ebc_available',  # A+页面
    'video_available',  # 视频介绍
    'ac_keyword',  # AC关键词
    'weight',  # 重量
    'weight(g)',  # 重量(g)
    'dimensions',  # 体积
    'sync_time',  # 最近更新
    'fbm_possibility',  # 直发FBM可能性
    'fba_proportion',  # 预估FBA占比
    'first_mail_proportion',  # 预估头程占比
    'product_proportion',  # 预估货值占比
    'gross_margin_proportion',  # 预估毛利率
    'gross_margin_level',  # 毛利率级别
    'product_use_level',  # 毛估资金利用率
    'month_available',  # 开售月数
    'qa_month_avg',  # 月均QA数
    'famous_brand_possibility',  # 疑似知名品牌
    'festival_possibility',  # 疑似节日性
    'festival_tag',  # 节日名
    'monthly_revenue_level',  # 销额级数
    'high_product_use_level',  # 高资金利用率
    'high_sales_low_lqs',  # 高销低LQS
    'long_available_few_qa',  # 长期上架少Q&A
    'long_available_no_ebc',  # 长期上架无A+
    'long_available_no_video',  # 长期上架无视频
    'light_small_fbm',  # 类轻小直发FBM
    'low_rating_high_sales',  # 差评好卖
    'famous_brand',  # 知名品牌
    'monthly_revenue_level_variations_avg',  # 平均变体月销额等级
    'variations_level',  # 变体等级
    'few_variations',  # 少变体
    'new_available_high_sales',  # 新品爬坡快
    'new_available_high_ratings',  # 新品增评好
    'new_available_nsr',  # 新品NSR
    'new_available_ac',  # 新品AC标
    'few_ratings_high_sales',  # 少评好卖
    'customize',  # 是否个人定制
    'remake',  # 是否翻新
    'recommend',  # 推荐度
    'secondary_category',  # 二级类目
    'ignore_category',  # 剔除类目
    'update_time'  # 数据更新时间
]

# product_cpc
cpc_id = 'id'  # 主键
cpc_asin = 'asin'  # ASIN
cpc_price = 'price'  # 价格
cpc_recommend = 'recommend'  # 推荐度
cpc_keywords = 'keywords'  # 关键词
cpc_cpc_avg_sp = 'cpc_avg_sp'  # 加权SP_CPC
cpc_cpc_avg_amz = 'cpc_avg_amz'  # 加权AMZ_CPC
cpc_cpc_sp_amz_difference = 'cpc_sp_amz_difference'  # SP_AMZ差异度
cpc_cpc_avg = 'cpc_avg'  # 加权CPC
cpc_cr = 'cr'  # 预期CR
cpc_cpc = 'cpc'  # 预期CPC
cpc_net_conversion = 'net_conversion'  # 转化净值
cpc_cpc_factor = 'cpc_factor'  # CPC因子
cpc_blue_ocean_estimate = 'blue_ocean_estimate'  # 市场蓝海度
cpc_update_time = 'update_time'  # 数据更新时间

product_cpc_field = [
    'id',  # 主键
    'asin',  # ASIN
    'price',  # 价格
    'recommend',  # 推荐度
    'keywords',  # 关键词
    'cpc_avg_sp',  # 加权SP_CPC
    'cpc_avg_amz',  # 加权AMZ_CPC
    'cpc_sp_amz_difference',  # SP_AMZ差异度
    'cpc_avg',  # 加权CPC
    'cr',  # 预期CR
    'cpc',  # 预期CPC
    'net_conversion',  # 转化净值
    'cpc_factor',  # CPC因子
    'blue_ocean_estimate',  # 市场蓝海度
    'update_time'  # 数据更新时间
]

# product_keywords
keywords_id = 'id'  # 主键
keywords_asin = 'asin'  # ASIN
keywords_keyword = 'keyword'  # 关键词
keywords_searches = 'searches'  # 月搜索量
keywords_bid = 'bid'  # BID竞价
keywords_purchases = 'purchases'  # 月购买量
keywords_products = 'products'  # 商品数
keywords_supply_demand_ratio = 'supply_demand_ratio'  # 供需比
keywords_ad_products = 'ad_products'  # 广告竞品数
keywords_rank_position_position = 'rank_position_position'  # ASIN自然排名
keywords_rank_position_page = 'rank_position_page'  # ASIN自然排名页
keywords_ad_position_position = 'ad_position_position'  # ASIN广告排名
keywords_ad_position_page = 'ad_position_page'  # ASIN广告排名页
keywords_bid_rangeMedian = 'bid_rangeMedian'  # AMZ_BID推荐
keywords_bid_rangeEnd = 'bid_rangeEnd'  # AMZ_BID上限
keywords_relation = 'relation'  # ASIN_KW相关度
keywords_supply_demand_ratio_avg = 'supply_demand_ratio_avg'  # 供需比均值
keywords_ad_products_avg = 'ad_products_avg'  # 广告竞品数均值
keywords_purchases_rate = 'purchases_rate'  # 月购买转化率
keywords_relation_rank = 'relation_rank'  # 相关度排名
keywords_update_time = 'update_time'  # 数据更新时间

product_keywords_field = [
    'id',  # 主键
    'asin',  # ASIN
    'keyword',  # 关键词
    'searches',  # 月搜索量
    'bid',  # BID竞价
    'purchases',  # 月购买量
    'products',  # 商品数
    'supply_demand_ratio',  # 供需比
    'ad_products',  # 广告竞品数
    'rank_position_position',  # ASIN自然排名
    'rank_position_page',  # ASIN自然排名页
    'ad_position_position',  # ASIN广告排名
    'ad_position_page',  # ASIN广告排名页
    'bid_rangeMedian',  # AMZ_BID推荐
    'bid_rangeEnd',  # AMZ_BID上限
    'relation',  # ASIN_KW相关度
    'supply_demand_ratio_avg',  # 供需比均值
    'ad_products_avg',  # 广告竞品数均值
    'purchases_rate',  # 月购买转化率
    'relation_rank',  # 相关度排名
    'update_time'  # 数据更新时间
]

# product_group
group_id = 'id'  # 主键
group_asin = 'asin'  # 原ASIN
group_price = 'price'  # 价格($)
group_recommend = 'recommend'  # 原ASIN推荐度
group_traffic = 'traffic'  # 相关竞品款数
group_traffic_revenue = 'traffic_revenue'  # 有销额竞品款数
group_traffic_revenue_rate = 'traffic_revenue_rate'  # 有销额竞品款数占比
group_traffic_revenue_recommend = 'traffic_revenue_recommend'  # 有销额推荐达标款数
group_traffic_recommend = 'traffic_recommend'  # 综合竞品推荐度
group_traffic_recommend_rate = 'traffic_recommend_rate'  # 达标推荐度占比
group_monthly_revenue_top5_avg = 'monthly_revenue_top5_avg'  # TOP5月均销额
group_monthly_revenue_top5 = 'monthly_revenue_top5'  # TOP5月销额
group_monthly_revenue_avg = 'monthly_revenue_avg'  # 利基月GMV
group_price_median = 'price_median'  # 价格中位数
group_price_centralization = 'price_centralization'  # 价格集中度
group_gross_margin_proportion = 'gross_margin_proportion'  # 预估平均毛利率
group_product_use_level = 'product_use_level'  # 预估平均资金利用率
group_fba_fees_avg = 'fba_fees_avg'  # 加权FBA运费
group_fbm_proportion = 'fbm_proportion'  # FBM配送占比
group_fbm_product_proportion = 'fbm_product_proportion'  # 直发FBM产品占比
group_fbm_revenue_proportion = 'fbm_revenue_proportion'  # 直发FBM销额占比
group_monthly_revenue_fbm = 'monthly_revenue_fbm'  # 直发FBM月均销额
group_blue_ocean_estimate = 'blue_ocean_estimate'  # 广告蓝海度
group_amz_revenue_proportion = 'amz_revenue_proportion'  # AMZ直营销额占比
group_famous_brand_revenue_proportion = 'famous_brand_revenue_proportion'  # 大牌商标销额占比
group_cn_proportion = 'cn_proportion'  # 中国卖家占比
group_lqs_top5_avg = 'lqs_top5_avg'  # TOP5平均LQS
group_lqs_revenue_pass_avg = 'lqs_revenue_pass_avg'  # 冒出品平均LQS
group_ebc_revenue_pass_proportion = 'ebc_revenue_pass_proportion'  # 冒出品A+占比
group_video_revenue_pass_proportion = 'video_revenue_pass_proportion'  # 冒出品视频占比
group_qa_revenue_pass_proportion = 'qa_revenue_pass_proportion'  # 冒出品QA占比
group_rating_revenue_available_avg = 'rating_revenue_available_avg'  # 动销品平均星级
group_ratings_revenue_pass = 'ratings_revenue_pass'  # 冒出品低星款数
group_ratings_revenue_pass_proportion = 'ratings_revenue_pass_proportion'  # 冒出品低星占比
group_monthly_revenue_top5_proportion = 'monthly_revenue_top5_proportion'  # TOP5销额占比
group_monthly_revenue_untop5_proportion = 'monthly_revenue_untop5_proportion'  # 非TOP5销额占比
group_monthly_revenue_untop5_avg = 'monthly_revenue_untop5_avg'  # 非TOP5月均销额
group_variations_monthly_revenue_median = 'variations_monthly_revenue_median'  # 动销品变体中位数
group_month_available_avg = 'month_available_avg'  # 平均开售月数
group_new_revenue_pass_proportion = 'new_revenue_pass_proportion'  # 冒出品新品占比
group_monthly_revenue_new_avg = 'monthly_revenue_new_avg'  # 新品平均月销额
group_ratings_avg = 'ratings_avg'  # 平均星数
group_ratings_revenue_level_rate = 'ratings_revenue_level_rate'  # 销级星数比
group_new_nsr_proportion = 'new_nsr_proportion'  # 三标新品占比
group_sales_growth_avg = 'sales_growth_avg'  # 月销增长率
group_reviews_rate_avg = 'reviews_rate_avg'  # 加权留评率
group_p_score = 'p_score'  # P得分
group_m_score = 'm_score'  # M得分
group_i_score = 'i_score'  # I得分
group_pmi_score = 'pmi_score'  # PMI得分
group_p_tag = 'p_tag'  # P标签
group_m_tag = 'm_tag'  # M标签
group_i_tag = 'i_tag'  # I标签
group_recommend_level1 = 'recommend_level1'  # 推荐级别v1
group_recommend_level2 = 'recommend_level2'  # 推荐级别v2
group_recommend_vote = 'recommend_vote'  # 综合投票分
group_recommend_vote_level = 'recommend_vote_level'  # 综合推荐级别
group_represent_category = 'represent_category'  # 代表节点
group_represent_score = 'represent_score'  # 代表度
group_duplicate = 'duplicate'  # 重复利基
group_asins = 'asins'  # ASINs
group_update_time = 'update_time'  # 数据更新时间

product_group_field = [
    'id',  # 主键
    'asin',  # 原ASIN
    'price',  # 价格($)
    'recommend',  # 原ASIN推荐度
    'traffic',  # 相关竞品款数
    'traffic_revenue',  # 有销额竞品款数
    'traffic_revenue_rate',  # 有销额竞品款数占比
    'traffic_revenue_recommend',  # 有销额推荐达标款数
    'traffic_recommend',  # 综合竞品推荐度
    'traffic_recommend_rate',  # 达标推荐度占比
    'monthly_revenue_top5_avg',  # TOP5月均销额
    'monthly_revenue_top5',  # TOP5月销额
    'monthly_revenue_avg',  # 利基月GMV
    'price_median',  # 价格中位数
    'price_centralization',  # 价格集中度
    'gross_margin_proportion',  # 预估平均毛利率
    'product_use_level',  # 预估平均资金利用率
    'fba_fees_avg',  # 加权FBA运费
    'fbm_proportion',  # FBM配送占比
    'fbm_product_proportion',  # 直发FBM产品占比
    'fbm_revenue_proportion',  # 直发FBM销额占比
    'monthly_revenue_fbm',  # 直发FBM月均销额
    'blue_ocean_estimate',  # 广告蓝海度
    'amz_revenue_proportion',  # AMZ直营销额占比
    'famous_brand_revenue_proportion',  # 大牌商标销额占比
    'cn_proportion',  # 中国卖家占比
    'lqs_top5_avg',  # TOP5平均LQS
    'lqs_revenue_pass_avg',  # 冒出品平均LQS
    'ebc_revenue_pass_proportion',  # 冒出品A+占比
    'video_revenue_pass_proportion',  # 冒出品视频占比
    'qa_revenue_pass_proportion',  # 冒出品QA占比
    'rating_revenue_available_avg',  # 动销品平均星级
    'ratings_revenue_pass',  # 冒出品低星款数
    'ratings_revenue_pass_proportion',  # 冒出品低星占比
    'monthly_revenue_top5_proportion',  # TOP5销额占比
    'monthly_revenue_untop5_proportion',  # 非TOP5销额占比
    'monthly_revenue_untop5_avg',  # 非TOP5月均销额
    'variations_monthly_revenue_median',  # 动销品变体中位数
    'month_available_avg',  # 平均开售月数
    'new_revenue_pass_proportion',  # 冒出品新品占比
    'monthly_revenue_new_avg',  # 新品平均月销额
    'ratings_avg',  # 平均星数
    'ratings_revenue_level_rate',  # 销级星数比
    'new_nsr_proportion',  # 三标新品占比
    'sales_growth_avg',  # 月销增长率
    'reviews_rate_avg',  # 加权留评率
    'p_score',  # P得分
    'm_score',  # M得分
    'i_score',  # I得分
    'pmi_score',  # PMI得分
    'p_tag',  # P标签
    'm_tag',  # M标签
    'i_tag',  # I标签
    'recommend_level1',  # 推荐级别v1
    'recommend_level2',  # 推荐级别v2
    'recommend_vote',  # 综合投票分
    'recommend_vote_level',  # 综合推荐级别
    'represent_category',  # 代表节点
    'represent_score',  # 代表度
    'duplicate',  # 重复利基
    'asins',  # ASINs
    'update_time'  # 数据更新时间
]
