# 数据查询
import pt_product_report_path as path
from conn import mysql_config as config

# 辅助表
holiday_sql = 'select 节日关键词 from ' + path.product_database + '.' + path.product_holiday
famous_brand_sql = 'select brand,预估影响力 as "疑似知名品牌" from ' + path.product_database + '.' + path.product_famous_brand
category_risk_sql = 'select category_path,prohibited_risk from ' + path.product_database + '.' + path.product_category_risk

# cpc表
clear_sql_product_cpc = "TRUNCATE TABLE " + path.product_cpc
clear_sql_product_keywords = "TRUNCATE TABLE " + path.product_keywords

create_sql_product_get_group = """
CREATE TABLE `pt_product_get_group` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键',
  `country` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT 'US' COMMENT '站点',
  `asin` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT 'ASIN',
  `price` decimal(10,2) DEFAULT NULL COMMENT '价格',
  `recommend` decimal(10,2) DEFAULT NULL COMMENT '推荐度',
  `blue_ocean_estimate` decimal(10,2) DEFAULT NULL COMMENT '蓝海度',
  `sub_category` varchar(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL,
  `fba_fees` decimal(10,2) DEFAULT NULL,
  `status` int DEFAULT '0' COMMENT '卖家精灵状态',
  PRIMARY KEY (`id`) USING BTREE,
  KEY `index` (`asin`),
  KEY `price_index` (`price`),
  KEY `recommend_index` (`recommend`),
  KEY `blue_ocean_estimate_index` (`blue_ocean_estimate`),
  KEY `sub_category` (`sub_category`)
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci ROW_FORMAT=DYNAMIC COMMENT='当月需要爬取竞品数据的线索';
"""

clear_sql_product_cpc_tag_temporary = "TRUNCATE TABLE " + path.product_cpc_tag_temporary

update_sql_product_tag = "UPDATE product_tag_history INNER JOIN product_cpc_tag_temporary " \
                         "ON product_tag_history.data_id=product_cpc_tag_temporary.data_id " \
                         "SET product_tag_history.cpc_status=1," \
                         "product_tag_history.`SP_AMZ差异度分布`=product_cpc_tag_temporary.`SP_AMZ差异度分布`," \
                         "product_tag_history.`加权CPC分布`=product_cpc_tag_temporary.`加权CPC分布`," \
                         "product_tag_history.`市场蓝海度分布`=product_cpc_tag_temporary.`市场蓝海度分布`"

update_sql_product_keyword = "UPDATE product_cpc_history INNER JOIN product_keyword_temporary " \
                             "ON product_cpc_history.ASIN=product_keyword_temporary.ASIN " \
                             "SET product_cpc_history.`关键词`=product_keyword_temporary.`关键词`"

clear_sql_product_table = "TRUNCATE TABLE " + path.product_table
clear_sql_product_tag = "TRUNCATE TABLE " + path.product_tag

create_sql_product_get_cpc = "CREATE TABLE `pt_product_get_cpc` (" \
                             "`id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键'," \
                             "`country` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT 'US' COMMENT '站点'," \
                             "`asin` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT 'ASIN'," \
                             "`price` decimal(10,2) DEFAULT NULL," \
                             "`recommend` decimal(10,2) DEFAULT NULL," \
                             "`status` int DEFAULT '0' COMMENT '卖家精灵状态'," \
                             "PRIMARY KEY (`id`) USING BTREE," \
                             "KEY `index` (`asin`)," \
                             "KEY `price_index` (`price`)," \
                             "KEY `recommend_index` (`recommend`)" \
                             ") ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci " \
                             "ROW_FORMAT=DYNAMIC COMMENT='当月需要爬取CPC数据的线索'"

duplicate_sql_product_report = "DELETE FROM pt_product_report WHERE id IN(SELECT b.id FROM (SELECT * FROM " \
                               "(SELECT *,ROW_NUMBER() over (PARTITION by asin ORDER BY id) asin_rank " \
                               "FROM pt_product_report) a WHERE a.asin_rank>1) b)"

update_sql_product_get_cpc = "UPDATE pt_product_get_cpc INNER JOIN(SELECT a.asin,@rank:=@rank+1 as recommend_rank FROM " \
                             "(SELECT * FROM pt_product_get_cpc ORDER BY recommend DESC) a, (SELECT @rank:=0) b) c " \
                             "on pt_product_get_cpc.asin=c.asin SET pt_product_get_cpc.recommend_rank=c.recommend_rank"

clear_sql_product_recommend = "TRUNCATE TABLE " + path.product_recommend_modify

update_sql_product_recommend = "UPDATE product_report_history INNER JOIN product_recommend_modify ON " \
                               "product_report_history.ASIN=product_recommend_modify.ASIN AND " \
                               "product_report_history.`数据更新时间`=product_recommend_modify.`数据更新时间` " \
                               "SET product_report_history.`销额级数`=product_recommend_modify.`销额级数`," \
                               "product_report_history.`推荐度`=product_recommend_modify.`推荐度`"

update_sql_sub_category = "UPDATE pt_product_get_group INNER JOIN pt_product_report ON pt_product_get_group.asin=" \
                          "pt_product_report.asin SET pt_product_get_group.sub_category=pt_product_report.sub_category," \
                          "pt_product_get_group.fba_fees=pt_product_report.fba_fees"

sql_get_group = 'select * from ' + path.pt_product_get_group

# group表

# 索引创建
create_index_sql_relevance_1 = 'CREATE INDEX idx_asin ON pt_relevance_asins(asin);'
create_index_sql_relevance_2 = 'CREATE INDEX idx_relation_traffic_id ON pt_relevance_asins(relation_traffic_id);'
create_index_sql_supplement = 'CREATE INDEX idx_clue_asin ON supplement_competitors(clue_asin);'

duplicate_sql_supplement = 'DELETE FROM supplement_competitors WHERE id IN(SELECT id FROM(SELECT *,ROW_NUMBER() over ' \
                           '(PARTITION by ASIN,clue_asin ORDER BY id DESC) "rank" FROM supplement_competitors) a WHERE a.rank>1)'

clear_sql_product_traffic = "TRUNCATE TABLE " + path.product_traffic
clear_sql_product_traffic_tag = "TRUNCATE TABLE " + path.product_traffic_tag
clear_sql_product_group = "TRUNCATE TABLE " + path.product_group
clear_sql_product_group_tag = "TRUNCATE TABLE " + path.product_group_tag

update_sql_product_group_tag = "UPDATE product_tag_history INNER JOIN product_group_tag_history ON " \
                               "product_tag_history.data_id=product_group_tag_history.data_id " \
                               "SET product_tag_history.traffic_status=1"

# pt_keywords创建
create_sql_product_pt_keywords = """
CREATE TABLE `pt_keywords` (
`id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键',
`asin` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT 'ASIN',
`keyword` varchar(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '关键词',
`search_frequency` bigint DEFAULT NULL COMMENT '搜索量',
`update_time` date NOT NULL COMMENT '最近更新',
`crawler_status` int DEFAULT '0' COMMENT '采集亚马逊CPC状态 0未采集 1采集 -1异常',
`relevance` decimal(10,1) DEFAULT '0.0' COMMENT '关联度',
PRIMARY KEY (`id`) USING BTREE,
KEY `pi_asin` (`asin`) USING BTREE,
KEY `keyword` (`keyword`)
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci ROW_FORMAT=DYNAMIC COMMENT='关键词临时表';
"""

# cpc_from_keywords创建
create_sql_cpc_from_keywords = """
CREATE TABLE `cpc_from_keywords` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `keyword` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NOT NULL,
  `auxiliary_k` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NOT NULL,
  `bid_rangeMedian` decimal(10,2) DEFAULT NULL,
  `bid_rangeEnd` decimal(10,2) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `keyword` (`keyword`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
"""

insert_sql_cpc_from_keywords = """
INSERT INTO cpc_from_keywords ( `id`, `keyword`, `auxiliary_k`, `bid_rangeMedian`, `bid_rangeEnd` ) (
    SELECT DISTINCT
		`sellersprite_cpc`.`cpc_from_keywords`.`id` AS `id`,
		`sellersprite_cpc`.`cpc_from_keywords`.`keyword` AS `keyword`,
		`sellersprite_cpc`.`cpc_from_keywords`.`auxiliary_k` AS `auxiliary_k`,
		`sellersprite_cpc`.`cpc_from_keywords`.`bid_rangeMedian` AS `bid_rangeMedian`,
		`sellersprite_cpc`.`cpc_from_keywords`.`bid_rangeEnd` AS `bid_rangeEnd` 
	FROM
		(
			`pt_keywords`
			LEFT JOIN `sellersprite_cpc`.`cpc_from_keywords` ON ((
					`pt_keywords`.`keyword` = `sellersprite_cpc`.`cpc_from_keywords`.`keyword` 
				))) 
	WHERE
		`sellersprite_cpc`.`cpc_from_keywords`.`id` > 0 
		AND ((
				`pt_keywords`.`crawler_status` = 1 
				) 
			OR (
				`pt_keywords`.`crawler_status` = -(
					2 
				))) 
ORDER BY
	`sellersprite_cpc`.`cpc_from_keywords`.`id`)
"""

#
create_sql_pt_product_duplicate = """
CREATE TABLE `pt_product_duplicate` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键',
  `asin` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT 'ASIN',
  `price_list` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '价格划分区间',
  `price_tag_rank` int DEFAULT '1' COMMENT '价格区间排名',
  `fba_fees_list` varchar(512) COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT 'FBA费用划分区间',
  `fba_fees_tag_rank` int DEFAULT '1' COMMENT 'FBA费用区间排名',
  `rank` int DEFAULT '1' COMMENT '价格区间内排名',
  `duplicate_tag` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT '1' COMMENT '重复度',
  `duplicate_type` int DEFAULT NULL COMMENT '重复类型',
  PRIMARY KEY (`id`) USING BTREE,
  KEY `index` (`asin`),
  KEY `duplicate_tag` (`duplicate_tag`)
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci ROW_FORMAT=DYNAMIC COMMENT='类目节点重复度计算表';
"""

sql_clue_self = """
SELECT DISTINCT
	pt_product_report.price,
	pt_product_report.fba_fees,
	pt_product_report.sub_category,(
	CASE
			pt_clue_asin.data_tag 
			WHEN "历史开售" THEN
			3 
			WHEN "直发开品" THEN
			4 
			WHEN "直发线索" THEN
			5 ELSE 0 
		END 
		) "duplicate_type" 
	FROM
		pt_clue_asin
		LEFT JOIN pt_product_report ON pt_clue_asin.asin = pt_product_report.asin 
	WHERE
		pt_clue_asin.data_tag <> "自主提报" 
	AND price > 0 
	AND sub_category IS NOT NULL
"""
sql_group = "SELECT * FROM pt_product_get_group"

# sql_group = "SELECT id,asin,price,recommend,blue_ocean_estimate,sub_category,fba_fees FROM pt_product_get_group limit 5000"

# 将上个月数据写入

# 创建表
create_sql_pt_relevance_asins_old = """
CREATE TABLE `pt_relevance_asins_old` (
  `id` bigint NOT NULL DEFAULT '1' COMMENT '主键',
  `relation_traffic_id` bigint NOT NULL COMMENT '关联流量id',
  `product_report_id` bigint NOT NULL COMMENT '关键词反查id',
  `asin` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '关联ASIN',
  `category_relevance` int DEFAULT NULL COMMENT '相关性',
  `relevance` double DEFAULT NULL COMMENT '关联度',
  `data_sources_type` int DEFAULT '0' COMMENT '0: 关联流量 1:竞品数据',
  PRIMARY KEY (`id`) USING BTREE,
  KEY `idx_asin` (`asin`),
  KEY `idx_relation_traffic_id` (`relation_traffic_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci ROW_FORMAT=DYNAMIC COMMENT='关联度计算中间表_上个月';
"""

create_sql_pt_relation_traffic_old = """
CREATE TABLE `pt_relation_traffic_old` (
  `traffic_id` bigint NOT NULL AUTO_INCREMENT,
  `id` bigint NOT NULL DEFAULT '1' COMMENT '主键',
  `asin` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NOT NULL COMMENT 'ASIN',
  `related_asin_num` int DEFAULT NULL COMMENT '关联ASIN数',
  `related_asins` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '关联ASIN',
  `related_type` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '关联类型',
  `sku` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT 'SKU',
  `brand` varchar(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '品牌',
  `title` varchar(2048) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '商品标题',
  `product_link` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '商品详情链接',
  `image` varchar(2048) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '商品主图',
  `parent` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '父体',
  `category_path` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '类目路径',
  `category` varchar(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '大类目',
  `category_bsr` bigint DEFAULT NULL COMMENT '大类BSR',
  `category_bsr_increase` int DEFAULT NULL COMMENT '大类BSR增长数',
  `category_bsr_growth` double DEFAULT NULL COMMENT '大类BSR增长率',
  `sub_category` varchar(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '小类目',
  `sub_category_bsr` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '小类BSR',
  `sales` int DEFAULT NULL COMMENT '月销量',
  `sales_growth` double DEFAULT NULL COMMENT '月销量增长率',
  `monthly_revenue` int DEFAULT NULL COMMENT '月销售额',
  `monthly_revenue_increase` varchar(6) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '月销售额增长率',
  `variation_sold` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '子体销量',
  `variation_revenue` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '子体销售额($)',
  `price` double DEFAULT NULL COMMENT '价格',
  `qa` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT 'Q&A',
  `gross_margin` double DEFAULT NULL COMMENT '毛利率',
  `fba_fees` double DEFAULT NULL COMMENT 'FBA运费',
  `ratings` int DEFAULT NULL COMMENT '评分数',
  `reviews_rate` double DEFAULT NULL COMMENT '留评率',
  `rating` double DEFAULT NULL COMMENT '评分',
  `monthly_rating_increase` int DEFAULT NULL COMMENT '月新增评分数',
  `date_available` date DEFAULT NULL COMMENT '上架时间',
  `seller_type` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '配送方式',
  `lqs` int DEFAULT NULL COMMENT 'LQS',
  `variations` int DEFAULT NULL COMMENT '变体数',
  `sellers` int DEFAULT NULL COMMENT '卖家数',
  `buybox_seller` varchar(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT 'BuyBox卖家',
  `buybox_location` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '卖家所属地',
  `seller_info` text CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci COMMENT '卖家信息',
  `buybox_type` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT 'BuyBox类型',
  `best_seller` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT 'Best Seller标识',
  `ac` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT 'Amazon s Choice',
  `new_release` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT 'New Release标识',
  `ebc_available` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT 'A+页面',
  `video_available` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '视频介绍',
  `ac_keyword` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT 'AC关键词',
  `weight` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '重量',
  `dimensions` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '体积',
  `update_time` date DEFAULT NULL COMMENT '引流时间',
  PRIMARY KEY (`traffic_id`) USING BTREE,
  KEY `id` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci ROW_FORMAT=DYNAMIC COMMENT='卖家精灵关联流量_上个月';
"""

# 写入表
insert_sql_pt_relevance_asins_old = \
    "INSERT INTO sellersprite_202411.`pt_relevance_asins_old` " + \
    "( `relation_traffic_id`, `product_report_id`, `asin`, `category_relevance`, `relevance` ) (" + \
    "SELECT distinct " \
    "pt_relevance_asins.`relation_traffic_id`," \
    "pt_relevance_asins.`product_report_id`," \
    "pt_relevance_asins.`asin`," \
    "pt_relevance_asins.`category_relevance`," \
    "pt_relevance_asins.`relevance` " \
    "FROM " \
    + config.sellersprite_database + ".pt_product_get_group INNER JOIN " \
    + config.sellersprite_database_old + ".pt_relevance_asins ON " \
    + config.sellersprite_database + ".pt_product_get_group.asin = " \
    + config.sellersprite_database_old + ".pt_relevance_asins.asin);"

insert_sql_pt_relation_traffic_old = \
    'INSERT INTO `pt_relation_traffic_old` (' + \
    '`id`,' + \
    '`asin`,' + \
    '`related_asin_num`,' + \
    '`related_asins`,' + \
    '`related_type`,' + \
    '`sku`,' + \
    '`brand`,' + \
    '`title`,' + \
    '`product_link`,' + \
    '`image`,' + \
    '`parent`,' + \
    '`category_path`,' + \
    '`category`,' + \
    '`category_bsr`,' + \
    '`category_bsr_increase`,' + \
    '`category_bsr_growth`,' + \
    '`sub_category`,' + \
    '`sub_category_bsr`,' + \
    '`sales`,' + \
    '`sales_growth`,' + \
    '`monthly_revenue`,' + \
    '`monthly_revenue_increase`,' + \
    '`variation_sold`,' + \
    '`variation_revenue`,' + \
    '`price`,' + \
    '`qa`,' + \
    '`gross_margin`,' + \
    '`fba_fees`,' + \
    '`ratings`,' + \
    '`reviews_rate`,' + \
    '`rating`,' + \
    '`monthly_rating_increase`,' + \
    '`date_available`,' + \
    '`seller_type`,' + \
    '`lqs`,' + \
    '`variations`,' + \
    '`sellers`,' + \
    '`buybox_seller`,' + \
    '`buybox_location`,' + \
    '`seller_info`,' + \
    '`buybox_type`,' + \
    '`best_seller`,' + \
    '`ac`,' + \
    '`new_release`,' + \
    '`ebc_available`,' + \
    '`video_available`,' + \
    '`ac_keyword`,' + \
    '`weight`,' + \
    '`dimensions`,' + \
    '`update_time`' + \
    ')(SELECT DISTINCT ' + config.sellersprite_database_old + '.pt_relation_traffic.* FROM ' + \
    config.sellersprite_database + '.pt_product_get_group INNER JOIN ' + \
    config.sellersprite_database_old + '.pt_relevance_asins ON ' + \
    config.sellersprite_database + '.pt_product_get_group.asin=' + \
    config.sellersprite_database_old + '.pt_relevance_asins.asin INNER JOIN ' + \
    config.sellersprite_database_old + '.pt_relation_traffic ON ' + \
    config.sellersprite_database_old + '.pt_relevance_asins.relation_traffic_id=' + \
    config.sellersprite_database_old + '.pt_relation_traffic.id);'

# 自主提报表
create_sql_group_self = "CREATE TABLE `pt_clue_asin` (" \
                        "`id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键'," \
                        "`country` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT 'US' COMMENT '站点'," \
                        "`asin` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT 'ASIN'," \
                        "`name` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '提报人'," \
                        "`status` int DEFAULT '0' COMMENT '卖家精灵状态'," \
                        "`update_time` date DEFAULT NULL COMMENT '数据更新时间'," \
                        "PRIMARY KEY (`id`) USING BTREE," \
                        "KEY `index` (`asin`) USING BTREE," \
                        "KEY `name` (`name`) USING BTREE" \
                        ") ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci " \
                        "ROW_FORMAT=DYNAMIC COMMENT='当月自主提报需要爬取竞品数据的线索';"

# 自主提报查重
sql_sellersprite_clue_self = 'select `asin`,`data_tag` from ' + path.pt_clue_asin

# 竞品提报查重
sql_sellersprite_clue_position = 'select `asin` from ' + path.pt_clue_asin

# 店铺挖掘提报查重
sql_sellersprite_clue_shop = 'select `asin` from ' + path.pt_seed_asin

# 类目去重
# 更新日期
update_date = str(config.sellersprite_database)[-6:-2] + "-" + str(config.sellersprite_database)[-2:] + "-01"

duplicate_sql1 = 'UPDATE product_group_history SET `重复利基`=0 WHERE `数据更新时间`="' + str(update_date) + \
                 '" AND `代表节点` IN (SELECT `代表节点` FROM(SELECT `代表节点`,count(`原ASIN`) "利基数量" FROM ' \
                 'product_group_history WHERE `数据更新时间`="' + str(update_date) + '" GROUP BY `代表节点`) ' + \
                 'group_category WHERE `利基数量`<10)'

pmi_rank_sql = 'UPDATE product_group_history INNER JOIN (SELECT `原ASIN`,ROW_NUMBER() OVER (PARTITION BY `代表节点` ' \
               'ORDER BY `PMI得分` DESC) "pmi_rank" FROM product_group_history WHERE `数据更新时间`="' + str(update_date) + \
               '") group_rank ON product_group_history.`原ASIN`=group_rank.`原ASIN` SET product_group_history.pmi_rank=' \
               'group_rank.pmi_rank WHERE product_group_history.`数据更新时间`="' + str(update_date) + '"'

duplicate_sql2 = 'UPDATE product_group_history SET `重复利基`=0 WHERE `数据更新时间`="' + str(update_date) + \
                 '" AND id IN(SELECT a.id FROM(SELECT id FROM product_group_history LEFT JOIN (SELECT `代表节点`,' \
                 'MAX(pmi_rank) pmi_max,if(round(MAX(pmi_rank)*0.05)>10,round(MAX(pmi_rank)*0.05),10) pmi_count FROM ' \
                 'product_group_history WHERE `数据更新时间`="' + str(update_date) + '" GROUP BY `代表节点`) group_count' + \
                 ' ON product_group_history.`代表节点`=group_count.`代表节点` WHERE product_group_history.`数据更新时间`="' + \
                 str(update_date) + '" AND product_group_history.`重复利基`<2 AND product_group_history.pmi_rank<=group_count.pmi_count) a);'

duplicate_sql2_old = 'UPDATE product_group_history SET `重复利基`=0 WHERE `数据更新时间`="' + str(update_date) + \
                     '" AND `原ASIN` IN(SELECT `原ASIN` FROM (SELECT `原ASIN`,' \
                     'ROW_NUMBER() OVER (PARTITION BY `代表节点` ORDER BY `PMI得分` DESC) "pmi_rank" FROM ' \
                     'product_group_history WHERE `数据更新时间`="' + str(update_date) + \
                     '" AND `代表度`>=0.4 AND `有销额竞品款数`>=10) group_duplicate WHERE pmi_rank<=5)'

# 父体去重
duplicate_sql3 = 'UPDATE product_group_history SET `重复利基`=2 WHERE `数据更新时间`="' + str(update_date) + \
                 '" AND `重复利基`=0 AND `原ASIN` IN(SELECT `原ASIN` FROM (SELECT product_group_history.`原ASIN`,' \
                 'ROW_NUMBER() OVER (PARTITION BY product_report_history.`父体`,product_group_history.`代表节点` ' \
                 'ORDER BY `PMI得分` DESC) "pmi_rank" FROM product_group_history LEFT JOIN product_report_history ' \
                 'ON product_group_history.`原ASIN`=product_report_history.ASIN AND product_group_history.`数据更新时间` ' \
                 '=product_report_history.`数据更新时间` WHERE product_group_history.`数据更新时间`="' + str(update_date) + \
                 '" AND product_group_history.`有销额竞品款数`>=10) group_duplicate WHERE pmi_rank>1)'

# 历史开售产品去重
duplicate_sql4 = 'UPDATE product_group_history SET `重复利基`=3 WHERE `数据更新时间`="' + str(update_date) + \
                 '" AND `重复利基`=0 AND `代表节点` IN(SELECT DISTINCT`代表节点` FROM (SELECT product_clue_sbi.`ASIN`,' \
                 'product_group_self.`代表节点` FROM product_clue_sbi LEFT JOIN product_group_self ON ' \
                 'product_clue_sbi.asin=product_group_self.`原ASIN` WHERE product_group_self.`代表节点` IS NOT NULL) gd)'

duplicate_sql5 = 'UPDATE product_group_history SET `重复利基`=3 WHERE `数据更新时间`="' + str(update_date) + \
                 '" AND `重复利基`=0 AND `代表节点` IN(SELECT DISTINCT `小类目` FROM (SELECT product_clue_sbi.`ASIN`,' \
                 'product_report_self.`小类目` FROM product_clue_sbi LEFT JOIN product_report_self ON ' \
                 'product_clue_sbi.asin=product_report_self.ASIN WHERE product_report_self.`小类目` IS NOT NULL) gd)'

duplicate_sql6 = 'UPDATE product_group_history SET `重复利基`=3 WHERE `数据更新时间`="' + str(update_date) + \
                 '" AND `重复利基`=0 AND `代表节点` IN(SELECT DISTINCT `小类目` FROM (SELECT product_clue_sampling.`ASIN`,' \
                 'product_report_self.`小类目` FROM product_clue_sampling LEFT JOIN product_report_self ON ' \
                 'product_clue_sampling.clue_asin=product_report_self.ASIN WHERE product_report_self.`小类目` IS NOT NULL) gd)'

# 直发线索去重
duplicate_sql7 = 'UPDATE product_group_history SET `重复利基`=4 WHERE `数据更新时间`="' + str(update_date) + \
                 '" AND `重复利基`=0 AND `代表节点` IN(SELECT DISTINCT `代表节点` FROM (SELECT product_clue_fbm.`ASIN`,' \
                 'product_group_self.`代表节点` FROM product_clue_fbm LEFT JOIN product_group_self ON ' \
                 'product_clue_fbm.asin=product_group_self.`原ASIN` WHERE product_clue_fbm.data_tag="直发开品" AND ' \
                 'product_group_self.`代表节点` IS NOT NULL) group_duplicate);'

duplicate_sql8 = 'UPDATE product_group_history SET `重复利基`=4 WHERE `数据更新时间`="' + str(update_date) + \
                 '" AND `重复利基`=0 AND `代表节点` IN(SELECT DISTINCT `小类目` FROM (SELECT product_clue_fbm.`ASIN`,' \
                 'product_report_self.`小类目` FROM product_clue_fbm LEFT JOIN product_report_self ON ' \
                 'product_clue_fbm.asin=product_report_self.ASIN WHERE product_clue_fbm.data_tag="直发开品" AND ' \
                 'product_report_self.`小类目` IS NOT NULL) group_duplicate);'

duplicate_sql9 = 'UPDATE product_group_history SET `重复利基`=5 WHERE `数据更新时间`="' + str(update_date) + \
                 '" AND `重复利基`=0 AND `代表节点` IN(SELECT DISTINCT `代表节点` FROM (SELECT product_clue_fbm.`ASIN`,' \
                 'product_group_self.`代表节点` FROM product_clue_fbm LEFT JOIN product_group_self ON ' \
                 'product_clue_fbm.asin=product_group_self.`原ASIN` WHERE product_clue_fbm.data_tag="直发线索" AND ' \
                 'product_group_self.`代表节点` IS NOT NULL) group_duplicate);'

duplicate_sql10 = 'UPDATE product_group_history SET `重复利基`=5 WHERE `数据更新时间`="' + str(update_date) + \
                  '" AND `重复利基`=0 AND `代表节点` IN(SELECT DISTINCT `小类目` FROM (SELECT product_clue_fbm.`ASIN`,' \
                  'product_report_self.`小类目` FROM product_clue_fbm LEFT JOIN product_report_self ON ' \
                  'product_clue_fbm.asin=product_report_self.ASIN WHERE product_clue_fbm.data_tag="直发线索" AND ' \
                  'product_report_self.`小类目` IS NOT NULL) group_duplicate);'

# clue_self表
clue_sql = 'SELECT pt_product_report.*,SUBSTRING_INDEX(pt_product_report.category_path,":",2) as "二级类目",' \
           'pt_clue_asin.update_time "数据更新时间" FROM pt_clue_asin LEFT JOIN pt_product_report ON ' \
           'pt_clue_asin.asin=pt_product_report.asin WHERE pt_product_report.id>0 and pt_clue_asin.`status`=1'

update_clue_sql = 'UPDATE pt_clue_asin SET `status`=2 WHERE asin IN(SELECT asin FROM pt_product_get_cpc)'

update_sql_group_self_tag = "UPDATE product_tag_self INNER JOIN product_group_tag_self ON " \
                            "product_tag_self.data_id=product_group_tag_self.data_id " \
                            "SET product_tag_self.traffic_status=1"

update_sql_product_tag_self = "UPDATE product_tag_self INNER JOIN product_cpc_tag_temporary " \
                              "ON product_tag_self.data_id=product_cpc_tag_temporary.data_id " \
                              "SET product_tag_self.cpc_status=1," \
                              "product_tag_self.`SP_AMZ差异度分布`=product_cpc_tag_temporary.`SP_AMZ差异度分布`," \
                              "product_tag_self.`加权CPC分布`=product_cpc_tag_temporary.`加权CPC分布`," \
                              "product_tag_self.`市场蓝海度分布`=product_cpc_tag_temporary.`市场蓝海度分布`"

update_sql_group_sampling_tag = "UPDATE product_tag_sampling INNER JOIN product_group_tag_sampling ON " \
                                "product_tag_sampling.data_id=product_group_tag_sampling.data_id " \
                                "SET product_tag_sampling.traffic_status=1"

update_sql_product_tag_sampling = "UPDATE product_tag_sampling INNER JOIN product_cpc_tag_temporary " \
                                  "ON product_tag_sampling.data_id=product_cpc_tag_temporary.data_id " \
                                  "SET product_tag_sampling.cpc_status=1," \
                                  "product_tag_sampling.`SP_AMZ差异度分布`=product_cpc_tag_temporary.`SP_AMZ差异度分布`," \
                                  "product_tag_sampling.`加权CPC分布`=product_cpc_tag_temporary.`加权CPC分布`," \
                                  "product_tag_sampling.`市场蓝海度分布`=product_cpc_tag_temporary.`市场蓝海度分布`"

update_clue_cpc_sql = 'UPDATE pt_product_get_cpc SET `status`=2 WHERE asin IN(SELECT asin FROM pt_product_get_group)'

sql_asin = 'select * from ' + path.pt_product_get_cpc + ' where status=1'
# sql_asin = 'select * from ' + path.pt_product_get_cpc
# sql_asin = 'select * from ' + path.pt_product_get_cpc + ' where update_time="2024-08-16"'
sql_kw = 'select ' + path.pt_keywords + '.*,pt_product.price,pt_product.recommend,pt_product.update_time as ' + \
         '"数据更新时间" from (' + sql_asin + ') pt_product left join ' + path.pt_keywords + ' on pt_product.asin = ' + \
         path.pt_keywords + '.asin'
sql_cpc = 'select DISTINCT ' + path.cpc_from_keywords + '.* from (' + sql_kw + ') pt_kw left join ' + \
          path.cpc_from_keywords + ' on pt_kw.keyword = ' + path.cpc_from_keywords + '.keyword where ' + \
          '(cpc_from_keywords.bid_rangeMedian + cpc_from_keywords.bid_rangeEnd)>0'

# sql_cpc ='select DISTINCT cpc_from_keywords.* from (select * from pt_product_get_cpc where `status`=1) pt_kw
# left join cpc_from_keywords on pt_kw.keyword = cpc_from_keywords.keyword where
# (cpc_from_keywords.bid_rangeMedian + cpc_from_keywords.bid_rangeEnd)>0'

update_clue_group_sql = 'UPDATE pt_product_get_group SET `status`=2 WHERE asin IN(SELECT DISTINCT asin FROM pt_relevance_asins)'

sql_asin_self = 'select asin as "related_asin",update_time as "数据更新时间" from ' + path.pt_product_get_group + ' where status=1'

sql_relevance_self = 'SELECT ' + path.pt_relevance_asins + '.* FROM (' + sql_asin_self + ') pt_product LEFT JOIN ' + \
                     path.pt_relevance_asins + ' ON pt_product.related_asin = ' + path.pt_relevance_asins + '.asin'

sql_traffic_self = 'SELECT pt_relevance.asin as "related_asin",pt_relevance.relevance,pt_relevance.category_relevance,' \
                   + path.pt_relation_traffic + '.*,SUBSTRING_INDEX(' + path.pt_relation_traffic + \
                   '.category_path,":",2) as "二级类目" FROM( ' + sql_relevance_self + ' ) pt_relevance LEFT JOIN ' + \
                   path.pt_relation_traffic + ' ON pt_relevance.relation_traffic_id = ' + path.pt_relation_traffic + \
                   '.id WHERE pt_relevance.asin IS NOT NULL'

sql_asin_sampling = 'select asin as "related_asin",update_time as "数据更新时间" from ' + path.pt_clue_asin + \
                    ' where clue_tag<>"捡漏"'

sql_relevance_sampling = 'SELECT ' + path.pt_relevance_asins + '.* FROM (' + sql_asin_sampling + ') pt_product LEFT JOIN ' + \
                         path.pt_relevance_asins + ' ON pt_product.related_asin = ' + path.pt_relevance_asins + '.asin'

sql_traffic_sampling = 'SELECT pt_relevance.asin as "related_asin",pt_relevance.relevance,pt_relevance.category_relevance,' \
                       + path.pt_relation_traffic + '.*,SUBSTRING_INDEX(' + path.pt_relation_traffic + \
                       '.category_path,":",2) as "二级类目" FROM( ' + sql_relevance_sampling + ' ) pt_relevance LEFT JOIN ' + \
                       path.pt_relation_traffic + ' ON pt_relevance.relation_traffic_id = ' + path.pt_relation_traffic + \
                       '.id WHERE pt_relevance.asin IS NOT NULL'

sql_asin_group_sampling = 'SELECT product_group_self.*,product_clue_sampling.clue_tag,product_clue_sampling.update_time ' \
                          'FROM product_clue_sampling LEFT JOIN product_group_self ON ' \
                          'product_clue_sampling.clue_asin=product_group_self.`原ASIN` ' \
                          'WHERE product_group_self.id>0 AND product_clue_sampling.status>1'

sql_corretion_pmi = 'SELECT * FROM ' + path.product_group_sampling_pmi

# sbi
sql_product_sampling = 'select * from clue_sampling_pmi'
sql_product_sbi = 'select * from pt_sbi'
# 查重
sql_clue_self_history = 'select * from product_clue_self'
sql_clue_sampling_history = 'select * from product_clue_sampling'
sql_clue_sbi_history = 'select * from product_clue_sbi'
sql_clue_fbm_history = 'select * from product_clue_fbm'

sampling_knn_sql = 'SELECT product_group_self.*,IF(product_clue_sampling.clue_tag="运营有效",1,0) "clue_tag" FROM ' \
                   'product_clue_sampling LEFT JOIN product_group_self ON ' \
                   'product_clue_sampling.clue_asin=product_group_self.`原ASIN` WHERE product_clue_sampling.`status`=2 AND ' \
                   'product_group_self.`有销额竞品款数`>=5'

sampling_knn_group_sql = 'SELECT * FROM product_group_history WHERE `数据更新时间`="2024-07-01" AND `有销额竞品款数`>=5 AND `重复利基`=0'

# 竞品提报查重
sql_position_competitior = 'SELECT CONCAT(' + path.expand_competitors + '.ASIN," | ",' + path.expand_competitors + \
                           '.site) AS data_id,' + path.expand_competitors + '.id AS expand_competitors_id,' + \
                           'supplement_competitors.* FROM ' + path.expand_competitors + \
                           ' LEFT JOIN supplement_competitors ON ' + path.expand_competitors + '.associate_asin = ' + \
                           'supplement_competitors.asin AND ' + path.expand_competitors + '.site=' + \
                           'supplement_competitors.site WHERE supplement_competitors.id>0 and ' \
                           + path.expand_competitors + '.tag_status=0'

sql_position_ai = 'SELECT ' + path.expand_competitors + '.id AS expand_competitors_id,' + path.expand_competitors + \
                  '.ASIN,' + path.expand_competitors + '.site,' + path.competitors_ai + '.similarity_score,' \
                  + path.competitors_ai + '.overall_competitiveness FROM ' + path.expand_competitors + ' LEFT JOIN ' \
                  + path.competitors_ai + ' ON ' + path.expand_competitors + '.id = ' + path.competitors_ai + \
                  '.parent_id WHERE ' + path.competitors_ai + '.id>0 and ' + path.expand_competitors + '.tag_status=0'

update_position_sql1 = 'UPDATE ' + path.expand_competitors + ' SET tag_status=1 WHERE id IN' \
                                                             '(SELECT expand_competitors_id FROM pt_clue_tag)'
update_position_sql2 = 'UPDATE ' + path.expand_competitors + ' SET tag_status=-1 WHERE id NOT IN' \
                                                             '(SELECT expand_competitors_id FROM pt_clue_tag)'

update_position_sql3 = 'UPDATE ' + path.expand_competitors + ' SET profit_status=1 WHERE asin IN' \
                                                             '(SELECT ASIN FROM pt_clue_profit)'

# sql_clue_asin = 'select asin as ASIN,length_max,length_mid,length_min,weight,price_value from pt_clue_asin where clue_status=1'
sql_clue_asin = 'select asin as ASIN,length_max,length_mid,length_min,weight,price_value from pt_clue_asin'
# 店铺挖掘
sql_brand_report = 'SELECT * FROM ' + path.pt_brand_competing_report + ' WHERE brand_status + seller_status = 2'

sql_seller_product = 'SELECT * FROM ' + path.pt_sellers_product + ' LEFT JOIN ' + path.pt_brand_competing_report + \
                     ' ON ' + path.pt_sellers_product + '.task_tag = ' + path.pt_brand_competing_report + '.task_tag ' + \
                     'WHERE ' + path.pt_sellers_product + '.id IS NULL AND ' + \
                     path.pt_sellers_product + '.brand_status + ' + path.pt_sellers_product + '.seller_status = 2 and ' \
                     + path.pt_sellers_product + '.task_tag="某杂货公司"'

# 状态码更新
sql_report_brand_status = 'UPDATE pt_sellers_product INNER JOIN pt_brand_insight ON ' \
                          'pt_sellers_product.task_tag=pt_brand_insight.task_tag AND ' \
                          'pt_sellers_product.brand=pt_brand_insight.`name` SET ' \
                          'pt_sellers_product.brand_status=1 WHERE pt_brand_insight.`status`=1 ' \
                          'AND pt_sellers_product.brand_status=0;'

sql_report_seller_status = 'UPDATE pt_sellers_product INNER JOIN pt_sellers_insight ON ' \
                           'pt_sellers_product.task_tag=pt_sellers_insight.task_tag AND ' \
                           'pt_sellers_product.buybox_seller=pt_sellers_insight.`name` SET ' \
                           'pt_sellers_product.seller_status=1 WHERE pt_sellers_insight.`status`=1 ' \
                           'AND pt_sellers_product.seller_status=0;'

sql_seller_brand_status = 'UPDATE pt_brand_competing_report INNER JOIN pt_brand_insight ON ' \
                          'pt_brand_competing_report.task_tag=pt_brand_insight.task_tag AND ' \
                          'pt_brand_competing_report.brand=pt_brand_insight.`name` SET ' \
                          'pt_brand_competing_report.brand_status=1 WHERE pt_brand_insight.`status`=1 ' \
                          'AND pt_brand_competing_report.brand_status=0;'

sql_seller_seller_status = 'UPDATE pt_brand_competing_report INNER JOIN pt_sellers_insight ON ' \
                           'pt_brand_competing_report.task_tag=pt_sellers_insight.task_tag AND ' \
                           'pt_brand_competing_report.buybox_seller=pt_sellers_insight.`name` SET ' \
                           'pt_brand_competing_report.seller_status=1 WHERE pt_sellers_insight.`status`=1 ' \
                           'AND pt_brand_competing_report.seller_status=0;'

update_brand_report_sql = 'UPDATE ' + path.pt_brand_competing_report + \
                          ' SET brand_status =2, seller_status = 2 WHERE brand_status + seller_status = 2'
