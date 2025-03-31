from better.better.niche_data import niche_data_path as path

# 创建索引
create_index_sql_pt_category = 'CREATE INDEX parent_id ON pt_category(parent_id);'
create_index_sql_pt_niche_commodity = 'CREATE INDEX asin ON pt_niche_commodity(asin);'
create_index_sql_pt_commodity = 'CREATE INDEX asin ON pt_commodity(asin);'
create_index_sql_pt_niche_trends = 'CREATE INDEX niche_id ON pt_niche_trends(niche_id);'
create_index_sql_pt_keywords = 'CREATE INDEX niche_id ON pt_keywords(niche_id);'
create_index_sql_pt_keywords_kw = 'CREATE INDEX kw ON pt_keywords(keyword);'
create_index_sql_pt_keywords_sv = 'CREATE INDEX sv ON pt_keywords(search_volume_t_360);'
create_index_sql_cpc_from_keywords = 'CREATE INDEX parent_id ON cpc_from_keywords(parent_id);'

niche_famous_brand = 'select 关键词, 类型 from ' + path.niche_forbidden + ' where 类型 = "知名品牌"'

niche_x = 'select 利基站点,"类目_X" from ' + path.niche_x

niche_season = 'select 利基站点, 季节性标签 from ' + path.niche_season + ' where 旺季个数 > 0'

niche_translate = 'select 利基站点, 中文名 from ' + path.niche_translate

niche_cpc = 'select 利基站点, CPC from ' + path.niche_cpc

niche_new = 'select 利基站点 from ' + path.niche_new

niche_keywords_cpc_sql = """
    SELECT
	parent_id,
	keyword AS 'keyword_amz',
	bid_rangeMedian,
	bid_rangeEnd 
FROM
	cpc_from_keywords 
WHERE
	crawler_status =1
	"""

niche_category_sql = """
SELECT
	`c`.`一级类目id` AS `一级类目id`,
	`c`.`一级类目名称` AS `一级类目名称`,
	`c`.`二级类目id` AS `二级类目id`,
	`c`.`二级类目名称` AS `二级类目名称`,
	`c`.`三级类目id` AS `三级类目id`,
	`c`.`三级类目名称` AS `三级类目名称`,
	`pt_category`.`category_id` AS `四级类目id`,
	`pt_category`.`category_name` AS `四级类目名称` 
FROM
	(
		`pt_category`
		LEFT JOIN (
		SELECT
			`b`.`一级类目id` AS `一级类目id`,
			`b`.`一级类目名称` AS `一级类目名称`,
			`b`.`二级类目id` AS `二级类目id`,
			`b`.`二级类目名称` AS `二级类目名称`,
			`pt_category`.`category_id` AS `三级类目id`,
			`pt_category`.`category_name` AS `三级类目名称` 
		FROM
			(
				`pt_category`
				LEFT JOIN (
				SELECT
					`a`.`category_id` AS `一级类目id`,
					`a`.`category_name` AS `一级类目名称`,
					`pt_category`.`category_id` AS `二级类目id`,
					`pt_category`.`category_name` AS `二级类目名称` 
				FROM
					(
						`pt_category`
						LEFT JOIN (
						SELECT
							`pt_category`.`category_id` AS `category_id`,
							`pt_category`.`category_name` AS `category_name` 
						FROM
							`pt_category` 
						WHERE
							( `pt_category`.`parent_id` IS NULL )) `a` ON ((
								`pt_category`.`parent_id` = `a`.`category_id` 
							))) 
				WHERE
					((
							`pt_category`.`parent_id` IS NOT NULL 
							) 
						AND ( `a`.`category_id` IS NOT NULL ))) `b` ON ((
						`pt_category`.`parent_id` = `b`.`二级类目id` 
					))) 
		WHERE
			((
					`pt_category`.`parent_id` IS NOT NULL 
					) 
				AND ( `b`.`一级类目id` IS NOT NULL ))) `c` ON ((
				`pt_category`.`parent_id` = `c`.`三级类目id` 
			))) 
WHERE
	((
			`pt_category`.`parent_id` IS NOT NULL 
		) 
	AND ( `c`.`一级类目id` IS NOT NULL ))
"""

# niche_smile表处理
clear_smile_sql = "TRUNCATE TABLE " + path.niche_smile
insert_smile_sql = 'INSERT INTO `niche_smile` (' \
                   '`利基站点`,' \
                   '`利基图片URL`,' \
                   '`所属利基`,' \
                   '`站点`,' \
                   '`是否探索`,' \
                   '`中文名`,' \
                   '`业务线`,' \
                   '`SMILE打分`,' \
                   '`Scale`,' \
                   '`Monetary`,' \
                   '`Involution`,' \
                   '`Longtail`,' \
                   '`Emerging`,' \
                   '`平均价格`,' \
                   '`价格集中度`,' \
                   '`搜索量_360`,' \
                   '`月均GMV`,' \
                   '`TOP5平均月销额`,' \
                   '`总售出件数_360`,' \
                   '`搜索量_90`,' \
                   '`资金利用效率AVG`,' \
                   '`资金利用效率1`,' \
                   '`资金利用效率2`,' \
                   '`加权平均重量`,' \
                   '`加权平均体积重`,' \
                   '`实抛偏差率`,' \
                   '`体积超重占比`,' \
                   '`重量分布`,' \
                   '`实抛分布`,' \
                   '`平均每单件数`,' \
                   '`加权平均FBA`,' \
                   '`FBA占比`,' \
                   '`FBA货值比`,' \
                   '`头程`,' \
                   '`头程占比`,' \
                   '`货值占比`,' \
                   '`营销前毛利反推`,' \
                   '`转化净值`,' \
                   '`转化净值偏差`,' \
                   '`转化净值分级`,' \
                   '`搜索点击比`,' \
                   '`CR`,' \
                   '`毛估蓝海度`,' \
                   '`CPC`,' \
                   '`广告蓝海度`,' \
                   '`蓝海度差异分`,' \
                   '`广告权重`,' \
                   '`综合蓝海度`,' \
                   '`搜索广告占比档`,' \
                   '`搜索广告商品占比_90`,' \
                   '`搜索广告商品占比_360`,' \
                   '`平均产品星级`,' \
                   '`商品listing平均得分_90`,' \
                   '`周SCR_AVG`,' \
                   '`长尾指数`,' \
                   '`点击最多的商品数`,' \
                   '`品牌长尾指数`,' \
                   '`品牌数量_360`,' \
                   '`非TOP5单品月销量`,' \
                   '`平均单品月销量`,' \
                   '`非TOP5平均月销额`,' \
                   '`利基平均月销售额`,' \
                   '`TOP5产品点击占比_360`,' \
                   '`TOP5品牌点击占比_360`,' \
                   '`知名品牌依赖`,' \
                   '`搜索量增长_360`,' \
                   '`搜索量增长_90`,' \
                   '`数据趋势覆盖周数`,' \
                   '`ASIN平均上架年数`,' \
                   '`留评意愿强度`,' \
                   '`平均评论数`,' \
                   '`销量潜力得分`,' \
                   '`近半年新品成功率`,' \
                   '`近180天上架新品数_90`,' \
                   '`近180天上架新品数_360`,' \
                   '`近180天成功上架新品数_90`,' \
                   '`近180天成功上架新品数_360`,' \
                   '`平均缺货率`,' \
                   '`平均缺货率_90`,' \
                   '`平均缺货率_360`,' \
                   '`轻小件规格`,' \
                   '`大件规格`,' \
                   '`准轻小件`,' \
                   '`存疑利基`,' \
                   '`季节性标签`,' \
                   '`ASIN1`,' \
                   '`ASIN1完整类名`,' \
                   '`ASIN2`,' \
                   '`ASIN2完整类名`,' \
                   '`ASIN3`,' \
                   '`ASIN3完整类名`,' \
                   '`类目路径`,' \
                   '`一级类目`,' \
                   '`新冒出利基`,' \
                   '`搜索词`,' \
                   '`数据更新时间`)' \
                   'SELECT' \
                   '`利基站点`,' \
                   '`利基图片URL`,' \
                   '`所属利基`,' \
                   '`站点`,' \
                   '`是否探索`,' \
                   '`中文名`,' \
                   '`业务线`,' \
                   '`SMILE打分`,' \
                   '`Scale`,' \
                   '`Monetary`,' \
                   '`Involution`,' \
                   '`Longtail`,' \
                   '`Emerging`,' \
                   '`平均价格`,' \
                   '`价格集中度`,' \
                   '`搜索量_360`,' \
                   '`月均GMV`,' \
                   '`TOP5平均月销额`,' \
                   '`总售出件数_360`,' \
                   '`搜索量_90`,' \
                   '`资金利用效率AVG`,' \
                   'null as "资金利用效率1",' \
                   'null as "资金利用效率2",' \
                   '`加权平均重量`,' \
                   '`加权平均体积重`,' \
                   '`实抛偏差率`,' \
                   '`体积超重占比`,' \
                   '`重量分布`,' \
                   '`实抛分布`,' \
                   '`平均每单件数`,' \
                   '`加权平均FBA`,' \
                   '`FBA占比`,' \
                   '`FBA货值比`,' \
                   '`头程`,' \
                   '`头程占比`,' \
                   '`货值占比`,' \
                   '`营销前毛利反推`,' \
                   '`转化净值`,' \
                   '`转化净值偏差`,' \
                   '`转化净值分级`,' \
                   '`搜索点击比`,' \
                   '`CR`,' \
                   '`毛估蓝海度`,' \
                   '`CPC`,' \
                   '`广告蓝海度`,' \
                   '`蓝海度差异分`,' \
                   '`广告权重`,' \
                   '`综合蓝海度`,' \
                   '`搜索广告占比档`,' \
                   '`搜索广告商品占比_90`,' \
                   '`搜索广告商品占比_360`,' \
                   '`平均产品星级`,' \
                   '`商品listing平均得分_90`,' \
                   '`周SCR_AVG`,' \
                   '`长尾指数`,' \
                   '`点击最多的商品数`,' \
                   '`品牌长尾指数`,' \
                   '`品牌数量_360`,' \
                   '`非TOP5单品月销量`,' \
                   '`平均单品月销量`,' \
                   '`非TOP5平均月销额`,' \
                   '`利基平均月销售额`,' \
                   '`TOP5产品点击占比_360`,' \
                   '`TOP5品牌点击占比_360`,' \
                   '`知名品牌依赖`,' \
                   '`搜索量增长_360`,' \
                   '`搜索量增长_90`,' \
                   '`数据趋势覆盖周数`,' \
                   '`ASIN平均上架年数`,' \
                   '`留评意愿强度`,' \
                   '`平均评论数`,' \
                   '`销量潜力得分`,' \
                   '`近半年新品成功率`,' \
                   '`近180天上架新品数_90`,' \
                   '`近180天上架新品数_360`,' \
                   '`近180天成功上架新品数_90`,' \
                   '`近180天成功上架新品数_360`,' \
                   '`平均缺货率`,' \
                   '`平均缺货率_90`,' \
                   '`平均缺货率_360`,' \
                   'null as "轻小件规格",' \
                   'null as "大件规格",' \
                   'null as "准轻小件",' \
                   'null as "存疑利基",' \
                   '`季节性标签`,' \
                   '`ASIN1`,' \
                   '`ASIN1完整类名`,' \
                   '`ASIN2`,' \
                   '`ASIN2完整类名`,' \
                   '`ASIN3`,' \
                   '`ASIN3完整类名`,' \
                   '`类目路径`,' \
                   '`一级类目`,' \
                   'null as "新冒出利基",' \
                   'null as "搜索词",' \
                   '`数据更新时间`' \
                   'FROM' \
                   '(SELECT *,ROW_NUMBER() over (PARTITION by `利基站点` ORDER BY `数据更新时间` DESC) as number FROM niche_smile_h) smile_h_clear' \
                   ' WHERE number=1 AND `数据更新时间`>=(SELECT DATE_SUB(MAX(`数据更新时间`),INTERVAL 180 DAY) end_date FROM niche_smile_h);'

update_smile_url_sql = "UPDATE niche_smile INNER JOIN niche_top_asin_unique ON " \
                       "niche_smile.`利基站点`=niche_top_asin_unique.`利基站点` " \
                       "SET niche_smile.`利基图片URL`=niche_top_asin_unique.`利基图片URL` " \
                       "WHERE niche_top_asin_unique.`利基图片URL` IS NOT NULL AND niche_smile.`利基图片URL` IS NULL;"

# 打标签
clear_smile_tag_sql = "DROP TABLE niche_tag;"
create_smile_tag_sql = 'CREATE TABLE niche_tag' \
                       '(SELECT niche_smile.`利基站点`,' \
                       '(CASE ' \
                       'WHEN niche_smile.`转化净值偏差`>=3 THEN "6"' \
                       'WHEN niche_smile.`转化净值偏差`>=2 THEN "5"' \
                       'WHEN niche_smile.`转化净值偏差`>=1.5 THEN "4"' \
                       'WHEN niche_smile.`转化净值偏差`>=0.9 THEN "3"' \
                       'WHEN niche_smile.`转化净值偏差`>=0.7 THEN "2"' \
                       'ELSE "0" END) "转化净值等级排序",' \
                       'IF(`点击最多的商品数`>=20 AND `搜索量增长_360`>=0 AND `搜索量增长_90`>=0.05 AND `利基平均月销售额`>=600 ' \
                       'AND `转化净值偏差`>=1.5 AND niche_smile.`平均价格`<=250 AND niche_smile.`转化净值偏差`<2,' \
                       '"精铺-有增长鸡肋POST专享",NULL) "精铺-有增长鸡肋POST专享",' \
                       'IF(niche_smile.`点击最多的商品数`<=10 AND niche_smile.`转化净值偏差`>2,"精铺-高净值低产品数",NULL) "精铺-高净值低产品数",' \
                       'IF(niche_smile.`平均价格`>=7 AND niche_smile.`平均价格`<=14 AND niche_smile.`点击最多的商品数`>=5 ' \
                       'AND niche_smile.`搜索量增长_360`>=0.1 AND niche_smile.`搜索量增长_90`>=0.1 AND ' \
                       'niche_smile.`利基平均月销售额`>=600,"精铺-高增长轻小件",NULL) "精铺-高增长轻小件",' \
                       'IF(niche_smile.`点击最多的商品数`>=40 AND niche_smile.`搜索量增长_360`>=0.1 AND ' \
                       'niche_smile.`搜索量增长_90`>=0.1 AND niche_smile.`利基平均月销售额`>=4000,"高增高潜高分数",NULL) "高增高潜高分数",' \
                       'IF(niche_smile.`点击最多的商品数`>=20 AND niche_smile.`搜索量增长_360`>=-0.1 AND ' \
                       'niche_smile.`搜索量增长_90`>=0 AND niche_smile.`利基平均月销售额`>=600 AND ' \
                       'niche_smile.`平均评论数`<=1000 AND niche_smile.`转化净值偏差`>=2,"有增长高潜新兴市场-非高价",NULL) "有增长高潜新兴市场-非高价",' \
                       'IF(niche_smile.`平均价格`<=50 AND niche_smile.`转化净值偏差`>=2,"低价+高售价转化",NULL) "低价+高售价转化",' \
                       'IF(niche_smile.`平均产品星级`<=4.2 AND niche_smile.`平均单品月销量`>=300,"低星级高销量",NULL) "低星级高销量",' \
                       'IF(niche_smile.`点击最多的商品数`>=4 AND niche_smile.`搜索量增长_360`>=-0.2 AND ' \
                       'niche_smile.`搜索量增长_90`>=0 AND niche_smile.`平均每单件数`>=1.3 AND ' \
                       'niche_smile.`利基平均月销售额`>=600 AND niche_smile.`转化净值偏差`>=2,"精铺-高增长高每单件数",NULL) "精铺-高增长高每单件数",' \
                       'IF(niche_smile.`平均价格`>=50 AND niche_smile.`平均价格`<=120 AND ' \
                       'niche_smile.`转化净值偏差`>=2,"中高价+高转化净值",NULL) "中高价+高转化净值",' \
                       'IF(niche_smile.`平均价格`>=30 AND niche_smile.`点击最多的商品数`>=3 AND ' \
                       'niche_smile.`搜索量增长_360`>=-0.1 AND niche_smile.`搜索量增长_90`>=0.1 AND ' \
                       'niche_smile.`利基平均月销售额`>=4000 AND niche_smile.`转化净值偏差`>=2,"中高客单价高增长",NULL) "中高客单价高增长",' \
                       'ROUND(niche_smile.Scale) "Scale分布",' \
                       'ROUND(niche_smile.Monetary) "Monetary分布",' \
                       'ROUND(niche_smile.Involution) "Involution分布",' \
                       'ROUND(niche_smile.Longtail) "Longtail分布",' \
                       'ROUND(niche_smile.Emerging) "Emerging分布",' \
                       'ROUND(niche_smile.`搜索量_360`/10000)*10000 "年搜索量分布",' \
                       'ROUND(niche_smile.`月均GMV`/10000)*10000 "月均GMV分布",' \
                       'ROUND(niche_smile.`TOP5平均月销额`/10000)*10000 "TOP5平均月销额分布",' \
                       'ROUND(niche_smile.`平均价格`/2)*2 "平均价格分布",' \
                       'ROUND(niche_smile.`毛估蓝海度`,1) "毛估蓝海度分布",' \
                       'ROUND(niche_smile.`广告蓝海度`,1) "广告蓝海度分布",' \
                       'ROUND(niche_smile.`综合蓝海度`,1) "综合蓝海度分布",' \
                       'ROUND(niche_smile.`平均每单件数`,1) "平均每单件数分布",' \
                       'ROUND(niche_smile.`搜索点击比`,1) "搜索点击比分布",' \
                       'ROUND(niche_smile.`搜索广告占比档`/0.02)*0.02 "搜索广告占比档分布",' \
                       'ROUND(niche_smile.`平均产品星级`,1) "平均产品星级分布",' \
                       'ROUND(niche_smile.`长尾指数`/0.05)*0.05 "长尾指数分布",' \
                       'ROUND(niche_smile.`品牌长尾指数`/0.05)*0.05 "品牌长尾指数分布",' \
                       'ROUND(niche_smile.`平均单品月销量`/100)*100 "平均单品月销量分布",' \
                       'ROUND(niche_smile.`TOP5产品点击占比_360`,2) "TOP5产品点击占比_360分布",' \
                       'ROUND(niche_smile.`TOP5品牌点击占比_360`,2) "TOP5品牌点击占比_360分布",' \
                       'ROUND(niche_smile.`搜索量增长_360`,2) "搜索量增长_360分布",' \
                       'ROUND(niche_smile.`搜索量增长_90`,2) "搜索量增长_90分布",' \
                       'ROUND(niche_smile.`ASIN平均上架年数`,1) "ASIN平均上架年数分布",' \
                       'ROUND(niche_smile.`平均评论数`/100)*100 "平均评论数分布",' \
                       'ROUND(niche_smile.`平均缺货率`/0.02)*0.02 "平均缺货率分布",' \
                       'IF (' \
                       'niche_smile.`SMILE打分`>=11 ' \
                       'AND niche_smile.`Involution`>=3.5 ' \
                       'AND niche_smile.`Longtail`>=2.5 ' \
                       'AND niche_smile.`Emerging`>=2 ' \
                       'AND niche_smile.`平均价格`>=8 AND niche_smile.`平均价格`<=250 ' \
                       'AND niche_smile.`加权平均FBA`>=0 AND niche_smile.`加权平均FBA`<=30 ' \
                       'AND niche_smile.`头程`>=0 AND niche_smile.`头程`<=50 ' \
                       'AND niche_smile.`CR`>=0.15 ' \
                       'AND niche_smile.`毛估蓝海度`>=2 ' \
                       'AND niche_smile.`广告蓝海度`>=2 ' \
                       'AND niche_smile.`长尾指数`>=0.25 ' \
                       'AND niche_smile.`点击最多的商品数`>=12 ' \
                       'AND niche_smile.`TOP5品牌点击占比_360`>=0 AND niche_smile.`TOP5品牌点击占比_360`<=0.82 ' \
                       'AND niche_smile.`ASIN平均上架年数`>=0 AND niche_smile.`ASIN平均上架年数`<=8 ' \
                       'AND niche_smile.`平均评论数`>=0 AND niche_smile.`平均评论数`<=8000,' \
                       '"子策略-高转化市场",' \
                       'NULL ' \
                       ') "子策略-高转化市场",' \
                       'IF (' \
                       'niche_smile.`Scale`>=0 ' \
                       'AND niche_smile.`Involution`>=2.5 ' \
                       'AND niche_smile.`Longtail`>=4 ' \
                       'AND niche_smile.`Emerging`>=3.5 ' \
                       'AND niche_smile.`平均价格`>=8 AND niche_smile.`平均价格`<=250 ' \
                       'AND niche_smile.`加权平均FBA`>=0 AND niche_smile.`加权平均FBA`<=30 ' \
                       'AND niche_smile.`头程`>=0 AND niche_smile.`头程`<=50 ' \
                       'AND niche_smile.`毛估蓝海度`>=2 ' \
                       'AND niche_smile.`广告蓝海度`>=2 ' \
                       'AND niche_smile.`平均产品星级`>=4 ' \
                       'AND niche_smile.`长尾指数`>=0.25 ' \
                       'AND niche_smile.`点击最多的商品数`>=12 ' \
                       'AND niche_smile.`TOP5品牌点击占比_360`>=0 AND niche_smile.`TOP5品牌点击占比_360`<=0.7 ' \
                       'AND niche_smile.`ASIN平均上架年数`>=0 AND niche_smile.`ASIN平均上架年数`<=8 ' \
                       'AND niche_smile.`平均评论数`>=0 AND niche_smile.`平均评论数`<=8000,' \
                       '"子策略-精铺长尾市场",' \
                       'NULL ' \
                       ') "子策略精铺长尾市场",' \
                       'IF (' \
                       'niche_smile.`Scale`>=0 ' \
                       'AND niche_smile.`Involution`>=2.5 ' \
                       'AND niche_smile.`Longtail`>=3 ' \
                       'AND niche_smile.`Emerging`>=4 ' \
                       'AND niche_smile.`平均价格`>=8 AND niche_smile.`平均价格`<=250 ' \
                       'AND niche_smile.`加权平均FBA`>=0 AND niche_smile.`加权平均FBA`<=30 ' \
                       'AND niche_smile.`头程`>=0 AND niche_smile.`头程`<=50 ' \
                       'AND niche_smile.`毛估蓝海度`>=1.8 ' \
                       'AND niche_smile.`广告蓝海度`>=2 ' \
                       'AND niche_smile.`平均产品星级`>=4 ' \
                       'AND niche_smile.`长尾指数`>=0.25 ' \
                       'AND niche_smile.`点击最多的商品数`>=12 ' \
                       'AND niche_smile.`TOP5品牌点击占比_360`>=0 AND niche_smile.`TOP5品牌点击占比_360`<=0.7 ' \
                       'AND niche_smile.`ASIN平均上架年数`>=0 AND niche_smile.`ASIN平均上架年数`<=2.5 ' \
                       'AND niche_smile.`平均评论数`>=0 AND niche_smile.`平均评论数`<=4000,' \
                       '"子策略-精铺新兴市场",' \
                       'NULL ' \
                       ') "子策略-精铺新兴市场",' \
                       'IF (' \
                       'niche_smile.`Scale`>=0 ' \
                       'AND niche_smile.`Involution`>=2.5 ' \
                       'AND niche_smile.`Longtail`>=2.5 ' \
                       'AND niche_smile.`Emerging`>=3.5 ' \
                       'AND niche_smile.`平均价格`>=8 AND niche_smile.`平均价格`<=250 ' \
                       'AND niche_smile.`加权平均FBA`>=0 AND niche_smile.`加权平均FBA`<=30 ' \
                       'AND niche_smile.`头程`>=0 AND niche_smile.`头程`<=50 ' \
                       'AND niche_smile.`毛估蓝海度`>=2 ' \
                       'AND niche_smile.`广告蓝海度`>=2 ' \
                       'AND niche_smile.`长尾指数`>=0.25 ' \
                       'AND niche_smile.`TOP5品牌点击占比_360`>=0 AND niche_smile.`TOP5品牌点击占比_360`<=0.7 ' \
                       'AND niche_smile.`ASIN平均上架年数`>=0 AND niche_smile.`ASIN平均上架年数`<=1 ' \
                       'AND niche_smile.`平均评论数`>=0 AND niche_smile.`平均评论数`<=4000,' \
                       '"子策略-超新市场",' \
                       'NULL ' \
                       ') "子策略-超新市场",' \
                       'IF (' \
                       'niche_smile.`SMILE打分`>=9 ' \
                       'AND niche_smile.`Longtail`>=3 ' \
                       'AND niche_smile.`Emerging`>=2.5 ' \
                       'AND niche_smile.`平均价格`>=8 AND niche_smile.`平均价格`<=250 ' \
                       'AND niche_smile.`加权平均FBA`>=0 AND niche_smile.`加权平均FBA`<=30 ' \
                       'AND niche_smile.`头程`>=0 AND niche_smile.`头程`<=50 ' \
                       'AND niche_smile.`毛估蓝海度`>=2 ' \
                       'AND niche_smile.`广告蓝海度`>=2 ' \
                       'AND niche_smile.`商品listing平均得分_90`>=0 AND niche_smile.`商品listing平均得分_90`<=80 ' \
                       'AND niche_smile.`长尾指数`>=0.25 ' \
                       'AND niche_smile.`点击最多的商品数`>=12 ' \
                       'AND niche_smile.`TOP5品牌点击占比_360`>=0 AND niche_smile.`TOP5品牌点击占比_360`<=0.75 ' \
                       'AND niche_smile.`ASIN平均上架年数`>=0 AND niche_smile.`ASIN平均上架年数`<=8 ' \
                       'AND niche_smile.`平均评论数`>=0 AND niche_smile.`平均评论数`<=8000,' \
                       '"子策略-listing差市场",' \
                       'NULL ' \
                       ') "子策略-listing差市场",' \
                       'IF (' \
                       'niche_smile.`SMILE打分`>=12 ' \
                       'AND niche_smile.`Involution`>=3 ' \
                       'AND niche_smile.`Longtail`>=2.5 ' \
                       'AND niche_smile.`Emerging`>=2 ' \
                       'AND niche_smile.`平均价格`>=49 AND niche_smile.`平均价格`<=600 ' \
                       'AND niche_smile.`加权平均FBA`>=8 AND niche_smile.`加权平均FBA`<=100 ' \
                       'AND niche_smile.`头程`>=8 AND niche_smile.`头程`<=100 ' \
                       'AND niche_smile.`毛估蓝海度`>=2.3 ' \
                       'AND niche_smile.`广告蓝海度`>=2.5 ' \
                       'AND niche_smile.`搜索广告占比档`>=0 AND niche_smile.`搜索广告占比档`<=0.85 ' \
                       'AND niche_smile.`长尾指数`>=0.25 ' \
                       'AND niche_smile.`点击最多的商品数`>=10 ' \
                       'AND niche_smile.`TOP5品牌点击占比_360`>=0 AND niche_smile.`TOP5品牌点击占比_360`<=0.82 ' \
                       'AND niche_smile.`ASIN平均上架年数`>=0 AND niche_smile.`ASIN平均上架年数`<=8 ' \
                       'AND niche_smile.`平均评论数`>=0 AND niche_smile.`平均评论数`<=8000,' \
                       '"子策略-大件低广告市场",' \
                       'NULL ' \
                       ') "子策略-大件低广告市场",' \
                       'IF (' \
                       'niche_smile.`SMILE打分`>=13 ' \
                       'AND niche_smile.`Monetary`>=0 AND niche_smile.`Monetary`<=3 ' \
                       'AND niche_smile.`Involution`>=3 ' \
                       'AND niche_smile.`Longtail`>=2.5 ' \
                       'AND niche_smile.`Emerging`>=2 ' \
                       'AND niche_smile.`平均价格`>=8 AND niche_smile.`平均价格`<=150 ' \
                       'AND niche_smile.`实抛分布`="大抛重" AND niche_smile.`实抛分布`="小抛重" ' \
                       'AND niche_smile.`平均每单件数`>=1.6 AND niche_smile.`平均每单件数`<=6 ' \
                       'AND niche_smile.`加权平均FBA`>=0 AND niche_smile.`加权平均FBA`<=20 ' \
                       'AND niche_smile.`头程`>=0 AND niche_smile.`头程`<=20 ' \
                       'AND niche_smile.`毛估蓝海度`>=2 ' \
                       'AND niche_smile.`广告蓝海度`>=2.5 ' \
                       'AND niche_smile.`搜索广告占比档`>=0 ' \
                       'AND niche_smile.`长尾指数`>=0.25 ' \
                       'AND niche_smile.`点击最多的商品数`>=10 ' \
                       'AND niche_smile.`TOP5品牌点击占比_360`>=0 AND niche_smile.`TOP5品牌点击占比_360`<=0.8 ' \
                       'AND niche_smile.`ASIN平均上架年数`>=0 AND niche_smile.`ASIN平均上架年数`<=8 ' \
                       'AND niche_smile.`平均评论数`>=0 AND niche_smile.`平均评论数`<=9000,' \
                       '"子策略-规格多件购市场",' \
                       'NULL ' \
                       ') "子策略-规格多件购市场",' \
                       'IF (' \
                       'niche_smile.`SMILE打分`>=11 ' \
                       'AND niche_smile.`Involution`>=2 ' \
                       'AND niche_smile.`Longtail`>=2.5 ' \
                       'AND niche_smile.`Emerging`>=2 ' \
                       'AND niche_smile.`平均价格`>=8 AND niche_smile.`平均价格`<=250 ' \
                       'AND niche_smile.`加权平均FBA`>=0 AND niche_smile.`加权平均FBA`<=30 ' \
                       'AND niche_smile.`头程`>=0 AND niche_smile.`头程`<=50 ' \
                       'AND niche_smile.`毛估蓝海度`>=2 ' \
                       'AND niche_smile.`广告蓝海度`>=2 ' \
                       'AND niche_smile.`搜索广告占比档`>=0 AND niche_smile.`搜索广告占比档`<=0.68 ' \
                       'AND niche_smile.`长尾指数`>=0.25 ' \
                       'AND niche_smile.`点击最多的商品数`>=12 ' \
                       'AND niche_smile.`TOP5品牌点击占比_360`>=0 AND niche_smile.`TOP5品牌点击占比_360`<=0.75 ' \
                       'AND niche_smile.`ASIN平均上架年数`>=0 AND niche_smile.`ASIN平均上架年数`<=8 ' \
                       'AND niche_smile.`平均评论数`>=0 AND niche_smile.`平均评论数`<=8000,' \
                       '"子策略-低广告市场",' \
                       'NULL ' \
                       ') "子策略-低广告市场",' \
                       'IF (' \
                       'niche_smile.`SMILE打分`>=12 ' \
                       'AND niche_smile.`Scale`>=2 ' \
                       'AND niche_smile.`Involution`>=3.5 ' \
                       'AND niche_smile.`Longtail`>=3 ' \
                       'AND niche_smile.`Emerging`>=3 ' \
                       'AND niche_smile.`平均价格`>=8 AND niche_smile.`平均价格`<=250 ' \
                       'AND niche_smile.`加权平均FBA`>=0 AND niche_smile.`加权平均FBA`<=30 ' \
                       'AND niche_smile.`头程`>=0 AND niche_smile.`头程`<=50 ' \
                       'AND niche_smile.`毛估蓝海度`>=2 ' \
                       'AND niche_smile.`广告蓝海度`>=2 ' \
                       'AND niche_smile.`长尾指数`>=0.25 ' \
                       'AND niche_smile.`点击最多的商品数`>=12 ' \
                       'AND niche_smile.`TOP5品牌点击占比_360`>=0 AND niche_smile.`TOP5品牌点击占比_360`<=0.78 ' \
                       'AND niche_smile.`ASIN平均上架年数`>=0 AND niche_smile.`ASIN平均上架年数`<=8 ' \
                       'AND niche_smile.`平均评论数`>=0 AND niche_smile.`平均评论数`<=8000,' \
                       '"子策略-高分启发",' \
                       'NULL ' \
                       ') "子策略-高分启发",' \
                       'IF (' \
                       'niche_smile.`SMILE打分`>=11 ' \
                       'AND niche_smile.`Involution`>=2 ' \
                       'AND niche_smile.`Longtail`>=2.5 ' \
                       'AND niche_smile.`Emerging`>=2 ' \
                       'AND niche_smile.`平均价格`>=9.8 AND niche_smile.`平均价格`<=1000 ' \
                       'AND niche_smile.`总售出件数_360`>=100000 ' \
                       'AND niche_smile.`加权平均FBA`>=0 AND niche_smile.`加权平均FBA`<=80 ' \
                       'AND niche_smile.`头程`>=0 AND niche_smile.`头程`<=80 ' \
                       'AND niche_smile.`毛估蓝海度`>=1.8 ' \
                       'AND niche_smile.`广告蓝海度`>=2 ' \
                       'AND niche_smile.`长尾指数`>=0.25 ' \
                       'AND niche_smile.`点击最多的商品数`>=12 ' \
                       'AND niche_smile.`品牌长尾指数`>=0.25 ' \
                       'AND niche_smile.`TOP5品牌点击占比_360`>=0 AND niche_smile.`TOP5品牌点击占比_360`<=0.82 ' \
                       'AND niche_smile.`ASIN平均上架年数`>=0 AND niche_smile.`ASIN平均上架年数`<=10 ' \
                       'AND niche_smile.`平均评论数`>=0 AND niche_smile.`平均评论数`<=9000,' \
                       '"子策略-头部大坑位",' \
                       'NULL ' \
                       ') "子策略-头部大坑位",' \
                       'CONCAT_WS(",",' \
                       'IF(niche_smile.Involution>=4 AND niche_smile.`平均价格`>150 AND niche_smile.`价格集中度`>0.3 AND ' \
                       'niche_smile.Longtail>=4 AND niche_smile.Emerging>=4,"ILE中有强项",NULL),' \
                       'IF(niche_smile.`搜索广告占比档`<0.3,"超低广告占比",NULL),' \
                       'IF(niche_smile.`搜索广告占比档`<0.6 AND niche_smile.`搜索广告占比档`>=0.3,"低广告占比",NULL),' \
                       'IF(niche_smile.`平均每单件数`>=1.6,"多件购潜力",NULL),' \
                       'IF(niche_smile.`平均每单件数`>=1.6 AND niche_smile.`实抛偏差率`>=0.8,"多件规格优化",NULL)' \
                       ') "特点标签",' \
                       'CONCAT_WS(",",' \
                       'IF(niche_smile.`平均价格`<10,"超低价",NULL),' \
                       'IF(niche_smile.Involution<2 OR niche_smile.`毛估蓝海度`<2,"低蓝海",NULL),' \
                       'IF(niche_smile.`搜索广告占比档`>2.5,"高搜索点击比",NULL),' \
                       'IF(niche_smile.`平均产品星级`<=4.2,"易差评",NULL),' \
                       'IF(niche_smile.`TOP5品牌点击占比_360`>=0.75,"潜在品牌垄断",NULL),' \
                       'IF(niche_smile.`知名品牌依赖`="H" OR niche_smile.`知名品牌依赖`="M","知名品牌",NULL),' \
                       'IF(niche_smile.`平均价格`<20 AND niche_smile.Scale<=3 AND niche_smile.Involution<=3 AND ' \
                       'niche_smile.Longtail<=3 AND niche_smile.Emerging<=3,"低价平庸",NULL),' \
                       'IF(niche_smile.`ASIN平均上架年数`>=6,"老品多",NULL),' \
                       'IF(niche_smile.`平均评论数`>=7000,"评论超多",NULL),' \
                       'IF(niche_smile.`广告蓝海度`>=0 AND niche_smile.`广告蓝海度`<=2.25 AND ' \
                       'ABS(niche_smile.`毛估蓝海度`-niche_smile.`广告蓝海度`)>=1 AND niche_smile.`搜索广告占比档`>=0.75,"蓝海度矛盾大",NULL)' \
                       ') "告警标签"' \
                       'FROM niche_smile);'
