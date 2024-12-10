# 做数据库操作，连接数据库等等


def group_asin_sql(group_page_size, group_start_index):
    sql_group_asin = """
    SELECT
        asin AS "related_asin",price,recommend,blue_ocean_estimate
    FROM
        pt_product_get_group
    """
    sql_asin = sql_group_asin + ' LIMIT ' + str(group_page_size) + ' OFFSET ' + str(group_start_index)
    return sql_asin


def group_duplicate_sql(group_page_size, group_start_index, group_duplicate):
    sql_duplicate = 'SELECT ' + group_duplicate + '.asin AS "related_asin",' + group_duplicate + '.rank as pmi_rank,' \
                    + group_duplicate + '.duplicate_tag,' + group_duplicate + '.duplicate_type FROM (' + \
                    group_asin_sql(group_page_size, group_start_index) + ') pt_product INNER JOIN ' + group_duplicate \
                    + ' ON pt_product.related_asin = ' + group_duplicate + '.asin'
    return sql_duplicate


def group_traffic_sql(group_page_size, group_start_index, group_relevance, group_traffic):
    sql_relevance = 'SELECT ' + group_relevance + '.* FROM (' + group_asin_sql(group_page_size, group_start_index) + \
                    ') pt_product INNER JOIN ' + group_relevance + \
                    ' ON pt_product.related_asin = ' + group_relevance + '.asin'
    sql_traffic = 'SELECT pt_relevance.asin as "related_asin",pt_relevance.relevance,pt_relevance.category_relevance,' \
                  + group_traffic + '.*,SUBSTRING_INDEX(' + group_traffic + '.category_path,":",2) as "二级类目" FROM( ' \
                  + sql_relevance + ' ) pt_relevance INNER JOIN ' + group_traffic + \
                  ' ON pt_relevance.relation_traffic_id = ' + group_traffic + '.id'
    return sql_traffic


def group_traffic_add_sql(group_page_size, group_start_index, group_supplement_competitors):
    sql_traffic_add = 'SELECT ' + group_supplement_competitors + '.clue_asin as related_asin,' \
                      + group_supplement_competitors + '.*,SUBSTRING_INDEX(' + group_supplement_competitors + \
                      '.category_path,":",2) as "二级类目" FROM ( ' + group_asin_sql(group_page_size, group_start_index) \
                      + ' ) pt_asin LEFT JOIN ' + group_supplement_competitors + ' ON pt_asin.related_asin=' + \
                      group_supplement_competitors + '.clue_asin WHERE supplement_competitors.id>0'
    return sql_traffic_add


"""
sql_asin = 'select asin as "related_asin",price,recommend,blue_ocean_estimate from ' + path.pt_product_get_group + \
               ' limit ' + str(page_size) + ' offset ' + str(start_index)
sql_relevance = 'SELECT ' + path.pt_relevance_asins + '.* FROM (' + sql_asin + ') pt_product LEFT JOIN ' + \
                    path.pt_relevance_asins + ' ON pt_product.related_asin = ' + path.pt_relevance_asins + '.asin'
sql_traffic = 'SELECT pt_relevance.asin as "related_asin",pt_relevance.relevance,pt_relevance.category_relevance,' \
                  + path.pt_relation_traffic + '.*,SUBSTRING_INDEX(' + path.pt_relation_traffic + \
                  '.category_path,":",2) as "二级类目" FROM( ' + sql_relevance + ' ) pt_relevance LEFT JOIN ' + \
                  path.pt_relation_traffic + ' ON pt_relevance.relation_traffic_id = ' + path.pt_relation_traffic + \
                  '.id WHERE pt_relevance.id>0'
"""
