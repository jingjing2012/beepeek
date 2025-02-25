# 做数据库操作，连接数据库等等

def group_asin_sql(id_start, id_increment):
    sql_group_asin = """
    SELECT
        asin AS "related_asin",price,recommend,blue_ocean_estimate
    FROM
        pt_product_get_group
    """
    sql_asin = sql_group_asin + ' WHERE id >' + str(id_start) + ' ORDER BY id ASC LIMIT ' + str(id_increment)
    return sql_asin


def group_duplicate_sql(id_start, id_increment, group_duplicate):
    sql_duplicate = 'SELECT ' + group_duplicate + '.asin AS "related_asin",' + group_duplicate + '.rank as pmi_rank,' \
                    + group_duplicate + '.duplicate_tag,' + group_duplicate + '.duplicate_type FROM (' + \
                    group_asin_sql(id_start, id_increment) + ') pt_product INNER JOIN ' + group_duplicate \
                    + ' ON pt_product.related_asin = ' + group_duplicate + '.asin'
    return sql_duplicate


def group_traffic_sql(id_start, id_increment, group_relevance, group_traffic):
    sql_relevance = 'SELECT ' + group_relevance + '.* FROM (' + group_asin_sql(id_start, id_increment) + \
                    ') pt_product INNER JOIN ' + group_relevance + \
                    ' ON pt_product.related_asin = ' + group_relevance + '.asin'
    sql_traffic = 'SELECT pt_relevance.asin as "related_asin",pt_relevance.relevance,pt_relevance.category_relevance,' \
                  + group_traffic + '.*,SUBSTRING_INDEX(' + group_traffic + '.category_path,":",2) as "二级类目" FROM( ' \
                  + sql_relevance + ' ) pt_relevance INNER JOIN ' + group_traffic + \
                  ' ON pt_relevance.relation_traffic_id = ' + group_traffic + '.id'
    return sql_traffic


def group_traffic_add_sql(id_start, id_increment, group_supplement_competitors):
    sql_traffic_add = 'SELECT ' + group_supplement_competitors + '.clue_asin as related_asin,' \
                      + group_supplement_competitors + '.*,SUBSTRING_INDEX(' + group_supplement_competitors + \
                      '.category_path,":",2) as "二级类目" FROM ( ' + group_asin_sql(id_start, id_increment) \
                      + ' ) pt_asin LEFT JOIN ' + group_supplement_competitors + ' ON pt_asin.related_asin=' + \
                      group_supplement_competitors + '.clue_asin WHERE supplement_competitors.id>0'
    return sql_traffic_add


def group_asin_page_sql(group_page_size, group_start_index):
    sql_group_asin = """
    SELECT
        asin AS "related_asin",price,recommend,blue_ocean_estimate
    FROM
        pt_product_get_group
    """
    sql_asin = sql_group_asin + ' LIMIT ' + str(group_page_size) + ' OFFSET ' + str(group_start_index)
    return sql_asin


def group_duplicate_page_sql(group_page_size, group_start_index, group_duplicate):
    sql_duplicate = 'SELECT ' + group_duplicate + '.asin AS "related_asin",' + group_duplicate + '.rank as pmi_rank,' \
                    + group_duplicate + '.duplicate_tag,' + group_duplicate + '.duplicate_type FROM (' + \
                    group_asin_sql(group_page_size, group_start_index) + ') pt_product INNER JOIN ' + group_duplicate \
                    + ' ON pt_product.related_asin = ' + group_duplicate + '.asin'
    return sql_duplicate


def group_traffic_page_sql(group_page_size, group_start_index, group_relevance, group_traffic):
    sql_relevance = 'SELECT ' + group_relevance + '.* FROM (' + group_asin_sql(group_page_size, group_start_index) + \
                    ') pt_product INNER JOIN ' + group_relevance + \
                    ' ON pt_product.related_asin = ' + group_relevance + '.asin'
    sql_traffic = 'SELECT pt_relevance.asin as "related_asin",pt_relevance.relevance,pt_relevance.category_relevance,' \
                  + group_traffic + '.*,SUBSTRING_INDEX(' + group_traffic + '.category_path,":",2) as "二级类目" FROM( ' \
                  + sql_relevance + ' ) pt_relevance INNER JOIN ' + group_traffic + \
                  ' ON pt_relevance.relation_traffic_id = ' + group_traffic + '.id'
    return sql_traffic


def group_traffic_add_page_sql(group_page_size, group_start_index, group_supplement_competitors):
    sql_traffic_add = 'SELECT ' + group_supplement_competitors + '.clue_asin as related_asin,' \
                      + group_supplement_competitors + '.*,SUBSTRING_INDEX(' + group_supplement_competitors + \
                      '.category_path,":",2) as "二级类目" FROM ( ' + group_asin_sql(group_page_size, group_start_index) \
                      + ' ) pt_asin LEFT JOIN ' + group_supplement_competitors + ' ON pt_asin.related_asin=' + \
                      group_supplement_competitors + '.clue_asin WHERE supplement_competitors.id>0'
    return sql_traffic_add


# 关联流量复用
def group_traffic_status_sql(original_database, new_database, new_status):
    sql_traffic_status = 'UPDATE ' + new_database + '.pt_product_get_group INNER JOIN ' + original_database + \
                         '.pt_product_get_group ON ' + new_database + '.pt_product_get_group.asin=' + \
                         original_database + '.pt_product_get_group.asin SET ' + new_database + \
                         '.pt_product_get_group.`status`=' + str(new_status) + ' WHERE ' + \
                         original_database + '.pt_product_get_group.`status`=1'
    return sql_traffic_status


def group_traffic_old_create_sql(original_table, new_table):
    sql_traffic_old_create = 'CREATE TABLE ' + new_table + ' LIKE ' + original_table
    return sql_traffic_old_create


def group_traffic_old_insert_sql(original_database, new_database, original_table, new_table, new_status):
    sql_traffic_old_insert = 'INSERT INTO ' + new_database + '.`' + new_table + \
                             '`(SELECT DISTINCT ' + original_database + '.' + original_table + '.* FROM ' + \
                             new_database + '.pt_product_get_group INNER JOIN ' + original_database + '.' + \
                             original_table + ' ON ' + new_database + '.pt_product_get_group.asin = ' + \
                             original_database + '.' + original_table + '.asin WHERE ' + \
                             new_database + '.pt_product_get_group.`status`=' + str(new_status) + ')'
    return sql_traffic_old_insert


def kw_ai_match(id_start, id_increment):
    sql_kw_ai_match = """
    SELECT DISTINCT
    	pt_clue_asin.asin,
    	clue_info.site,
    	clue_info.image,
    	clue_info.title 
    FROM
    	pt_clue_asin
    	INNER JOIN clue_info ON pt_clue_asin.asin = clue_info.ASIN 
    WHERE
    	pt_clue_asin.clue_status = 1 
    	AND pt_clue_asin.kw_status = 0 
    	AND (
    	LENGTH( clue_info.image )+ LENGTH( clue_info.title ))>0
    """
    sql_kw_match = sql_kw_ai_match + ' AND pt_clue_asin.id >=' + str(id_start) + \
                   ' LIMIT ' + str(id_increment)
    return sql_kw_match


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
