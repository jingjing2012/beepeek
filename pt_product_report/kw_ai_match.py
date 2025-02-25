import json
import pandas as pd
import requests

from conn import mysql_config as config, sql_engine
import pt_product_sql as sql
import pt_product_report_path as path
import pt_product_report_parameter as para


# API 调用函数
def call_api(api_url, params):
    """调用API"""
    try:
        headers = {
            'Content-Type': 'application/json',
            # 'Authorization': f'Bearer {API_KEY}'  # 如果需要认证
        }

        response = requests.post(api_url, json=params, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"API调用错误: {e}")
        return None


# ----------------------------------------------自动筛词---------------------------------------------------------

df_kw = sql_engine.connect_pt_product(config.sellersprite_hostname, config.sellersprite_password,
                                      config.clue_position_database, sql.sql_kw_ai_match)
if df_kw.empty:
    print('df_kw.empty')
else:
    clue_records = json.loads(df_kw.to_json(orient='records'))

    clue_results = []
    for clue_record in clue_records:
        params = {
            'asin': clue_record.get('asin'),
            'site': clue_record.get('site'),
            'image': clue_record.get('image'),
            'title': clue_record.get('title')
        }

        # 调用 API 获取数据
        api_data = call_api(para.api_url, params)

        if api_data is None:
            continue
        clue_result = api_data.get('result')

        # 数据清洗
        clue_df = pd.json_normalize(clue_result['keywords_result'])
        clue_df['asin'] = params.get('asin')
        # clue_results.append(clue_df)

        # df_clue_kw = pd.concat(clue_results)

        # 数据入库
        sql_engine.data_to_sql(clue_df, path.pt_clue_kw, 'append', config.connet_clue_position_db_sql)
    sql_engine.connect_product(config.sellersprite_hostname, config.sellersprite_password,
                               config.clue_position_database, sql.update_position_sql4)
