import json
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_fixed

from conn import mysql_config as config, sql_engine
import pt_product_sql as sql
import pt_product_report_path as path
import pt_product_report_parameter as para


# API 调用函数
# 设置重试策略：最多重试 3 次，每次间隔 600 秒（10 分钟）
@retry(stop=stop_after_attempt(3), wait=wait_fixed(600))
def call_api(api_url, params):
    """调用API"""
    try:
        headers = {
            'Content-Type': 'application/json',
            # 'Authorization': f'Bearer {API_KEY}'  # 如果需要认证
        }

        response = requests.post(api_url, json=params, headers=headers, timeout=600)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"API调用错误: {e}")
        return None


# ----------------------------------------------自动筛词---------------------------------------------------------

df_kw = sql_engine.connect_pt_product(config.sellersprite_hostname, config.sellersprite_password,
                                      config.clue_position_database, sql.sql_kw_ai_match)

# df_kw = df_kw[df_kw['site'] == 'de']

df_kw = df_kw.drop_duplicates()

if df_kw.empty:
    print('df_kw.empty')
else:
    clue_records = json.loads(df_kw.to_json(orient='records'))

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
        import pandas as pd

        # 假设 clue_result 是 API 或 JSON 解析的结果
        if clue_result.get('keywords_result') is None:
            print("⚠️ keywords_result 为空，无法创建 DataFrame")
            clue_df = pd.DataFrame()  # 创建空 DataFrame，防止代码崩溃
        else:
            clue_df = pd.DataFrame.from_records(clue_result['keywords_result'])

        # clue_df = pd.DataFrame.from_records(clue_result['keywords_result'])
        # clue_df = pd.json_normalize(clue_result['keywords_result'])   # pandas2旧方法，已不稳定，舍弃
        clue_df['asin'] = params.get('asin')
        clue_df['site'] = params.get('site')

        print(params.get('asin'))
        # print(clue_df.columns)

        # 数据入库
        sql_engine.data_to_sql(clue_df, path.pt_clue_kw, 'append', config.connet_clue_position_db_sql)
    sql_engine.connect_product(config.sellersprite_hostname, config.sellersprite_password,
                               config.clue_position_database, sql.update_position_sql4)
